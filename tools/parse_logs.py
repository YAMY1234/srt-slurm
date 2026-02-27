#!/usr/bin/env python3
"""
Parse single run logs and generate metadata.yaml

Usage:
    python tools/parse_logs.py                    # parse all runs under logs/
    python tools/parse_logs.py --run-dir logs/2252_1A_20260106_015710-xxx  # parse a single run
    python tools/parse_logs.py --force            # force regenerate (overwrite existing metadata)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import yaml


def parse_server_args(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse server_args=ServerArgs(...) from log content
    This is the only source of parameters
    """
    # Match server_args=ServerArgs(...) format
    pattern = r"server_args=ServerArgs\((.*?)\)(?:\n|$)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return None

    args_str = match.group(1)
    result = {}

    # Parse key=value pairs, handle nested brackets and quotes
    current_key = None
    current_value = []
    depth = 0
    in_string = False
    string_char = None

    i = 0
    while i < len(args_str):
        char = args_str[i]

        # Handle string
        if char in ('"', "'") and (i == 0 or args_str[i - 1] != "\\"):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None

        # Handle bracket depth
        if not in_string:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1

        # Handle parameter separator
        if char == "," and depth == 0 and not in_string:
            if current_key:
                value_str = "".join(current_value).strip()
                result[current_key] = _parse_value(value_str)
            current_key = None
            current_value = []
            i += 1
            continue

        # Handle key=value
        if char == "=" and depth == 0 and not in_string and current_key is None:
            current_key = "".join(current_value).strip()
            current_value = []
            i += 1
            continue

        current_value.append(char)
        i += 1

    # Handle last parameter
    if current_key:
        value_str = "".join(current_value).strip()
        result[current_key] = _parse_value(value_str)

    return result


def _parse_value(value_str: str) -> Any:
    """Convert string value to Python type"""
    value_str = value_str.strip()

    if value_str == "None":
        return None
    if value_str == "True":
        return True
    if value_str == "False":
        return False

    # Try parsing as number
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Strip quotes
    if (value_str.startswith("'") and value_str.endswith("'")) or (
        value_str.startswith('"') and value_str.endswith('"')
    ):
        return value_str[1:-1]

    # Keep lists or dicts as-is (as string)
    if value_str.startswith("[") or value_str.startswith("{"):
        return value_str

    return value_str


class LogParser:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.config_path = run_dir / "config.yaml"
        self.metadata_path = run_dir / "metadata.yaml"

        # Detect log file location: may be in run_dir root or run_dir/logs subdir
        self.logs_subdir = self._detect_logs_location()

        # benchmark.out location
        if self.logs_subdir:
            self.benchmark_path = self.logs_subdir / "benchmark.out"
        else:
            self.benchmark_path = run_dir / "benchmark.out"

        # Determine agg or disagg mode
        self.mode = self._detect_mode()

    def _detect_logs_location(self) -> Optional[Path]:
        """
        Detect log file location
        Return logs subdir path, or None if logs are in root dir
        """
        logs_subdir = self.run_dir / "logs"

        # Check if logs subdir exists and contains log files
        if logs_subdir.exists() and logs_subdir.is_dir():
            # Check for _w0.out or _w1.out worker log files
            for f in logs_subdir.iterdir():
                if f.name.endswith("_w0.out") or f.name.endswith("_w1.out"):
                    return logs_subdir

        # Otherwise logs are in root dir
        return None

    def _detect_mode(self) -> str:
        """Detect run mode: agg or disagg"""
        # Infer from dir name: _1A_ means agg, _P_D_ means disagg
        dir_name = self.run_dir.name
        if "_1A_" in dir_name or "_2A_" in dir_name:
            return "agg"
        elif "_P_" in dir_name and "_D_" in dir_name:
            return "disagg"

        # Infer from file existence
        # Determine scan directory (root or logs subdir)
        scan_dir = self.logs_subdir if self.logs_subdir else self.run_dir

        for f in scan_dir.iterdir():
            if f.name.endswith("_agg_w0.out"):
                return "agg"
            if f.name.endswith("_prefill_w0.out") or f.name.endswith("_decode_w0.out"):
                return "disagg"

        return "unknown"

    def _find_server_logs(self) -> Dict[str, Any]:
        """Find server log files (supports multiple workers)"""
        logs = {}

        # Determine scan directory (root or logs subdir)
        scan_dir = self.logs_subdir if self.logs_subdir else self.run_dir

        prefill_logs = []
        decode_logs = []

        for f in sorted(scan_dir.iterdir()):
            if f.is_file():
                # Match _agg_w*.out
                if "_agg_w" in f.name and f.name.endswith(".out"):
                    if "agg" not in logs:
                        logs["agg"] = []
                    logs["agg"].append(f)
                # Match _prefill_w*.out
                elif "_prefill_w" in f.name and f.name.endswith(".out"):
                    prefill_logs.append(f)
                # Match _decode_w*.out
                elif "_decode_w" in f.name and f.name.endswith(".out"):
                    decode_logs.append(f)

        if prefill_logs:
            logs["prefill"] = prefill_logs
        if decode_logs:
            logs["decode"] = decode_logs

        return logs

    def parse(self) -> Dict[str, Any]:
        """Parse run directory, return metadata"""
        metadata = {
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "parsed_at": datetime.now().isoformat(),
            "mode": self.mode,
        }

        # Find server logs
        server_logs = self._find_server_logs()

        if not server_logs:
            metadata["error"] = "No server log found"
            return metadata

        # Parse based on mode
        if self.mode == "agg":
            metadata["server_args"] = self._parse_agg_mode(server_logs)
        elif self.mode == "disagg":
            metadata["prefill_args"] = self._parse_worker_log(server_logs.get("prefill"))
            metadata["decode_args"] = self._parse_worker_log(server_logs.get("decode"))
        else:
            # Try parsing any existing log
            for log_type, log_path in server_logs.items():
                metadata[f"{log_type}_args"] = self._parse_worker_log(log_path)

        # Parse performance stats
        metadata["server_stats"] = self._parse_all_stats(server_logs)

        # Parse benchmark.out for Score
        metadata["benchmark_results"] = self._parse_benchmark()

        # Supplement missing info from config.yaml
        if self.config_path.exists():
            metadata["config_fallback"] = self._parse_config_fallback()

        return metadata

    def _parse_agg_mode(self, server_logs: Dict[str, Path]) -> Dict[str, Any]:
        """Parse aggregated mode logs"""
        if "agg" not in server_logs:
            return {"error": "agg log not found"}

        return self._parse_worker_log(server_logs["agg"])

    def _parse_worker_log(self, log_path: Any) -> Dict[str, Any]:
        """Parse worker logs, extract ServerArgs (supports single or multiple workers)"""
        # Support single Path or list of Paths
        if log_path is None:
            return {"error": "log file not found"}

        # If list, take first (all workers should share same config)
        if isinstance(log_path, list):
            if not log_path:
                return {"error": "log file not found"}
            actual_path = log_path[0]
            worker_count = len(log_path)
        else:
            actual_path = log_path
            worker_count = 1

        if not actual_path.exists():
            return {"error": "log file not found"}

        try:
            with open(actual_path, "r", errors="ignore") as f:
                content = f.read()

            server_args = parse_server_args(content)
            if server_args is None:
                return {"error": "ServerArgs not found in log"}

            # Extract key parameters
            result = self._extract_key_params(server_args)
            result["_raw_count"] = len(server_args)
            result["_worker_count"] = worker_count

            return result

        except Exception as e:
            return {"error": str(e)}

    def _extract_key_params(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key parameters from ServerArgs"""
        key_fields = [
            # Model info
            "model_path",
            "served_model_name",
            "context_length",
            # Quantization
            "quantization",
            "kv_cache_dtype",
            # Parallelism config
            "tp_size",
            "pp_size",
            "dp_size",
            "ep_size",
            # Attention
            "attention_backend",
            "enable_dp_attention",
            # Memory
            "mem_fraction_static",
            "chunked_prefill_size",
            # Speculative decoding
            "speculative_algorithm",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
            # MoE
            "moe_runner_backend",
            "moe_dense_tp_size",
            "moe_a2a_backend",
            # Other important config
            "disable_radix_cache",
            "disable_overlap_schedule",
            "enable_symm_mem",
            "enable_torch_symm_mem",
            "enable_flashinfer_allreduce_fusion",
            # Disaggregation
            "disaggregation_mode",
            "disaggregation_transfer_backend",
            # Scheduling
            "max_running_requests",
            "schedule_conservativeness",
        ]

        result = {}
        for field in key_fields:
            if field in args:
                result[field] = args[field]

        return result

    def _parse_all_stats(self, server_logs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse performance stats from all logs (supports multiple workers)"""
        stats = {}

        for log_type, log_path in server_logs.items():
            if log_path:
                # Support single log or list of logs
                if isinstance(log_path, list):
                    # Multiple workers: merge all stats
                    log_stats = self._parse_multiple_log_stats(log_path)
                else:
                    # Single worker
                    if log_path.exists():
                        log_stats = self._parse_log_stats(log_path)
                    else:
                        log_stats = {"error": "log file not found"}

                stats[log_type] = log_stats

        return stats

    def _parse_log_stats(self, log_path: Path) -> Dict[str, Any]:
        """Parse performance stats from a single log file"""
        try:
            with open(log_path, "r", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            return {"error": str(e)}

        prefill_stats = []
        decode_stats = []

        # Regex patterns
        prefill_pattern = re.compile(
            r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
            r"token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)"
        )

        # Decode batch has two formats:
        # agg mode: Decode batch, #running-req: 14, #token: 785280, token usage: 0.27, accept len: 2.47, accept rate: 0.82, cuda graph: True, gen throughput (token/s): 167.86, #queue-req: 0,
        # disagg mode: Decode batch, #running-req: 1, #token: 41536, token usage: 0.01, accept len: 2.20, accept rate: 0.73, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 1, #retracted-req: 0, cuda graph: True, gen throughput (token/s): 0.40, #queue-req: 0,
        decode_pattern = re.compile(
            r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), "
            r"accept len: ([\d.]+), accept rate: ([\d.]+), "
            r"(?:pre-allocated usage: [\d.]+, #prealloc-req: \d+, #transfer-req: \d+, #retracted-req: \d+, )?"  # optional disagg fields
            r"cuda graph: \w+, "
            r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
        )

        for line in lines:
            # Parse Prefill
            match = prefill_pattern.search(line)
            if match:
                prefill_stats.append(
                    {
                        "new_seq": int(match.group(1)),
                        "new_token": int(match.group(2)),
                        "cached_token": int(match.group(3)),
                        "token_usage": float(match.group(4)),
                        "running_req": int(match.group(5)),
                        "queue_req": int(match.group(6)),
                    }
                )
                continue

            # Parse Decode
            match = decode_pattern.search(line)
            if match:
                decode_stats.append(
                    {
                        "running_req": int(match.group(1)),
                        "token": int(match.group(2)),
                        "token_usage": float(match.group(3)),
                        "accept_len": float(match.group(4)),
                        "accept_rate": float(match.group(5)),
                        "gen_throughput": float(match.group(6)),
                        "queue_req": int(match.group(7)),
                    }
                )

        result = {
            "prefill_count": len(prefill_stats),
            "decode_count": len(decode_stats),
        }

        # Compute Prefill stats
        if prefill_stats:
            result["prefill"] = self._compute_stats(
                prefill_stats, ["new_seq", "new_token", "cached_token", "token_usage", "running_req", "queue_req"]
            )

        # Compute Decode stats
        if decode_stats:
            result["decode"] = self._compute_stats(
                decode_stats,
                ["running_req", "token", "token_usage", "accept_len", "accept_rate", "gen_throughput", "queue_req"],
            )

        return result

    def _parse_multiple_log_stats(self, log_paths: List[Path]) -> Dict[str, Any]:
        """Parse and merge stats from multiple worker logs"""
        all_prefill_stats = []
        all_decode_stats = []

        for log_path in log_paths:
            if not log_path.exists():
                continue

            try:
                with open(log_path, "r", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue

            # Regex patterns
            prefill_pattern = re.compile(
                r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
                r"token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)"
            )

            decode_pattern = re.compile(
                r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), "
                r"accept len: ([\d.]+), accept rate: ([\d.]+), "
                r"(?:pre-allocated usage: [\d.]+, #prealloc-req: \d+, #transfer-req: \d+, #retracted-req: \d+, )?"
                r"cuda graph: \w+, "
                r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
            )

            for line in lines:
                # Parse Prefill
                match = prefill_pattern.search(line)
                if match:
                    all_prefill_stats.append(
                        {
                            "new_seq": int(match.group(1)),
                            "new_token": int(match.group(2)),
                            "cached_token": int(match.group(3)),
                            "token_usage": float(match.group(4)),
                            "running_req": int(match.group(5)),
                            "queue_req": int(match.group(6)),
                        }
                    )
                    continue

                # Parse Decode
                match = decode_pattern.search(line)
                if match:
                    all_decode_stats.append(
                        {
                            "running_req": int(match.group(1)),
                            "token": int(match.group(2)),
                            "token_usage": float(match.group(3)),
                            "accept_len": float(match.group(4)),
                            "accept_rate": float(match.group(5)),
                            "gen_throughput": float(match.group(6)),
                            "queue_req": int(match.group(7)),
                        }
                    )

        result = {
            "prefill_count": len(all_prefill_stats),
            "decode_count": len(all_decode_stats),
            "worker_count": len(log_paths),
        }

        # Compute merged Prefill stats
        if all_prefill_stats:
            result["prefill"] = self._compute_stats(
                all_prefill_stats, ["new_seq", "new_token", "cached_token", "token_usage", "running_req", "queue_req"]
            )

        # Compute merged Decode stats
        if all_decode_stats:
            result["decode"] = self._compute_stats(
                all_decode_stats,
                ["running_req", "token", "token_usage", "accept_len", "accept_rate", "gen_throughput", "queue_req"],
            )

        return result

    def _compute_stats(self, data: List[Dict], fields: List[str]) -> Dict[str, Dict]:
        """Compute stats (mean, max, min) for specified fields"""
        stats = {}
        for field in fields:
            values = [d[field] for d in data if field in d]
            if values:
                stats[field] = {
                    "avg": round(sum(values) / len(values), 4),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values),
                }
        return stats

    def _parse_benchmark(self) -> Dict[str, Any]:
        """Parse benchmark.out for evaluation results"""
        result = {}

        if not self.benchmark_path.exists():
            result["error"] = "benchmark.out not found"
            return result

        try:
            with open(self.benchmark_path, "r", errors="ignore") as f:
                content = f.read()

            # Match Score: 0.550 (legacy format)
            score_match = re.search(r"^Score:\s*([\d.]+)", content, re.MULTILINE)
            if score_match:
                result["score"] = float(score_match.group(1))

            # Match Total latency: 1249.009 s
            latency_match = re.search(r"Total latency:\s*([\d.]+)\s*s", content)
            if latency_match:
                result["total_latency_s"] = float(latency_match.group(1))

            # Match detailed score dict (legacy format)
            # {'chars': np.float64(5731.87...), ..., 'score': np.float64(0.5498...)}
            score_dict_match = re.search(r"'score':\s*np\.float64\(([\d.]+)\)", content)
            if score_dict_match:
                result["score_detailed"] = float(score_dict_match.group(1))

            # difficulty_easy
            easy_match = re.search(r"'difficulty_easy':\s*np\.float64\(([\d.]+)\)", content)
            if easy_match:
                result["difficulty_easy"] = float(easy_match.group(1))

            # difficulty_hard
            hard_match = re.search(r"'difficulty_hard':\s*np\.float64\(([\d.]+)\)", content)
            if hard_match:
                result["difficulty_hard"] = float(hard_match.group(1))

            # GPQA format:[METRIC] gpqa_mean_score=0.7689393939393939
            gpqa_score_match = re.search(r"\[METRIC\]\s+gpqa_mean_score=([\d.]+)", content)
            if gpqa_score_match:
                result["gpqa_mean_score"] = float(gpqa_score_match.group(1))

            # GPQA: Repeat: 8, mean: 0.769
            gpqa_mean_match = re.search(r"Repeat:\s*(\d+),\s*mean:\s*([\d.]+)", content)
            if gpqa_mean_match:
                result["repeat"] = int(gpqa_mean_match.group(1))
                result["mean_score"] = float(gpqa_mean_match.group(2))

            # GPQA: Scores: ['0.773', '0.758', '0.803', ...]
            gpqa_scores_match = re.search(r"Scores:\s*\[(.*?)\]", content)
            if gpqa_scores_match:
                scores_str = gpqa_scores_match.group(1)
                scores = [float(s.strip().strip("'\"")) for s in scores_str.split(",")]
                result["scores"] = scores

            # GPQA: 'mean_score': np.float64(0.7689393939393939)
            gpqa_mean_detailed = re.search(r"'mean_score':\s*np\.float64\(([\d.]+)\)", content)
            if gpqa_mean_detailed:
                result["mean_score_detailed"] = float(gpqa_mean_detailed.group(1))

            # GPQA: 'chars': np.float64(23841.065656565657)
            chars_match = re.search(r"'chars':\s*np\.float64\(([\d.]+)\)", content)
            if chars_match:
                result["avg_chars"] = float(chars_match.group(1))

            return result

        except Exception as e:
            result["error"] = str(e)
            return result

    def _parse_config_fallback(self) -> Dict[str, Any]:
        """Get supplementary info from config.yaml"""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            result = {}

            # Basic info
            result["name"] = config.get("name", "")

            # Benchmark config
            benchmark_config = config.get("benchmark", {})
            result["benchmark_type"] = benchmark_config.get("type", "")

            # Resource config
            resources = config.get("resources", {})
            result["agg_nodes"] = resources.get("agg_nodes")
            result["agg_workers"] = resources.get("agg_workers")
            result["prefill_nodes"] = resources.get("prefill_nodes")
            result["prefill_workers"] = resources.get("prefill_workers")
            result["decode_nodes"] = resources.get("decode_nodes")
            result["decode_workers"] = resources.get("decode_workers")
            result["gpu_type"] = resources.get("gpu_type")
            result["gpus_per_node"] = resources.get("gpus_per_node")

            # Model config
            model = config.get("model", {})
            result["model_path_config"] = model.get("path", "")
            result["model_precision"] = model.get("precision", "")

            # Extract MTP version from environment variables
            result["mtp_version"] = self._extract_mtp_version(config)

            return result

        except Exception as e:
            return {"error": str(e)}

    def _extract_mtp_version(self, config: Dict[str, Any]) -> str:
        """Extract MTP version from config environment variables"""
        backend = config.get("backend", {})

        # Check decode_environment, prefill_environment, aggregated_environment
        for env_key in ["decode_environment", "prefill_environment", "aggregated_environment"]:
            env = backend.get(env_key, {})
            if env and isinstance(env, dict):
                # Check SGLANG_ENABLE_SPEC_V2
                if env.get("SGLANG_ENABLE_SPEC_V2") == "1":
                    return "V2"

        # If speculative_algorithm exists but no SGLANG_ENABLE_SPEC_V2, it's V1
        # Check if speculative is enabled in sglang_config
        sglang_config = backend.get("sglang_config", {})
        for worker_key in ["decode", "prefill", "aggregated"]:
            worker_config = sglang_config.get(worker_key, {})
            if worker_config and isinstance(worker_config, dict):
                if worker_config.get("speculative-algorithm"):
                    return "V1"

        return ""

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to file"""
        with open(self.metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"Saved: {self.metadata_path}")


def find_run_dirs(logs_dir: Path) -> List[Path]:
    """Find all run directories"""
    run_dirs = []
    for item in logs_dir.iterdir():
        if item.is_dir():
            # Check for server log or config.yaml
            # Logs may be in root dir or logs/ subdir
            has_log_in_root = any(f.name.endswith("_w0.out") for f in item.iterdir() if f.is_file())

            # Check logs subdir
            logs_subdir = item / "logs"
            has_log_in_subdir = False
            if logs_subdir.exists() and logs_subdir.is_dir():
                has_log_in_subdir = any(
                    f.name.endswith("_w0.out") or f.name.endswith("_w1.out")
                    for f in logs_subdir.iterdir()
                    if f.is_file()
                )

            has_config = (item / "config.yaml").exists()

            if has_log_in_root or has_log_in_subdir or has_config:
                run_dirs.append(item)

    return sorted(run_dirs)


def main():
    parser = argparse.ArgumentParser(description="Parse run logs and generate metadata")
    parser.add_argument("--run-dir", "-r", help="Specify a single run directory")
    parser.add_argument("--logs-dir", "-d", default="logs", help="logs directory path")
    parser.add_argument("--force", "-f", action="store_true", help="Force regenerate (overwrite existing)")

    args = parser.parse_args()

    # Determine directories to process
    project_root = Path(__file__).parent.parent

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        # Support absolute and relative paths
        logs_path = Path(args.logs_dir)
        if logs_path.is_absolute():
            logs_dir = logs_path
        else:
            logs_dir = project_root / args.logs_dir
        if not logs_dir.exists():
            print(f"Log directory not found: {logs_dir}")
            sys.exit(1)
        run_dirs = find_run_dirs(logs_dir)

    if not run_dirs:
        print("No run directories found")
        sys.exit(0)

    print(f"Found {len(run_dirs)}  run directories")
    print("=" * 60)

    parsed_count = 0
    skipped_count = 0

    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.yaml"

        if metadata_path.exists() and not args.force:
            print(f"Skipping (already exists): {run_dir.name}")
            skipped_count += 1
            continue

        print(f"Parsing: {run_dir.name}")
        try:
            parser_obj = LogParser(run_dir)
            metadata = parser_obj.parse()
            parser_obj.save_metadata(metadata)
            parsed_count += 1
        except Exception as e:
            print(f"Parse failed: {e}")

    print("=" * 60)
    print(f"Done: parsed {parsed_count} , skipped {skipped_count} ")


if __name__ == "__main__":
    main()
