#!/usr/bin/env python3
"""
计算 pareto metadata：从 sa-bench 结果 JSON 提取每个 concurrency 的 pareto 点。

用法:
    # 单个 run 目录
    python tools/compute_pareto_metadata.py /path/to/outputs/1192337-1p1d-dep4-staging/logs

    # 批量：扫描 outputs 下所有 run
    python tools/compute_pareto_metadata.py /path/to/outputs

    # 强制重新计算（覆盖已有 metadata）
    python tools/compute_pareto_metadata.py /path/to/outputs --force

输出:
    在每个 sa-bench_* 目录下生成 pareto_metadata.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


RESULT_FILE_PATTERN = re.compile(
    r"results_concurrency_(\d+)_gpus_(\d+)(?:_ctx_(\d+)_gen_(\d+))?\.json"
)

BENCH_DIR_PATTERN = re.compile(r"sa-bench_isl_(\d+)_osl_(\d+)")


def parse_result_file(filepath: Path) -> dict | None:
    """从单个 result JSON 提取关键指标。"""
    m = RESULT_FILE_PATTERN.match(filepath.name)
    if not m:
        return None

    concurrency = int(m.group(1))
    num_gpus = int(m.group(2))
    ctx_gpus = int(m.group(3)) if m.group(3) else None
    gen_gpus = int(m.group(4)) if m.group(4) else None

    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: skip {filepath.name}: {e}", file=sys.stderr)
        return None

    output_tps = data.get("output_throughput", 0)
    if output_tps <= 0:
        return None

    point = {
        "concurrency": concurrency,
        "num_gpus": num_gpus,
        "ctx_gpus": ctx_gpus,
        "gen_gpus": gen_gpus,
        "output_throughput": output_tps,
        "total_token_throughput": data.get("total_token_throughput", 0),
        "tps_per_user": output_tps / concurrency,
        "tps_per_gpu": output_tps / num_gpus,
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
        "p99_tpot_ms": data.get("p99_tpot_ms"),
        "mean_itl_ms": data.get("mean_itl_ms"),
        "p99_itl_ms": data.get("p99_itl_ms"),
        "mean_e2el_ms": data.get("mean_e2el_ms"),
        "p99_e2el_ms": data.get("p99_e2el_ms"),
        "completed": data.get("completed", 0),
        "num_prompts": data.get("num_prompts", 0),
    }
    return point


def compute_for_bench_dir(bench_dir: Path) -> dict | None:
    """为单个 sa-bench_* 目录计算 pareto metadata。"""
    dm = BENCH_DIR_PATTERN.match(bench_dir.name)
    isl = int(dm.group(1)) if dm else None
    osl = int(dm.group(2)) if dm else None

    result_files = sorted(bench_dir.glob("results_concurrency_*.json"))
    if not result_files:
        return None

    points = []
    model_id = None
    for rf in result_files:
        point = parse_result_file(rf)
        if point is None:
            continue
        points.append(point)
        if model_id is None:
            try:
                with open(rf) as f:
                    model_id = json.load(f).get("model_id")
            except Exception:
                pass

    if not points:
        return None

    points.sort(key=lambda p: p["concurrency"])

    metadata = {
        "model_id": model_id,
        "isl": isl,
        "osl": osl,
        "num_points": len(points),
        "points": points,
    }
    return metadata


def process_logs_dir(logs_dir: Path, force: bool = False) -> int:
    """处理单个 run 的 logs 目录，返回生成的 metadata 文件数。"""
    count = 0
    for bench_dir in sorted(logs_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        if not BENCH_DIR_PATTERN.match(bench_dir.name):
            continue

        out_path = bench_dir / "pareto_metadata.json"
        if out_path.exists() and not force:
            continue

        metadata = compute_for_bench_dir(bench_dir)
        if metadata is None:
            continue

        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
        count += 1
        print(f"  -> {out_path}")

    return count


def main():
    parser = argparse.ArgumentParser(description="计算 pareto metadata")
    parser.add_argument(
        "path",
        help="run 的 logs 目录，或包含多个 run 的 outputs 目录",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="强制重新计算"
    )
    args = parser.parse_args()

    target = Path(args.path).resolve()
    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        sys.exit(1)

    # 判断是单个 logs 目录还是 outputs 目录
    # 如果 target 下直接有 sa-bench_* 目录，就当作单个 logs 目录
    has_bench_dirs = any(
        BENCH_DIR_PATTERN.match(d.name)
        for d in target.iterdir()
        if d.is_dir()
    )

    total = 0
    if has_bench_dirs:
        print(f"Processing logs dir: {target}")
        total = process_logs_dir(target, args.force)
    else:
        # 扫描 outputs/*/logs/
        for run_dir in sorted(target.iterdir()):
            if not run_dir.is_dir():
                continue
            logs_dir = run_dir / "logs"
            if not logs_dir.is_dir():
                logs_dir = run_dir  # 有些 run 可能直接放在 run_dir 下
            has = any(
                BENCH_DIR_PATTERN.match(d.name)
                for d in logs_dir.iterdir()
                if d.is_dir()
            )
            if has:
                print(f"Processing: {run_dir.name}")
                total += process_logs_dir(logs_dir, args.force)

    print(f"\nDone. Generated {total} pareto_metadata.json file(s).")


if __name__ == "__main__":
    main()
