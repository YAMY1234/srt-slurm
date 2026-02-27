#!/usr/bin/env python3
"""
Export benchmark runs to Excel (incremental update)

Usage:
    # Default: export outputs directory to benchmark_summary.xlsx (incremental)
    python analysis/export_to_excel.py

    # Force full refresh (ignore existing Excel)
    python analysis/export_to_excel.py --full

    # Export additional directories
    python analysis/export_to_excel.py --output-dir outputs outputs-old

This script loads benchmark runs and exports them to an Excel file.
By default, it performs incremental updates - only adding new runs that
don't already exist in the Excel file.

Excel sheets:
    - Summary: One row per run with key config and best result
    - All Results: One row per concurrency level per run
    - Config Details: Detailed configuration for each run
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from analysis.srtlog import RunLoader
from analysis.srtlog.models import BenchmarkRun

# Default Excel filename (fixed, no timestamp for incremental updates)
DEFAULT_EXCEL_FILE = "benchmark_summary.xlsx"


def detect_mtp_info(run: BenchmarkRun) -> tuple[bool, int]:
    """Detect MTP (Multi-Token Prediction / speculative decoding) configuration.

    Detection methods (in priority order):
    1. Parse _mtpX from directory name (X > 0 means enabled)
    2. Parse _mtpX from config.yaml 'name' field
    3. Check config.yaml decode section for speculative-num-steps

    Args:
        run: BenchmarkRun instance

    Returns:
        Tuple of (mtp_enabled: bool, mtp_num_steps: int)
        Returns (False, 0) if MTP is not detected
    """
    run_path = run.metadata.path
    job_name = os.path.basename(run_path)

    # Method 1: Parse _mtpX from directory name
    match = re.search(r"[_-]mtp(\d+)", job_name, re.IGNORECASE)
    if match:
        num_steps = int(match.group(1))
        return (num_steps > 0, num_steps)

    # Method 2 & 3: Check config.yaml
    config_path = os.path.join(run_path, "config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

            if config:
                # Method 2: Parse _mtpX from config name field
                config_name = config.get("name", "")
                if config_name:
                    match = re.search(r"[_-]mtp(\d+)", config_name, re.IGNORECASE)
                    if match:
                        num_steps = int(match.group(1))
                        return (num_steps > 0, num_steps)

                # Method 3: Check decode sglang_config for speculative settings
                decode_config = config.get("backend", {}).get("sglang_config", {}).get("decode", {})
                if decode_config:
                    spec_steps = decode_config.get("speculative-num-steps", 0)
                    spec_algo = decode_config.get("speculative-algorithm", "")
                    if spec_steps and int(spec_steps) > 0:
                        return (True, int(spec_steps))
                    if spec_algo:
                        return (True, int(spec_steps) if spec_steps else 0)
        except Exception:
            pass

    return (False, 0)


def get_existing_job_ids(excel_file: str) -> set[str]:
    """Read existing job IDs from Excel file.

    Args:
        excel_file: Path to existing Excel file

    Returns:
        Set of job IDs already in the Excel
    """
    if not os.path.exists(excel_file):
        return set()

    try:
        df = pd.read_excel(excel_file, sheet_name="Summary")
        # Convert to string to handle both int and string job IDs
        return set(df["Job ID"].astype(str).tolist())
    except Exception as e:
        print(f"Warning: Could not read existing Excel file: {e}")
        return set()


def get_best_result(run: BenchmarkRun) -> dict:
    """Get the best benchmark result for a run (highest Output TPS)."""
    if not run.profiler.output_tps:
        return {}

    # Find the index with max output TPS
    max_idx = 0
    max_tps = 0
    for i, tps in enumerate(run.profiler.output_tps):
        if tps and tps > max_tps:
            max_tps = tps
            max_idx = i

    total_gpus = run.total_gpus
    tps = run.profiler.output_tps[max_idx] if max_idx < len(run.profiler.output_tps) else 0
    tps_per_gpu = tps / total_gpus if total_gpus > 0 else 0

    # Calculate TPS/User from TPOT
    tpot = run.profiler.mean_tpot_ms[max_idx] if max_idx < len(run.profiler.mean_tpot_ms) else None
    tps_per_user = 1000 / tpot if tpot and tpot > 0 else 0

    return {
        "Best Concurrency": run.profiler.concurrency_values[max_idx]
        if max_idx < len(run.profiler.concurrency_values)
        else None,
        "Best Output TPS": tps,
        "Best Output TPS/GPU": tps_per_gpu,
        "Best Output TPS/User": tps_per_user,
        "Best Mean TTFT (ms)": run.profiler.mean_ttft_ms[max_idx] if max_idx < len(run.profiler.mean_ttft_ms) else None,
        "Best Mean TPOT (ms)": tpot,
        "Best Mean ITL (ms)": run.profiler.mean_itl_ms[max_idx] if max_idx < len(run.profiler.mean_itl_ms) else None,
    }


def create_summary_df(runs: list[BenchmarkRun]) -> pd.DataFrame:
    """Create summary DataFrame with one row per run."""
    rows = []

    for run in runs:
        # Extract job name from directory path
        job_name = os.path.basename(run.metadata.path)

        # Detect MTP configuration
        mtp_enabled, mtp_steps = detect_mtp_info(run)

        # Basic config
        row = {
            "Job ID": run.job_id,
            "Job Name": job_name,
            "Run Date": run.metadata.run_date,
            "GPU Type": run.metadata.gpu_type,
            "GPUs/Node": run.metadata.gpus_per_node,
            "Total GPUs": run.total_gpus,
            "Topology": run.metadata.topology_label,
            "Prefill Nodes": run.metadata.prefill_nodes,
            "Decode Nodes": run.metadata.decode_nodes,
            "Prefill Workers": run.metadata.prefill_workers,
            "Decode Workers": run.metadata.decode_workers,
            "Agg Workers": run.metadata.agg_workers,
            "Mode": run.metadata.mode,
            "MTP Enabled": mtp_enabled,
            "MTP Steps": mtp_steps,
            "ISL": run.profiler.isl,
            "OSL": run.profiler.osl,
            "Profiler": run.profiler.profiler_type,
            "Container": run.metadata.container,
            "Is Complete": run.is_complete,
            "Tags": ", ".join(run.tags) if run.tags else "",
        }

        # Add best result metrics
        best_result = get_best_result(run)
        row.update(best_result)

        rows.append(row)

    return pd.DataFrame(rows)


def create_all_results_df(runs: list[BenchmarkRun]) -> pd.DataFrame:
    """Create DataFrame with all benchmark results (one row per concurrency per run)."""
    rows = []

    for run in runs:
        job_name = os.path.basename(run.metadata.path)
        total_gpus = run.total_gpus
        mtp_enabled, mtp_steps = detect_mtp_info(run)

        for i in range(len(run.profiler.output_tps)):
            tps = run.profiler.output_tps[i]
            tps_per_gpu = tps / total_gpus if total_gpus > 0 else 0

            # Calculate TPS/User from TPOT
            tpot = run.profiler.mean_tpot_ms[i] if i < len(run.profiler.mean_tpot_ms) else None
            tps_per_user = 1000 / tpot if tpot and tpot > 0 else 0

            # Get total TPS
            total_tps = run.profiler.total_tps[i] if i < len(run.profiler.total_tps) else None
            total_tps_per_gpu = total_tps / total_gpus if total_tps and total_gpus > 0 else None

            row = {
                "Job ID": run.job_id,
                "Job Name": job_name,
                "Run Date": run.metadata.run_date,
                "Topology": run.metadata.topology_label,
                "MTP Enabled": mtp_enabled,
                "MTP Steps": mtp_steps,
                "ISL": run.profiler.isl,
                "OSL": run.profiler.osl,
                "GPU Type": run.metadata.gpu_type,
                "Total GPUs": total_gpus,
                "Concurrency": run.profiler.concurrency_values[i] if i < len(run.profiler.concurrency_values) else None,
                "Request Rate": run.profiler.request_rate[i] if i < len(run.profiler.request_rate) else None,
                "Output TPS": tps,
                "Output TPS/GPU": tps_per_gpu,
                "Output TPS/User": tps_per_user,
                "Total TPS": total_tps,
                "Total TPS/GPU": total_tps_per_gpu,
                "Mean TTFT (ms)": run.profiler.mean_ttft_ms[i] if i < len(run.profiler.mean_ttft_ms) else None,
                "Mean TPOT (ms)": tpot,
                "Mean ITL (ms)": run.profiler.mean_itl_ms[i] if i < len(run.profiler.mean_itl_ms) else None,
                "Mean E2EL (ms)": run.profiler.mean_e2el_ms[i] if i < len(run.profiler.mean_e2el_ms) else None,
                "Median TTFT (ms)": run.profiler.median_ttft_ms[i] if i < len(run.profiler.median_ttft_ms) else None,
                "Median TPOT (ms)": run.profiler.median_tpot_ms[i] if i < len(run.profiler.median_tpot_ms) else None,
                "P99 TTFT (ms)": run.profiler.p99_ttft_ms[i] if i < len(run.profiler.p99_ttft_ms) else None,
                "P99 TPOT (ms)": run.profiler.p99_tpot_ms[i] if i < len(run.profiler.p99_tpot_ms) else None,
                "P99 ITL (ms)": run.profiler.p99_itl_ms[i] if i < len(run.profiler.p99_itl_ms) else None,
                "Request Throughput": run.profiler.request_throughput[i]
                if i < len(run.profiler.request_throughput)
                else None,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def create_config_details_df(runs: list[BenchmarkRun]) -> pd.DataFrame:
    """Create DataFrame with detailed configuration for each run."""
    rows = []

    for run in runs:
        job_name = os.path.basename(run.metadata.path)
        mtp_enabled, mtp_steps = detect_mtp_info(run)

        row = {
            "Job ID": run.job_id,
            "Job Name": job_name,
            "Run Date": run.metadata.run_date,
            "Mode": run.metadata.mode,
            "GPU Type": run.metadata.gpu_type,
            "GPUs per Node": run.metadata.gpus_per_node,
            "Prefill Nodes": run.metadata.prefill_nodes,
            "Decode Nodes": run.metadata.decode_nodes,
            "Prefill Workers": run.metadata.prefill_workers,
            "Decode Workers": run.metadata.decode_workers,
            "Agg Nodes": run.metadata.agg_nodes,
            "Agg Workers": run.metadata.agg_workers,
            "Total GPUs": run.total_gpus,
            "MTP Enabled": mtp_enabled,
            "MTP Steps": mtp_steps,
            "Container": run.metadata.container,
            "Model Dir": run.metadata.model_dir,
            "Profiler Type": run.profiler.profiler_type,
            "ISL": run.profiler.isl,
            "OSL": run.profiler.osl,
            "Concurrencies Config": run.profiler.concurrencies,
            "Request Rate Config": run.profiler.req_rate,
            "Is Complete": run.is_complete,
            "Missing Concurrencies": ", ".join(map(str, run.missing_concurrencies))
            if run.missing_concurrencies
            else "",
            "Tags": ", ".join(run.tags) if run.tags else "",
            "Path": run.metadata.path,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def load_existing_data(excel_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load existing data from Excel file.

    Returns:
        Tuple of (summary_df, all_results_df, config_df) or empty DataFrames if file doesn't exist
    """
    if not os.path.exists(excel_file):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        summary_df = pd.read_excel(excel_file, sheet_name="Summary")
        all_results_df = pd.read_excel(excel_file, sheet_name="All Results")
        config_df = pd.read_excel(excel_file, sheet_name="Config Details")
        return summary_df, all_results_df, config_df
    except Exception as e:
        print(f"Warning: Could not read existing Excel file: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def adjust_column_widths(writer: pd.ExcelWriter) -> None:
    """Auto-adjust column widths for all sheets."""
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        for column_cells in worksheet.columns:
            max_length = 0
            column = column_cells[0].column_letter
            for cell in column_cells:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50
            worksheet.column_dimensions[column].width = adjusted_width


def export_to_excel(logs_dirs: list[str], output_file: str, full_refresh: bool = False) -> None:
    """Export benchmark runs to Excel file with incremental updates.

    Args:
        logs_dirs: List of paths to directories containing benchmark run subdirectories
        output_file: Path to output Excel file
        full_refresh: If True, ignore existing data and do full export
    """
    # Get existing job IDs for incremental update
    existing_job_ids = set() if full_refresh else get_existing_job_ids(output_file)

    if existing_job_ids:
        print(f"Found {len(existing_job_ids)} existing runs in Excel")

    # Load runs from directories
    all_runs = []
    new_runs = []
    total_skipped = []

    for logs_dir in logs_dirs:
        print(f"Scanning: {logs_dir}")

        loader = RunLoader(logs_dir)
        runs, skipped = loader.load_all_with_skipped()

        all_runs.extend(runs)

        # Filter to only new runs
        for run in runs:
            if str(run.job_id) not in existing_job_ids:
                new_runs.append(run)

        total_skipped.extend(skipped)

    print(f"\nFound {len(all_runs)} total runs, {len(new_runs)} new runs to add")

    if not new_runs and not full_refresh:
        print("No new runs to add. Excel is up to date.")
        return

    if not all_runs:
        print("No runs found with benchmark data!")
        return

    # For incremental update, load existing data and append
    if not full_refresh and os.path.exists(output_file):
        print(f"Performing incremental update...")
        existing_summary, existing_results, existing_config = load_existing_data(output_file)

        # Create DataFrames for new runs only
        new_summary_df = create_summary_df(new_runs)
        new_results_df = create_all_results_df(new_runs)
        new_config_df = create_config_details_df(new_runs)

        # Concatenate with existing data
        summary_df = pd.concat([existing_summary, new_summary_df], ignore_index=True)
        all_results_df = pd.concat([existing_results, new_results_df], ignore_index=True)
        config_df = pd.concat([existing_config, new_config_df], ignore_index=True)

        print(f"  Added {len(new_runs)} new runs")
    else:
        # Full refresh - use all runs
        print(f"Performing full export...")
        summary_df = create_summary_df(all_runs)
        all_results_df = create_all_results_df(all_runs)
        config_df = create_config_details_df(all_runs)

    # Ensure Job ID is string type for consistent sorting
    summary_df["Job ID"] = summary_df["Job ID"].astype(str)
    all_results_df["Job ID"] = all_results_df["Job ID"].astype(str)
    config_df["Job ID"] = config_df["Job ID"].astype(str)

    # Sort by job ID (descending - newest first, treating as numeric for proper ordering)
    summary_df = summary_df.sort_values("Job ID", ascending=False, key=lambda x: x.astype(int))
    all_results_df = all_results_df.sort_values(
        ["Job ID", "Concurrency"], ascending=[False, True], key=lambda x: x.astype(int) if x.name == "Job ID" else x
    )
    config_df = config_df.sort_values("Job ID", ascending=False, key=lambda x: x.astype(int))

    # Export to Excel
    print(f"\nWriting to: {output_file}")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        all_results_df.to_excel(writer, sheet_name="All Results", index=False)
        config_df.to_excel(writer, sheet_name="Config Details", index=False)
        adjust_column_widths(writer)

    print(f"\nExcel export complete!")
    print(f"  - Summary: {len(summary_df)} runs")
    print(f"  - All Results: {len(all_results_df)} data points")
    print(f"  - Config Details: {len(config_df)} configurations")


def main():
    parser = argparse.ArgumentParser(
        description="Export benchmark runs to Excel (incremental update)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: incremental update from outputs directory
    python analysis/export_to_excel.py
    
    # Force full refresh (ignore existing data)
    python analysis/export_to_excel.py --full
    
    # Export from multiple directories
    python analysis/export_to_excel.py --output-dir outputs outputs-old
    
    # Custom Excel filename
    python analysis/export_to_excel.py --excel-file my_results.xlsx
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        nargs="+",
        default=["outputs"],
        help="Directory/directories containing benchmark runs (default: outputs)",
    )
    parser.add_argument(
        "--excel-file",
        type=str,
        default=DEFAULT_EXCEL_FILE,
        help=f"Output Excel filename (default: {DEFAULT_EXCEL_FILE})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full refresh, ignoring existing Excel data",
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.parent
    logs_dirs = []

    for output_dir in args.output_dir:
        logs_dir = script_dir / output_dir

        if not logs_dir.exists():
            # Try absolute path
            logs_dir = Path(output_dir)
            if not logs_dir.exists():
                print(f"Warning: Directory not found, skipping: {output_dir}")
                continue

        logs_dirs.append(str(logs_dir))

    if not logs_dirs:
        print("Error: No valid output directories found!")
        sys.exit(1)

    # Resolve output file path
    output_file = args.excel_file
    if not os.path.isabs(output_file):
        output_file = str(script_dir / output_file)

    export_to_excel(logs_dirs, output_file, full_refresh=args.full)


if __name__ == "__main__":
    main()
