#!/usr/bin/env python3
"""
轻量级 Pareto Dashboard — 只加载 pareto_metadata.json 画图。

用法:
    streamlit run tools/pareto_dashboard.py
    streamlit run tools/pareto_dashboard.py -- -d /path/to/outputs
    streamlit run tools/pareto_dashboard.py -- -d /path/to/outputs --port 8502
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def find_pareto_metadata(outputs_dir: Path) -> list[dict]:
    """扫描 outputs 目录，找到所有 pareto_metadata.json 并加载。
    
    只在固定深度搜索 (outputs/*/logs/sa-bench_*/pareto_metadata.json)，
    避免 rglob 在 lustre 上全目录树遍历太慢。
    """
    results = []
    candidates = []

    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        logs_dir = run_dir / "logs"
        if not logs_dir.is_dir():
            continue
        for bench_dir in logs_dir.iterdir():
            if not bench_dir.is_dir() or not bench_dir.name.startswith("sa-bench"):
                continue
            meta_path = bench_dir / "pareto_metadata.json"
            if meta_path.exists():
                candidates.append((run_dir.name, bench_dir.name, meta_path))

    for run_name, bench_name, meta_path in candidates:
        try:
            with open(meta_path) as f:
                data = json.load(f)
            data["_run_name"] = run_name
            data["_bench_dir"] = bench_name
            data["_meta_path"] = str(meta_path)
            parts = [run_name]
            if data.get("isl") and data.get("osl"):
                parts.append(f"isl{data['isl']}_osl{data['osl']}")
            data["_label"] = " / ".join(parts)
            results.append(data)
        except Exception as e:
            print(f"Warning: skip {meta_path}: {e}", file=sys.stderr)

    return results


def ensure_metadata(outputs_dir: Path) -> None:
    """如果没有 pareto_metadata.json，自动调用 compute 脚本生成。"""
    script = PROJECT_ROOT / "tools" / "compute_pareto_metadata.py"
    if not script.exists():
        st.error(f"compute_pareto_metadata.py not found at {script}")
        return

    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        logs_dir = run_dir / "logs"
        if not logs_dir.is_dir():
            continue
        for bench_dir in logs_dir.iterdir():
            if bench_dir.is_dir() and (bench_dir / "pareto_metadata.json").exists():
                return
    # 没找到任何 metadata

    with st.spinner("未找到 pareto_metadata.json，正在计算..."):
        result = subprocess.run(
            [sys.executable, str(script), str(outputs_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            st.error(f"compute failed:\n{result.stderr}")
        else:
            st.success(result.stdout.strip())


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

PLOT_CONFIGS = {
    "TPS/User vs TPS/GPU": {
        "x": "tps_per_user",
        "y": "tps_per_gpu",
        "x_title": "Output TPS / User",
        "y_title": "Output TPS / GPU",
        "hover": ["concurrency", "mean_ttft_ms", "mean_tpot_ms"],
    },
    "Throughput vs TTFT": {
        "x": "output_throughput",
        "y": "mean_ttft_ms",
        "x_title": "Total Output Throughput (tok/s)",
        "y_title": "Mean TTFT (ms)",
        "hover": ["concurrency", "tps_per_user", "tps_per_gpu"],
    },
    "Throughput vs TPOT": {
        "x": "output_throughput",
        "y": "mean_tpot_ms",
        "x_title": "Total Output Throughput (tok/s)",
        "y_title": "Mean TPOT (ms)",
        "hover": ["concurrency", "tps_per_user", "tps_per_gpu"],
    },
    "Throughput vs E2E Latency": {
        "x": "output_throughput",
        "y": "mean_e2el_ms",
        "x_title": "Total Output Throughput (tok/s)",
        "y_title": "Mean E2E Latency (ms)",
        "hover": ["concurrency", "tps_per_user", "tps_per_gpu"],
    },
    "TPS/GPU vs TPOT": {
        "x": "tps_per_gpu",
        "y": "mean_tpot_ms",
        "x_title": "Output TPS / GPU",
        "y_title": "Mean TPOT (ms)",
        "hover": ["concurrency", "tps_per_user", "output_throughput"],
    },
}


def build_figure(
    selected_runs: list[dict],
    plot_cfg: dict,
) -> go.Figure:
    fig = go.Figure()

    for run in selected_runs:
        points = run.get("points", [])
        if not points:
            continue

        xs = [p[plot_cfg["x"]] for p in points if p.get(plot_cfg["x"]) is not None]
        ys = [p[plot_cfg["y"]] for p in points if p.get(plot_cfg["y"]) is not None]
        concurrencies = [p["concurrency"] for p in points if p.get(plot_cfg["x"]) is not None]

        custom = []
        for p in points:
            if p.get(plot_cfg["x"]) is None:
                continue
            lines = [f"concurrency={p['concurrency']}"]
            for h in plot_cfg["hover"]:
                val = p.get(h)
                if val is not None:
                    if isinstance(val, float):
                        lines.append(f"{h}={val:.2f}")
                    else:
                        lines.append(f"{h}={val}")
            custom.append("<br>".join(lines))

        hover_tpl = (
            "<b>%{text}</b><br>"
            f"{plot_cfg['x_title']}: " + "%{x:.2f}<br>"
            f"{plot_cfg['y_title']}: " + "%{y:.2f}<br>"
            "%{customdata}"
            "<extra>%{fullData.name}</extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers+text",
                name=run["_label"],
                text=[f"c={c}" for c in concurrencies],
                textposition="top center",
                textfont=dict(size=9),
                customdata=custom,
                hovertemplate=hover_tpl,
                marker=dict(size=8),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        xaxis_title=plot_cfg["x_title"],
        yaxis_title=plot_cfg["y_title"],
        hovermode="closest",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ──────────────────────────────────────────────
# Streamlit app
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", "-d", default=None)
    args, _ = parser.parse_known_args()
    return args


def main():
    st.set_page_config(page_title="Pareto Dashboard", page_icon="📈", layout="wide")
    st.title("📈 Pareto Dashboard")

    cli_args = parse_args()
    query_params = st.query_params

    default_dir = (
        cli_args.outputs_dir
        or query_params.get("outputs_dir")
        or str(PROJECT_ROOT / "outputs")
    )

    # ── Sidebar ──
    st.sidebar.header("📁 Outputs Directory")
    outputs_dir_input = st.sidebar.text_input("Path", value=default_dir)
    outputs_dir = Path(outputs_dir_input).resolve()

    if not outputs_dir.exists():
        st.error(f"Directory not found: {outputs_dir}")
        return

    # Auto-compute if needed
    ensure_metadata(outputs_dir)

    all_runs = find_pareto_metadata(outputs_dir)

    if not all_runs:
        st.warning(
            f"未找到 pareto_metadata.json。\n\n"
            f"请先运行:\n```\npython tools/compute_pareto_metadata.py {outputs_dir}\n```"
        )
        return

    st.sidebar.markdown(f"**找到 {len(all_runs)} 组数据**")

    # ── 选择要对比的 runs ──
    st.sidebar.header("🔍 选择 Runs")
    labels = [r["_label"] for r in all_runs]
    selected_labels = st.sidebar.multiselect(
        "选择要对比的 run（可多选）",
        labels,
        default=labels[-3:] if len(labels) >= 3 else labels,
    )
    selected_runs = [r for r in all_runs if r["_label"] in selected_labels]

    if not selected_runs:
        st.info("请在左侧选择至少一个 run。")
        return

    # ── ISL/OSL 筛选 ──
    all_isl_osl = sorted(
        set(
            (r.get("isl"), r.get("osl"))
            for r in selected_runs
            if r.get("isl") is not None
        )
    )
    if len(all_isl_osl) > 1:
        selected_isl_osl = st.sidebar.multiselect(
            "ISL/OSL 筛选",
            [f"{i}/{o}" for i, o in all_isl_osl],
            default=[f"{i}/{o}" for i, o in all_isl_osl],
        )
        filter_set = set(selected_isl_osl)
        selected_runs = [
            r
            for r in selected_runs
            if f"{r.get('isl')}/{r.get('osl')}" in filter_set
        ]

    # ── 刷新按钮 ──
    if st.sidebar.button("🔄 重新计算所有 Metadata"):
        script = PROJECT_ROOT / "tools" / "compute_pareto_metadata.py"
        with st.spinner("重新计算中..."):
            result = subprocess.run(
                [sys.executable, str(script), str(outputs_dir), "--force"],
                capture_output=True,
                text=True,
            )
        if result.returncode == 0:
            st.sidebar.success("完成！请刷新页面。")
        else:
            st.sidebar.error(result.stderr)

    # ── Plot 选择 ──
    st.sidebar.header("📊 图表类型")
    plot_name = st.sidebar.selectbox("选择图表", list(PLOT_CONFIGS.keys()))
    plot_cfg = PLOT_CONFIGS[plot_name]

    # ── 主图 ──
    fig = build_figure(selected_runs, plot_cfg)
    st.plotly_chart(fig, use_container_width=True)

    # ── 数据表 ──
    st.subheader("📋 数据明细")
    for run in selected_runs:
        with st.expander(run["_label"], expanded=len(selected_runs) == 1):
            points = run.get("points", [])
            if not points:
                st.write("No data")
                continue

            header = [
                "Concurrency",
                "TPS/User",
                "TPS/GPU",
                "Total TPS",
                "TTFT(ms)",
                "TPOT(ms)",
                "E2E(ms)",
            ]
            rows = []
            for p in points:
                rows.append([
                    p["concurrency"],
                    f"{p['tps_per_user']:.2f}",
                    f"{p['tps_per_gpu']:.2f}",
                    f"{p['output_throughput']:.1f}",
                    f"{p['mean_ttft_ms']:.1f}" if p.get("mean_ttft_ms") else "-",
                    f"{p['mean_tpot_ms']:.1f}" if p.get("mean_tpot_ms") else "-",
                    f"{p['mean_e2el_ms']:.1f}" if p.get("mean_e2el_ms") else "-",
                ])

            df = pd.DataFrame(rows, columns=header)
            st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
