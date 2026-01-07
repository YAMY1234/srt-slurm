#!/usr/bin/env python3
"""
SGLang Benchmark Dashboard - 交互式结果查看器

用法:
    pip install streamlit pandas
    streamlit run tools/dashboard.py
    
    # 指定日志目录
    streamlit run tools/dashboard.py -- --logs-dir /path/to/logs
    streamlit run tools/dashboard.py -- -d /path/to/logs
    
    # 或者指定端口
    streamlit run tools/dashboard.py --server.port 8501
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SGLang Benchmark Dashboard')
    parser.add_argument('--logs-dir', '-d', default=None, help='日志目录路径')
    # 忽略 streamlit 的参数
    args, _ = parser.parse_known_args()
    return args


def load_all_metadata(logs_dir: Path) -> list:
    """加载所有 metadata"""
    metadata_list = []
    
    for run_dir in logs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.yaml"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                    metadata_list.append(metadata)
            except Exception as e:
                print(f"Warning: Failed to load {metadata_path}: {e}")
    
    return metadata_list


def get_server_args(metadata: dict) -> dict:
    """获取 server_args，支持 agg 和 disagg 模式"""
    mode = metadata.get('mode', 'unknown')
    
    if mode == 'agg':
        return metadata.get('server_args', {})
    elif mode == 'disagg':
        # 优先返回 decode_args（通常包含更完整的参数）
        decode_args = metadata.get('decode_args', {})
        prefill_args = metadata.get('prefill_args', {})
        # 合并，decode 优先
        merged = {**prefill_args, **decode_args}
        return merged
    else:
        # 尝试获取任何可用的 args
        for key in ['server_args', 'agg_args', 'decode_args', 'prefill_args']:
            if key in metadata and isinstance(metadata[key], dict):
                return metadata[key]
    
    return {}


def get_prefill_args(metadata: dict) -> dict:
    """获取 prefill 的 server_args（仅 disagg 模式）"""
    mode = metadata.get('mode', 'unknown')
    
    if mode == 'disagg':
        return metadata.get('prefill_args', {})
    elif mode == 'agg':
        return metadata.get('server_args', {})
    
    return {}


def get_decode_stats(metadata: dict) -> dict:
    """获取 decode 统计，支持 agg 和 disagg 模式"""
    stats = metadata.get('server_stats', {})
    
    # agg 模式: server_stats.agg.decode
    if 'agg' in stats:
        agg_stats = stats.get('agg', {})
        if isinstance(agg_stats, dict) and 'decode' in agg_stats:
            return agg_stats['decode']
    
    # disagg 模式: server_stats.decode.decode
    if 'decode' in stats:
        decode_stats = stats.get('decode', {})
        if isinstance(decode_stats, dict) and 'decode' in decode_stats:
            return decode_stats['decode']
    
    return {}


def get_prefill_stats(metadata: dict) -> dict:
    """获取 prefill 统计"""
    stats = metadata.get('server_stats', {})
    
    # agg 模式: server_stats.agg.prefill
    if 'agg' in stats:
        agg_stats = stats.get('agg', {})
        if isinstance(agg_stats, dict) and 'prefill' in agg_stats:
            return agg_stats['prefill']
    
    # disagg 模式: server_stats.prefill.prefill
    if 'prefill' in stats:
        prefill_stats = stats.get('prefill', {})
        if isinstance(prefill_stats, dict) and 'prefill' in prefill_stats:
            return prefill_stats['prefill']
    
    return {}


def get_total_counts(metadata: dict) -> tuple:
    """获取总的 prefill/decode count"""
    stats = metadata.get('server_stats', {})
    
    prefill_count = 0
    decode_count = 0
    
    for worker_stats in stats.values():
        if isinstance(worker_stats, dict):
            prefill_count += worker_stats.get('prefill_count', 0)
            decode_count += worker_stats.get('decode_count', 0)
    
    return prefill_count, decode_count


def flatten_metadata(metadata: dict) -> dict:
    """展平 metadata 为单层字典"""
    flat = {}
    
    # 基本信息
    flat['Run ID'] = metadata.get('run_id', '')
    flat['Parsed At'] = metadata.get('parsed_at', '')
    flat['Mode'] = metadata.get('mode', 'unknown')
    
    # 从 config_fallback 获取名称和 benchmark 类型
    config = metadata.get('config_fallback', {})
    flat['Name'] = config.get('name', '')
    flat['Benchmark'] = config.get('benchmark_type', '')
    flat['GPU Type'] = config.get('gpu_type', '')
    flat['MTP Version'] = config.get('mtp_version', '')
    
    # Benchmark 结果
    bench_results = metadata.get('benchmark_results', {})
    flat['Score'] = bench_results.get('score', None)
    flat['Total Latency (s)'] = bench_results.get('total_latency_s', None)
    flat['Difficulty Easy'] = bench_results.get('difficulty_easy', None)
    flat['Difficulty Hard'] = bench_results.get('difficulty_hard', None)
    
    # 从 server_args 获取参数（优先，decode 优先于 prefill）
    args = get_server_args(metadata)
    prefill_args = get_prefill_args(metadata)
    
    # 模型信息
    flat['Model'] = args.get('served_model_name', '') or Path(args.get('model_path', '')).name[:30]
    flat['Context Length'] = args.get('context_length', '')
    
    # 并行配置 - 使用 decode args
    flat['TP'] = args.get('tp_size', 1)
    flat['DP'] = args.get('dp_size', 1)
    flat['EP'] = args.get('ep_size', 1)
    flat['PP'] = args.get('pp_size', 1)
    
    # Prefill 的并行配置 (disagg 模式)
    if metadata.get('mode') == 'disagg':
        flat['PF TP'] = prefill_args.get('tp_size', '')
        flat['PF DP'] = prefill_args.get('dp_size', '')
        flat['PF EP'] = prefill_args.get('ep_size', '')
    
    # 量化
    flat['Quantization'] = args.get('quantization', '')
    flat['KV Cache Dtype'] = args.get('kv_cache_dtype', '')
    
    # Attention
    flat['Attention Backend'] = args.get('attention_backend', '')
    flat['DP Attention'] = args.get('enable_dp_attention', False)
    
    # 内存配置 - 重要！
    flat['Symm Mem'] = args.get('enable_symm_mem', False)
    flat['Torch Symm Mem'] = args.get('enable_torch_symm_mem', False)
    flat['Mem Fraction'] = args.get('mem_fraction_static', '')
    flat['Chunked Prefill'] = args.get('chunked_prefill_size', '')
    
    # 推测解码
    flat['Spec Algo'] = args.get('speculative_algorithm', '') or 'None'
    flat['Spec Steps'] = args.get('speculative_num_steps', 0)
    flat['Spec Draft Tokens'] = args.get('speculative_num_draft_tokens', 0)
    flat['Spec Eagle TopK'] = args.get('speculative_eagle_topk', 0)
    
    # MoE 配置
    flat['MoE Backend'] = args.get('moe_runner_backend', '')
    flat['MoE Dense TP'] = args.get('moe_dense_tp_size', '')
    
    # 其他配置
    flat['Disable Radix'] = args.get('disable_radix_cache', False)
    flat['FlashInfer AR Fusion'] = args.get('enable_flashinfer_allreduce_fusion', False)
    flat['Max Running Req'] = args.get('max_running_requests', '')
    
    # Disaggregation
    flat['Disagg Mode'] = args.get('disaggregation_mode', 'null')
    
    # 性能统计
    prefill_count, decode_count = get_total_counts(metadata)
    flat['Prefill Count'] = prefill_count
    flat['Decode Count'] = decode_count
    
    # Decode 统计
    decode = get_decode_stats(metadata)
    flat['Accept Len (avg)'] = decode.get('accept_len', {}).get('avg', None)
    flat['Accept Len (max)'] = decode.get('accept_len', {}).get('max', None)
    flat['Accept Len (min)'] = decode.get('accept_len', {}).get('min', None)
    flat['Accept Rate (avg)'] = decode.get('accept_rate', {}).get('avg', None)
    flat['Throughput (avg)'] = decode.get('gen_throughput', {}).get('avg', None)
    flat['Throughput (max)'] = decode.get('gen_throughput', {}).get('max', None)
    flat['Throughput (min)'] = decode.get('gen_throughput', {}).get('min', None)
    flat['DC Running Req (avg)'] = decode.get('running_req', {}).get('avg', None)
    flat['DC Token Usage (avg)'] = decode.get('token_usage', {}).get('avg', None)
    
    # Prefill 统计
    prefill = get_prefill_stats(metadata)
    flat['PF New Token (avg)'] = prefill.get('new_token', {}).get('avg', None)
    flat['PF Token Usage (avg)'] = prefill.get('token_usage', {}).get('avg', None)
    flat['PF Running Req (avg)'] = prefill.get('running_req', {}).get('avg', None)
    flat['PF Running Req (max)'] = prefill.get('running_req', {}).get('max', None)
    flat['PF Queue Req (avg)'] = prefill.get('queue_req', {}).get('avg', None)
    flat['PF Queue Req (max)'] = prefill.get('queue_req', {}).get('max', None)
    
    return flat


def create_dataframe(metadata_list: list) -> pd.DataFrame:
    """创建 DataFrame"""
    if not metadata_list:
        return pd.DataFrame()
    
    data = [flatten_metadata(m) for m in metadata_list]
    df = pd.DataFrame(data)
    
    # 按 Run ID 排序（时间戳在前所以自然排序就是时间顺序）
    df = df.sort_values('Run ID', ascending=False)
    
    return df


def main():
    st.set_page_config(
        page_title="SGLang Benchmark Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 SGLang Benchmark Dashboard")
    
    # 解析命令行参数
    cli_args = parse_args()
    
    # 优先级: 命令行参数 > URL参数 > 默认值
    query_params = st.query_params
    if cli_args.logs_dir:
        default_logs_dir = cli_args.logs_dir
    else:
        default_logs_dir = query_params.get("logs_dir", str(PROJECT_ROOT / "logs"))
    
    # 侧边栏输入日志目录
    st.sidebar.header("📁 Logs Directory")
    logs_dir_input = st.sidebar.text_input(
        "日志目录路径",
        value=default_logs_dir,
        help="输入日志目录的绝对路径或相对路径"
    )
    
    # 解析路径
    logs_path = Path(logs_dir_input)
    if logs_path.is_absolute():
        logs_dir = logs_path
    else:
        logs_dir = PROJECT_ROOT / logs_dir_input
    
    if not logs_dir.exists():
        st.error(f"Logs directory not found: {logs_dir}")
        return
    
    st.sidebar.caption(f"当前目录: `{logs_dir}`")
    
    metadata_list = load_all_metadata(logs_dir)
    
    if not metadata_list:
        st.warning(f"No metadata found in {logs_dir}. Please run `python tools/parse_logs.py -d {logs_dir}` first.")
        return
    
    df = create_dataframe(metadata_list)
    
    # 侧边栏筛选器
    st.sidebar.header("🔍 Filters")
    
    # Mode 筛选
    mode_values = sorted(df['Mode'].unique())
    selected_mode = st.sidebar.multiselect("Mode (agg/disagg)", mode_values, default=mode_values)
    df = df[df['Mode'].isin(selected_mode)]
    
    # 只显示有 decode 数据的
    has_decode = st.sidebar.checkbox("Only show runs with decode data", value=False)
    if has_decode:
        df = df[df['Decode Count'] > 0]
    
    # Symm Mem 筛选
    symm_mem_values = sorted(df['Symm Mem'].unique())
    selected_symm_mem = st.sidebar.multiselect("Symm Mem", symm_mem_values, default=symm_mem_values)
    df = df[df['Symm Mem'].isin(selected_symm_mem)]
    
    # TP 筛选
    tp_values = sorted(df['TP'].unique())
    selected_tp = st.sidebar.multiselect("Tensor Parallel (TP)", tp_values, default=tp_values)
    df = df[df['TP'].isin(selected_tp)]
    
    # DP 筛选
    dp_values = sorted(df['DP'].unique())
    selected_dp = st.sidebar.multiselect("Data Parallel (DP)", dp_values, default=dp_values)
    df = df[df['DP'].isin(selected_dp)]
    
    # EP 筛选
    ep_values = sorted(df['EP'].unique())
    selected_ep = st.sidebar.multiselect("Expert Parallel (EP)", ep_values, default=ep_values)
    df = df[df['EP'].isin(selected_ep)]
    
    # Spec Algo 筛选
    spec_values = df['Spec Algo'].unique().tolist()
    selected_spec = st.sidebar.multiselect("Speculative Algorithm", spec_values, default=spec_values)
    df = df[df['Spec Algo'].isin(selected_spec)]
    
    # MTP Version 筛选
    mtp_values = sorted([v for v in df['MTP Version'].unique() if v])
    if mtp_values:
        selected_mtp = st.sidebar.multiselect("MTP Version", mtp_values, default=mtp_values)
        df = df[df['MTP Version'].isin(selected_mtp) | (df['MTP Version'] == '')]
    
    # Chunked Prefill 筛选
    chunked_values = sorted([v for v in df['Chunked Prefill'].unique() if v != ''])
    if chunked_values:
        selected_chunked = st.sidebar.multiselect("Chunked Prefill Size", chunked_values, default=chunked_values)
        df = df[df['Chunked Prefill'].isin(selected_chunked) | (df['Chunked Prefill'] == '')]
    
    # 显示统计
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(df)} / {len(metadata_list)} runs**")
    
    # 主要内容 - 顶部统计
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Runs", len(df))
    with col2:
        avg_score = df['Score'].mean()
        st.metric("Avg Score", f"{avg_score:.3f}" if pd.notna(avg_score) else "N/A")
    with col3:
        avg_throughput = df['Throughput (avg)'].mean()
        st.metric("Avg Throughput", f"{avg_throughput:.1f}" if pd.notna(avg_throughput) else "N/A")
    with col4:
        avg_accept_len = df['Accept Len (avg)'].mean()
        st.metric("Avg Accept Len", f"{avg_accept_len:.2f}" if pd.notna(avg_accept_len) else "N/A")
    with col5:
        total_decodes = df['Decode Count'].sum()
        st.metric("Total Decode Batches", f"{total_decodes:,}")
    with col6:
        agg_count = len(df[df['Mode'] == 'agg'])
        disagg_count = len(df[df['Mode'] == 'disagg'])
        st.metric("Agg / Disagg", f"{agg_count} / {disagg_count}")
    
    st.markdown("---")
    
    # 选择要显示的列
    st.subheader("📋 Results Table")
    
    # 默认显示的列 - 包含 Score 和 Symm Mem
    default_columns = [
        'Run ID', 'Name', 'Mode', 'Score',
        'TP', 'DP', 'EP', 
        'Symm Mem', 'MTP Version', 'Spec Algo', 'Spec Steps', 'Chunked Prefill',
        'Accept Len (avg)', 'Accept Rate (avg)', 'Throughput (avg)',
        'Decode Count', 'Prefill Count'
    ]
    
    available_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display",
        available_columns,
        default=[c for c in default_columns if c in available_columns]
    )
    
    if selected_columns:
        # 显示可筛选、可排序的数据表
        st.dataframe(
            df[selected_columns],
            use_container_width=True,
            height=600
        )
    
    # 下载按钮
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="benchmark_results.csv",
            mime="text/csv"
        )
    
    with col2:
        json_str = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="benchmark_results.json",
            mime="application/json"
        )
    
    # 详细视图
    st.markdown("---")
    st.subheader("🔎 Detailed View")
    
    if len(df) > 0:
        selected_run = st.selectbox("Select a run to view details", df['Run ID'].tolist())
        
        if selected_run:
            run_data = df[df['Run ID'] == selected_run].iloc[0].to_dict()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Configuration**")
                config_keys = ['Name', 'Mode', 'Model', 'Context Length',
                              'TP', 'DP', 'EP', 'PP', 
                              'PF TP', 'PF DP', 'PF EP',
                              'Spec Algo', 'Spec Steps', 'Spec Draft Tokens', 'Spec Eagle TopK',
                              'Chunked Prefill', 'Disable Radix', 
                              'Attention Backend', 'DP Attention',
                              'Quantization', 'KV Cache Dtype', 
                              'MoE Backend', 'MoE Dense TP',
                              'Max Running Req', 'Disagg Mode', 'Benchmark']
                for key in config_keys:
                    if key in run_data and run_data[key] not in [None, '', False]:
                        st.text(f"{key}: {run_data[key]}")
            
            with col2:
                st.markdown("**Memory & Optimization**")
                mem_keys = ['Symm Mem', 'Torch Symm Mem', 'FlashInfer AR Fusion', 'Mem Fraction']
                for key in mem_keys:
                    if key in run_data:
                        val = run_data[key]
                        if isinstance(val, bool):
                            st.text(f"{key}: {'✅' if val else '❌'}")
                        else:
                            st.text(f"{key}: {val}")
                
                st.markdown("**Benchmark Results**")
                bench_keys = ['Score', 'Total Latency (s)', 'Difficulty Easy', 'Difficulty Hard']
                for key in bench_keys:
                    if key in run_data and run_data[key] is not None:
                        val = run_data[key]
                        if isinstance(val, float):
                            st.text(f"{key}: {val:.4f}")
                        else:
                            st.text(f"{key}: {val}")
            
            with col3:
                st.markdown("**Performance Metrics**")
                perf_keys = ['Prefill Count', 'Decode Count',
                            'Accept Len (avg)', 'Accept Len (max)', 'Accept Len (min)',
                            'Accept Rate (avg)', 
                            'Throughput (avg)', 'Throughput (max)', 'Throughput (min)',
                            'PF New Token (avg)', 'PF Token Usage (avg)',
                            'DC Running Req (avg)', 'DC Token Usage (avg)']
                for key in perf_keys:
                    if key in run_data and run_data[key] is not None:
                        val = run_data[key]
                        if isinstance(val, float):
                            st.text(f"{key}: {val:.4f}")
                        else:
                            st.text(f"{key}: {val}")


if __name__ == "__main__":
    main()
