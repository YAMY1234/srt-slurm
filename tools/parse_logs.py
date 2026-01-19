#!/usr/bin/env python3
"""
解析单个 run 的日志，生成 metadata.yaml

用法:
    python tools/parse_logs.py                    # 解析 logs/ 下所有 run
    python tools/parse_logs.py --run-dir logs/2252_1A_20260106_015710-xxx  # 解析单个 run
    python tools/parse_logs.py --force            # 强制重新生成（覆盖已有的 metadata）
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
    从日志内容中解析 server_args=ServerArgs(...) 
    这是唯一的参数来源
    """
    # 匹配 server_args=ServerArgs(...) 格式
    pattern = r'server_args=ServerArgs\((.*?)\)(?:\n|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return None
    
    args_str = match.group(1)
    result = {}
    
    # 解析 key=value 对，处理嵌套括号和引号
    current_key = None
    current_value = []
    depth = 0
    in_string = False
    string_char = None
    
    i = 0
    while i < len(args_str):
        char = args_str[i]
        
        # 处理字符串
        if char in ('"', "'") and (i == 0 or args_str[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
        
        # 处理括号深度
        if not in_string:
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
        
        # 处理参数分隔
        if char == ',' and depth == 0 and not in_string:
            if current_key:
                value_str = ''.join(current_value).strip()
                result[current_key] = _parse_value(value_str)
            current_key = None
            current_value = []
            i += 1
            continue
        
        # 处理 key=value
        if char == '=' and depth == 0 and not in_string and current_key is None:
            current_key = ''.join(current_value).strip()
            current_value = []
            i += 1
            continue
        
        current_value.append(char)
        i += 1
    
    # 处理最后一个参数
    if current_key:
        value_str = ''.join(current_value).strip()
        result[current_key] = _parse_value(value_str)
    
    return result


def _parse_value(value_str: str) -> Any:
    """将字符串值转换为 Python 类型"""
    value_str = value_str.strip()
    
    if value_str == 'None':
        return None
    if value_str == 'True':
        return True
    if value_str == 'False':
        return False
    
    # 尝试解析数字
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass
    
    # 去除引号
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]
    
    # 列表或字典保持原样（作为字符串）
    if value_str.startswith('[') or value_str.startswith('{'):
        return value_str
    
    return value_str


class LogParser:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.config_path = run_dir / "config.yaml"
        self.metadata_path = run_dir / "metadata.yaml"
        
        # 检测日志文件位置：可能在 run_dir 根目录或 run_dir/logs 子目录
        self.logs_subdir = self._detect_logs_location()
        
        # benchmark.out 的位置
        if self.logs_subdir:
            self.benchmark_path = self.logs_subdir / "benchmark.out"
        else:
            self.benchmark_path = run_dir / "benchmark.out"
        
        # 判断是 agg 还是 disagg 模式
        self.mode = self._detect_mode()
    
    def _detect_logs_location(self) -> Optional[Path]:
        """
        检测日志文件的位置
        返回 logs 子目录的路径，如果日志在根目录则返回 None
        """
        logs_subdir = self.run_dir / "logs"
        
        # 检查 logs 子目录是否存在且包含日志文件
        if logs_subdir.exists() and logs_subdir.is_dir():
            # 检查是否有 _w0.out 或 _w1.out 等日志文件
            for f in logs_subdir.iterdir():
                if f.name.endswith('_w0.out') or f.name.endswith('_w1.out'):
                    return logs_subdir
        
        # 否则日志在根目录
        return None
    
    def _detect_mode(self) -> str:
        """检测运行模式：agg 或 disagg"""
        # 从目录名判断：包含 _1A_ 表示 agg，包含 _P_D_ 表示 disagg
        dir_name = self.run_dir.name
        if '_1A_' in dir_name or '_2A_' in dir_name:
            return 'agg'
        elif '_P_' in dir_name and '_D_' in dir_name:
            return 'disagg'
        
        # 从文件存在性判断
        # 确定要扫描的目录（根目录或 logs 子目录）
        scan_dir = self.logs_subdir if self.logs_subdir else self.run_dir
        
        for f in scan_dir.iterdir():
            if f.name.endswith('_agg_w0.out'):
                return 'agg'
            if f.name.endswith('_prefill_w0.out') or f.name.endswith('_decode_w0.out'):
                return 'disagg'
        
        return 'unknown'
    
    def _find_server_logs(self) -> Dict[str, Any]:
        """查找 server 日志文件（支持多个 worker）"""
        logs = {}
        
        # 确定要扫描的目录（根目录或 logs 子目录）
        scan_dir = self.logs_subdir if self.logs_subdir else self.run_dir
        
        prefill_logs = []
        decode_logs = []
        
        for f in sorted(scan_dir.iterdir()):
            if f.is_file():
                # 匹配 _agg_w*.out
                if '_agg_w' in f.name and f.name.endswith('.out'):
                    if 'agg' not in logs:
                        logs['agg'] = []
                    logs['agg'].append(f)
                # 匹配 _prefill_w*.out
                elif '_prefill_w' in f.name and f.name.endswith('.out'):
                    prefill_logs.append(f)
                # 匹配 _decode_w*.out
                elif '_decode_w' in f.name and f.name.endswith('.out'):
                    decode_logs.append(f)
        
        if prefill_logs:
            logs['prefill'] = prefill_logs
        if decode_logs:
            logs['decode'] = decode_logs
        
        return logs
        
    def parse(self) -> Dict[str, Any]:
        """解析运行目录，返回 metadata"""
        metadata = {
            'run_id': self.run_dir.name,
            'run_dir': str(self.run_dir),
            'parsed_at': datetime.now().isoformat(),
            'mode': self.mode,
        }
        
        # 查找 server 日志
        server_logs = self._find_server_logs()
        
        if not server_logs:
            metadata['error'] = 'No server log found'
            return metadata
        
        # 根据模式解析
        if self.mode == 'agg':
            metadata['server_args'] = self._parse_agg_mode(server_logs)
        elif self.mode == 'disagg':
            metadata['prefill_args'] = self._parse_worker_log(server_logs.get('prefill'))
            metadata['decode_args'] = self._parse_worker_log(server_logs.get('decode'))
        else:
            # 尝试解析任何存在的日志
            for log_type, log_path in server_logs.items():
                metadata[f'{log_type}_args'] = self._parse_worker_log(log_path)
        
        # 解析性能统计
        metadata['server_stats'] = self._parse_all_stats(server_logs)
        
        # 解析 benchmark.out 获取 Score
        metadata['benchmark_results'] = self._parse_benchmark()
        
        # 如果 server_args 中没有某些信息，从 config.yaml 补充
        if self.config_path.exists():
            metadata['config_fallback'] = self._parse_config_fallback()
        
        return metadata
    
    def _parse_agg_mode(self, server_logs: Dict[str, Path]) -> Dict[str, Any]:
        """解析 aggregated 模式的日志"""
        if 'agg' not in server_logs:
            return {'error': 'agg log not found'}
        
        return self._parse_worker_log(server_logs['agg'])
    
    def _parse_worker_log(self, log_path: Any) -> Dict[str, Any]:
        """解析 worker 的日志，提取 ServerArgs（支持单个或多个 worker）"""
        # 支持单个 Path 或 Path 列表
        if log_path is None:
            return {'error': 'log file not found'}
        
        # 如果是列表，取第一个（所有 worker 配置应该相同）
        if isinstance(log_path, list):
            if not log_path:
                return {'error': 'log file not found'}
            actual_path = log_path[0]
            worker_count = len(log_path)
        else:
            actual_path = log_path
            worker_count = 1
        
        if not actual_path.exists():
            return {'error': 'log file not found'}
        
        try:
            with open(actual_path, 'r', errors='ignore') as f:
                content = f.read()
            
            server_args = parse_server_args(content)
            if server_args is None:
                return {'error': 'ServerArgs not found in log'}
            
            # 提取关键参数
            result = self._extract_key_params(server_args)
            result['_raw_count'] = len(server_args)
            result['_worker_count'] = worker_count
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_key_params(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """从 ServerArgs 提取关键参数"""
        key_fields = [
            # 模型信息
            'model_path', 'served_model_name', 'context_length',
            # 量化
            'quantization', 'kv_cache_dtype',
            # 并行配置
            'tp_size', 'pp_size', 'dp_size', 'ep_size',
            # Attention
            'attention_backend', 'enable_dp_attention',
            # 内存
            'mem_fraction_static', 'chunked_prefill_size',
            # 推测解码
            'speculative_algorithm', 'speculative_num_steps', 
            'speculative_eagle_topk', 'speculative_num_draft_tokens',
            # MoE
            'moe_runner_backend', 'moe_dense_tp_size', 'moe_a2a_backend',
            # 其他重要配置
            'disable_radix_cache', 'disable_overlap_schedule',
            'enable_symm_mem', 'enable_torch_symm_mem',
            'enable_flashinfer_allreduce_fusion',
            # Disaggregation
            'disaggregation_mode', 'disaggregation_transfer_backend',
            # 调度
            'max_running_requests', 'schedule_conservativeness',
        ]
        
        result = {}
        for field in key_fields:
            if field in args:
                result[field] = args[field]
        
        return result
    
    def _parse_all_stats(self, server_logs: Dict[str, Any]) -> Dict[str, Any]:
        """解析所有日志的性能统计（支持多个 worker）"""
        stats = {}
        
        for log_type, log_path in server_logs.items():
            if log_path:
                # 支持单个日志或日志列表
                if isinstance(log_path, list):
                    # 多个 worker：合并所有统计数据
                    log_stats = self._parse_multiple_log_stats(log_path)
                else:
                    # 单个 worker
                    if log_path.exists():
                        log_stats = self._parse_log_stats(log_path)
                    else:
                        log_stats = {'error': 'log file not found'}
                
                stats[log_type] = log_stats
        
        return stats
    
    def _parse_log_stats(self, log_path: Path) -> Dict[str, Any]:
        """解析单个日志文件的性能统计"""
        try:
            with open(log_path, 'r', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            return {'error': str(e)}
        
        prefill_stats = []
        decode_stats = []
        
        # 正则表达式
        prefill_pattern = re.compile(
            r'Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), '
            r'token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)'
        )
        
        # Decode batch 有两种格式：
        # agg 模式: Decode batch, #running-req: 14, #token: 785280, token usage: 0.27, accept len: 2.47, accept rate: 0.82, cuda graph: True, gen throughput (token/s): 167.86, #queue-req: 0, 
        # disagg 模式: Decode batch, #running-req: 1, #token: 41536, token usage: 0.01, accept len: 2.20, accept rate: 0.73, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 1, #retracted-req: 0, cuda graph: True, gen throughput (token/s): 0.40, #queue-req: 0,
        decode_pattern = re.compile(
            r'Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), '
            r'accept len: ([\d.]+), accept rate: ([\d.]+), '
            r'(?:pre-allocated usage: [\d.]+, #prealloc-req: \d+, #transfer-req: \d+, #retracted-req: \d+, )?'  # optional disagg fields
            r'cuda graph: \w+, '
            r'gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)'
        )
        
        for line in lines:
            # 解析 Prefill
            match = prefill_pattern.search(line)
            if match:
                prefill_stats.append({
                    'new_seq': int(match.group(1)),
                    'new_token': int(match.group(2)),
                    'cached_token': int(match.group(3)),
                    'token_usage': float(match.group(4)),
                    'running_req': int(match.group(5)),
                    'queue_req': int(match.group(6)),
                })
                continue
            
            # 解析 Decode
            match = decode_pattern.search(line)
            if match:
                decode_stats.append({
                    'running_req': int(match.group(1)),
                    'token': int(match.group(2)),
                    'token_usage': float(match.group(3)),
                    'accept_len': float(match.group(4)),
                    'accept_rate': float(match.group(5)),
                    'gen_throughput': float(match.group(6)),
                    'queue_req': int(match.group(7)),
                })
        
        result = {
            'prefill_count': len(prefill_stats),
            'decode_count': len(decode_stats),
        }
        
        # 计算 Prefill 统计
        if prefill_stats:
            result['prefill'] = self._compute_stats(prefill_stats, [
                'new_seq', 'new_token', 'cached_token', 'token_usage', 'running_req', 'queue_req'
            ])
        
        # 计算 Decode 统计
        if decode_stats:
            result['decode'] = self._compute_stats(decode_stats, [
                'running_req', 'token', 'token_usage', 'accept_len', 'accept_rate', 'gen_throughput', 'queue_req'
            ])
        
        return result
    
    def _parse_multiple_log_stats(self, log_paths: List[Path]) -> Dict[str, Any]:
        """解析并合并多个 worker 的日志统计"""
        all_prefill_stats = []
        all_decode_stats = []
        
        for log_path in log_paths:
            if not log_path.exists():
                continue
            
            try:
                with open(log_path, 'r', errors='ignore') as f:
                    lines = f.readlines()
            except Exception:
                continue
            
            # 正则表达式
            prefill_pattern = re.compile(
                r'Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), '
                r'token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)'
            )
            
            decode_pattern = re.compile(
                r'Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), '
                r'accept len: ([\d.]+), accept rate: ([\d.]+), '
                r'(?:pre-allocated usage: [\d.]+, #prealloc-req: \d+, #transfer-req: \d+, #retracted-req: \d+, )?'
                r'cuda graph: \w+, '
                r'gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)'
            )
            
            for line in lines:
                # 解析 Prefill
                match = prefill_pattern.search(line)
                if match:
                    all_prefill_stats.append({
                        'new_seq': int(match.group(1)),
                        'new_token': int(match.group(2)),
                        'cached_token': int(match.group(3)),
                        'token_usage': float(match.group(4)),
                        'running_req': int(match.group(5)),
                        'queue_req': int(match.group(6)),
                    })
                    continue
                
                # 解析 Decode
                match = decode_pattern.search(line)
                if match:
                    all_decode_stats.append({
                        'running_req': int(match.group(1)),
                        'token': int(match.group(2)),
                        'token_usage': float(match.group(3)),
                        'accept_len': float(match.group(4)),
                        'accept_rate': float(match.group(5)),
                        'gen_throughput': float(match.group(6)),
                        'queue_req': int(match.group(7)),
                    })
        
        result = {
            'prefill_count': len(all_prefill_stats),
            'decode_count': len(all_decode_stats),
            'worker_count': len(log_paths),
        }
        
        # 计算合并后的 Prefill 统计
        if all_prefill_stats:
            result['prefill'] = self._compute_stats(all_prefill_stats, [
                'new_seq', 'new_token', 'cached_token', 'token_usage', 'running_req', 'queue_req'
            ])
        
        # 计算合并后的 Decode 统计
        if all_decode_stats:
            result['decode'] = self._compute_stats(all_decode_stats, [
                'running_req', 'token', 'token_usage', 'accept_len', 'accept_rate', 'gen_throughput', 'queue_req'
            ])
        
        return result
    
    def _compute_stats(self, data: List[Dict], fields: List[str]) -> Dict[str, Dict]:
        """计算指定字段的统计值（平均、最大、最小）"""
        stats = {}
        for field in fields:
            values = [d[field] for d in data if field in d]
            if values:
                stats[field] = {
                    'avg': round(sum(values) / len(values), 4),
                    'max': max(values),
                    'min': min(values),
                    'count': len(values),
                }
        return stats
    
    def _parse_benchmark(self) -> Dict[str, Any]:
        """解析 benchmark.out 获取评测结果"""
        result = {}
        
        if not self.benchmark_path.exists():
            result['error'] = 'benchmark.out not found'
            return result
        
        try:
            with open(self.benchmark_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # 匹配 Score: 0.550 (旧格式)
            score_match = re.search(r'^Score:\s*([\d.]+)', content, re.MULTILINE)
            if score_match:
                result['score'] = float(score_match.group(1))
            
            # 匹配 Total latency: 1249.009 s
            latency_match = re.search(r'Total latency:\s*([\d.]+)\s*s', content)
            if latency_match:
                result['total_latency_s'] = float(latency_match.group(1))
            
            # 匹配详细的 score 字典 (旧格式)
            # {'chars': np.float64(5731.87...), ..., 'score': np.float64(0.5498...)}
            score_dict_match = re.search(r"'score':\s*np\.float64\(([\d.]+)\)", content)
            if score_dict_match:
                result['score_detailed'] = float(score_dict_match.group(1))
            
            # difficulty_easy
            easy_match = re.search(r"'difficulty_easy':\s*np\.float64\(([\d.]+)\)", content)
            if easy_match:
                result['difficulty_easy'] = float(easy_match.group(1))
            
            # difficulty_hard
            hard_match = re.search(r"'difficulty_hard':\s*np\.float64\(([\d.]+)\)", content)
            if hard_match:
                result['difficulty_hard'] = float(hard_match.group(1))
            
            # GPQA 格式：[METRIC] gpqa_mean_score=0.7689393939393939
            gpqa_score_match = re.search(r'\[METRIC\]\s+gpqa_mean_score=([\d.]+)', content)
            if gpqa_score_match:
                result['gpqa_mean_score'] = float(gpqa_score_match.group(1))
            
            # GPQA: Repeat: 8, mean: 0.769
            gpqa_mean_match = re.search(r'Repeat:\s*(\d+),\s*mean:\s*([\d.]+)', content)
            if gpqa_mean_match:
                result['repeat'] = int(gpqa_mean_match.group(1))
                result['mean_score'] = float(gpqa_mean_match.group(2))
            
            # GPQA: Scores: ['0.773', '0.758', '0.803', ...]
            gpqa_scores_match = re.search(r"Scores:\s*\[(.*?)\]", content)
            if gpqa_scores_match:
                scores_str = gpqa_scores_match.group(1)
                scores = [float(s.strip().strip("'\"")) for s in scores_str.split(',')]
                result['scores'] = scores
            
            # GPQA: 'mean_score': np.float64(0.7689393939393939)
            gpqa_mean_detailed = re.search(r"'mean_score':\s*np\.float64\(([\d.]+)\)", content)
            if gpqa_mean_detailed:
                result['mean_score_detailed'] = float(gpqa_mean_detailed.group(1))
            
            # GPQA: 'chars': np.float64(23841.065656565657)
            chars_match = re.search(r"'chars':\s*np\.float64\(([\d.]+)\)", content)
            if chars_match:
                result['avg_chars'] = float(chars_match.group(1))
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def _parse_config_fallback(self) -> Dict[str, Any]:
        """从 config.yaml 获取补充信息"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            result = {}
            
            # 基本信息
            result['name'] = config.get('name', '')
            
            # Benchmark 配置
            benchmark_config = config.get('benchmark', {})
            result['benchmark_type'] = benchmark_config.get('type', '')
            
            # 资源配置
            resources = config.get('resources', {})
            result['agg_nodes'] = resources.get('agg_nodes')
            result['agg_workers'] = resources.get('agg_workers')
            result['prefill_nodes'] = resources.get('prefill_nodes')
            result['prefill_workers'] = resources.get('prefill_workers')
            result['decode_nodes'] = resources.get('decode_nodes')
            result['decode_workers'] = resources.get('decode_workers')
            result['gpu_type'] = resources.get('gpu_type')
            result['gpus_per_node'] = resources.get('gpus_per_node')
            
            # 模型配置
            model = config.get('model', {})
            result['model_path_config'] = model.get('path', '')
            result['model_precision'] = model.get('precision', '')
            
            # 从环境变量提取 MTP 版本
            result['mtp_version'] = self._extract_mtp_version(config)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_mtp_version(self, config: Dict[str, Any]) -> str:
        """从 config 环境变量中提取 MTP 版本"""
        backend = config.get('backend', {})
        
        # 检查 decode_environment, prefill_environment, aggregated_environment
        for env_key in ['decode_environment', 'prefill_environment', 'aggregated_environment']:
            env = backend.get(env_key, {})
            if env and isinstance(env, dict):
                # 检查 SGLANG_ENABLE_SPEC_V2
                if env.get('SGLANG_ENABLE_SPEC_V2') == '1':
                    return 'V2'
        
        # 如果有 speculative_algorithm 但没有 SGLANG_ENABLE_SPEC_V2，则是 V1
        # 检查 sglang_config 中是否启用了 speculative
        sglang_config = backend.get('sglang_config', {})
        for worker_key in ['decode', 'prefill', 'aggregated']:
            worker_config = sglang_config.get(worker_key, {})
            if worker_config and isinstance(worker_config, dict):
                if worker_config.get('speculative-algorithm'):
                    return 'V1'
        
        return ''
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """保存 metadata 到文件"""
        with open(self.metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"✓ 已保存: {self.metadata_path}")


def find_run_dirs(logs_dir: Path) -> List[Path]:
    """查找所有 run 目录"""
    run_dirs = []
    for item in logs_dir.iterdir():
        if item.is_dir():
            # 检查是否有 server log 或 config.yaml
            # 日志可能在根目录或 logs/ 子目录中
            has_log_in_root = any(
                f.name.endswith('_w0.out') 
                for f in item.iterdir() 
                if f.is_file()
            )
            
            # 检查 logs 子目录
            logs_subdir = item / "logs"
            has_log_in_subdir = False
            if logs_subdir.exists() and logs_subdir.is_dir():
                has_log_in_subdir = any(
                    f.name.endswith('_w0.out') or f.name.endswith('_w1.out')
                    for f in logs_subdir.iterdir()
                    if f.is_file()
                )
            
            has_config = (item / "config.yaml").exists()
            
            if has_log_in_root or has_log_in_subdir or has_config:
                run_dirs.append(item)
    
    return sorted(run_dirs)


def main():
    parser = argparse.ArgumentParser(description='解析运行日志生成 metadata')
    parser.add_argument('--run-dir', '-r', help='指定单个 run 目录')
    parser.add_argument('--logs-dir', '-d', default='logs', help='logs 目录路径')
    parser.add_argument('--force', '-f', action='store_true', help='强制重新生成（覆盖已有）')
    
    args = parser.parse_args()
    
    # 确定要处理的目录
    project_root = Path(__file__).parent.parent
    
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        # 支持绝对路径和相对路径
        logs_path = Path(args.logs_dir)
        if logs_path.is_absolute():
            logs_dir = logs_path
        else:
            logs_dir = project_root / args.logs_dir
        if not logs_dir.exists():
            print(f"❌ 日志目录不存在: {logs_dir}")
            sys.exit(1)
        run_dirs = find_run_dirs(logs_dir)
    
    if not run_dirs:
        print("没有找到任何 run 目录")
        sys.exit(0)
    
    print(f"找到 {len(run_dirs)} 个 run 目录")
    print("=" * 60)
    
    parsed_count = 0
    skipped_count = 0
    
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.yaml"
        
        if metadata_path.exists() and not args.force:
            print(f"⏭️  跳过（已存在）: {run_dir.name}")
            skipped_count += 1
            continue
        
        print(f"解析: {run_dir.name}")
        try:
            parser_obj = LogParser(run_dir)
            metadata = parser_obj.parse()
            parser_obj.save_metadata(metadata)
            parsed_count += 1
        except Exception as e:
            print(f"❌ 解析失败: {e}")
    
    print("=" * 60)
    print(f"完成: 解析 {parsed_count} 个, 跳过 {skipped_count} 个")


if __name__ == "__main__":
    main()
