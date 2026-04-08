#!/usr/bin/env python3
"""
验证 prefill 与 decode 日志的匹配性，检查是否存在丢包或未 scatter 的情况。

用法:
  python verify_staging_match.py --log-dir <logs_dir>
  python verify_staging_match.py --run-dir <outputs/1157890-xxx>
  python verify_staging_match.py --prefill <prefill.out> --decode <decode.out> [--decode <decode2.out> ...]
  python verify_staging_match.py --run-dir <outputs/1157890-xxx> --per-rank-timing  # 打印各 chunk 不同 TP 的准备/发送时间及 Δ

验证规则:
  1. 对每个 (room, session, chunk_idx) 非 last chunk: decode 应收到 group_len=num_writers 的 READY
  2. 对每个 chunk: 应有对应的 WM-FREE (alloc_id 来自 STAGING-RECV type=REQ)
  3. prefill STG-RDMA 数量应与 num_writers 一致（非 last 为 4，last 可能为 1~4）
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")

# Prefill: [STG-RDMA] prefill_tp=X room=Y -> decode_tp=Z session=... chunk_idx=N is_last=True/False [rdma_ms=X wait_ms=Y defers=Z enqueue_times=N]
STG_RDMA_RE = re.compile(
    r"\[STG-RDMA\] prefill_tp=(?P<prefill_tp>\d+) room=(?P<room>\d+) -> "
    r"decode_tp=(?P<decode_tp>\d+) session=(?P<session>\S+) .* "
    r"chunk_idx=(?P<chunk_idx>\d+) is_last=(?P<is_last>\w+)"
    r"(?: rdma_ms=(?P<rdma_ms>[\d.]+) wait_ms=(?P<wait_ms>[\d.]+) defers=(?P<defers>\d+)(?: enqueue_times=(?P<enqueue_times>\d+))?)?"
)
# 新格式: [STG-RDMA] tp2w3 room=... -> decode_tp=0 session=... chunk_idx=0 is_last=False rdma_ms=128.6 get_ms=0.2 wait_ms=139.5 defers=0 enqueue_times=4237
STG_RDMA_RE_V2 = re.compile(
    r"\[STG-RDMA\] tp(?P<prefill_tp>\d+)w\d+ room=(?P<room>\d+) -> "
    r"decode_tp=(?P<decode_tp>\d+) session=(?P<session>\S+) .* "
    r"chunk_idx=(?P<chunk_idx>\d+) is_last=(?P<is_last>\w+)"
    r"(?: rdma_ms=(?P<rdma_ms>[\d.]+) .*? wait_ms=(?P<wait_ms>[\d.]+) defers=(?P<defers>\d+)(?: enqueue_times=(?P<enqueue_times>\d+))?)?"
)

# Decode REQ: [STAGING-RECV] decode_tp=X room=Y chunk=Z session=... type=REQ alloc_id=A
STAGING_REQ_RE = re.compile(
    r"\[STAGING-RECV\] decode_tp=(?P<decode_tp>\d+) room=(?P<room>\d+) chunk=(?P<chunk>\d+) "
    r"session=(?P<session>\S+) type=REQ alloc_id=(?P<alloc_id>\d+)"
)

# Decode READY: [STAGING-RECV] ... type=READY prefill_rank=X group_len=Y
STAGING_READY_RE = re.compile(
    r"\[STAGING-RECV\] decode_tp=(?P<decode_tp>\d+) room=(?P<room>\d+) chunk=(?P<chunk>\d+) "
    r"session=(?P<session>\S+) type=READY prefill_rank=(?P<prefill_rank>\S+) group_len=(?P<group_len>\d+)"
)

# Decode WM-FREE: [WM-FREE] decode_tp=X alloc_id=A room=Y
WM_FREE_RE = re.compile(
    r"\[WM-FREE\] decode_tp=(?P<decode_tp>\d+) alloc_id=(?P<alloc_id>\d+) room=(?P<room>\d+)"
)

# Prefill: [STAGING_RSP] room=X chunk=Y offset=... round=... end=... session=Z
# 收到 decode 的 staging 分配，可开始准备 send_kv
STAGING_RSP_RE = re.compile(
    r"\[STAGING_RSP\] room=(?P<room>\d+) chunk=(?P<chunk>\d+) "
    r"offset=(?P<offset>\d+) round=(?P<round>\d+) end=(?P<end>\d+) session=(?P<session>\S+)"
)

# Prefill: [SEND-ENQUEUE] prefill_tp=X room=Y chunk_start=Z pages=N is_last=True/False
# prefill 计算完 chunk 入队时刻
SEND_ENQUEUE_RE = re.compile(
    r"\[SEND-ENQUEUE\] prefill_tp=(?P<prefill_tp>\d+) room=(?P<room>\d+) "
    r"chunk_start=(?P<chunk_start>\d+) pages=(?P<pages>\d+) is_last=(?P<is_last>\w+)"
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_ts(text: str) -> Optional[float]:
    m = TS_RE.search(text)
    if not m:
        return None
    dt = datetime.strptime(m.group("ts"), "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    return dt.timestamp()


def format_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%H:%M:%S") + f".{dt.microsecond // 1000:03d}"


def discover_log_dir(run_dir: Path) -> Path:
    logs_dir = run_dir / "logs"
    return logs_dir if logs_dir.exists() else run_dir


def discover_log_files(log_dir: Path) -> Tuple[List[Path], List[Path]]:
    prefill_files = sorted(log_dir.glob("*_prefill_w*.out"))
    decode_files = sorted(log_dir.glob("*_decode_w*.out"))
    return prefill_files, decode_files


@dataclass
class ChunkExpectation:
    """单个 chunk 的期望与统计"""
    room: int
    session: str
    chunk_idx: int
    is_last: bool
    alloc_id: Optional[int] = None
    prefill_rdma_count: int = 0
    prefill_tps: Set[int] = field(default_factory=set)
    decode_ready_max_group_len: int = 0
    decode_ready_count: int = 0
    wm_free_seen: bool = False


def parse_prefill_log(path: Path) -> List[Dict]:
    events = []
    with path.open("r", errors="ignore") as f:
        for line_no, raw in enumerate(f, 1):
            line = strip_ansi(raw.rstrip("\n"))
            m = STG_RDMA_RE.search(line) or STG_RDMA_RE_V2.search(line)
            if m:
                g = m.groupdict()
                events.append({
                    "file": path.name,
                    "line_no": line_no,
                    "room": int(g["room"]),
                    "session": g["session"],
                    "chunk_idx": int(g["chunk_idx"]),
                    "prefill_tp": int(g["prefill_tp"]),
                    "decode_tp": int(g["decode_tp"]),
                    "is_last": g["is_last"].lower() == "true",
                })
    return events


def parse_prefill_log_with_timing(
    path: Path,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """解析 prefill 日志，返回 (rdma_events, staging_rsp_events, enqueue_events)"""
    rdma_events: List[Dict] = []
    staging_rsp_events: List[Dict] = []
    enqueue_events: List[Dict] = []
    with path.open("r", errors="ignore") as f:
        for line_no, raw in enumerate(f, 1):
            line = strip_ansi(raw.rstrip("\n"))
            ts = parse_ts(line)
            m = STG_RDMA_RE.search(line) or STG_RDMA_RE_V2.search(line)
            if m:
                g = m.groupdict()
                ev = {
                    "room": int(g["room"]),
                    "session": g["session"],
                    "chunk_idx": int(g["chunk_idx"]),
                    "prefill_tp": int(g["prefill_tp"]),
                    "decode_tp": int(g["decode_tp"]),
                    "is_last": g["is_last"].lower() == "true",
                    "ts": ts,
                }
                if g.get("rdma_ms") is not None:
                    ev["rdma_ms"] = float(g["rdma_ms"])
                if g.get("wait_ms") is not None:
                    ev["wait_ms"] = float(g["wait_ms"])
                if g.get("defers") is not None:
                    ev["defers"] = int(g["defers"])
                if g.get("enqueue_times") is not None:
                    ev["enqueue_times"] = int(g["enqueue_times"])
                rdma_events.append(ev)
                continue
            m = STAGING_RSP_RE.search(line)
            if m:
                g = m.groupdict()
                staging_rsp_events.append({
                    "room": int(g["room"]),
                    "session": g["session"],
                    "chunk_idx": int(g["chunk"]),
                    "offset": int(g["offset"]),
                    "round": int(g["round"]),
                    "end": int(g["end"]),
                    "ts": ts,
                })
                continue
            m = SEND_ENQUEUE_RE.search(line)
            if m:
                g = m.groupdict()
                enqueue_events.append({
                    "room": int(g["room"]),
                    "chunk_start": int(g["chunk_start"]),
                    "prefill_tp": int(g["prefill_tp"]),
                    "pages": int(g["pages"]),
                    "is_last": g["is_last"].lower() == "true",
                    "ts": ts,
                })
    return rdma_events, staging_rsp_events, enqueue_events


def parse_decode_log(path: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    reqs, readys, frees = [], [], []
    with path.open("r", errors="ignore") as f:
        for line_no, raw in enumerate(f, 1):
            line = strip_ansi(raw.rstrip("\n"))
            m = STAGING_REQ_RE.search(line)
            if m:
                g = m.groupdict()
                reqs.append({
                    "file": path.name,
                    "line_no": line_no,
                    "room": int(g["room"]),
                    "session": g["session"],
                    "chunk": int(g["chunk"]),
                    "alloc_id": int(g["alloc_id"]),
                    "decode_tp": int(g["decode_tp"]),
                })
                continue
            m = STAGING_READY_RE.search(line)
            if m:
                g = m.groupdict()
                readys.append({
                    "file": path.name,
                    "line_no": line_no,
                    "room": int(g["room"]),
                    "session": g["session"],
                    "chunk": int(g["chunk"]),
                    "prefill_rank": g["prefill_rank"],
                    "group_len": int(g["group_len"]),
                    "decode_tp": int(g["decode_tp"]),
                })
                continue
            m = WM_FREE_RE.search(line)
            if m:
                g = m.groupdict()
                frees.append({
                    "file": path.name,
                    "line_no": line_no,
                    "room": int(g["room"]),
                    "alloc_id": int(g["alloc_id"]),
                    "decode_tp": int(g["decode_tp"]),
                })
    return reqs, readys, frees


def build_chunk_key(room: int, session: str, chunk_idx: int) -> Tuple[int, str, int]:
    return (room, session, chunk_idx)


def run_verification(
    prefill_files: List[Path],
    decode_files: List[Path],
    expected_num_writers: int = 4,
) -> Tuple[Dict[Tuple[int, str, int], ChunkExpectation], List[str]]:
    """
    执行验证，返回 (chunk_expectations, issues)
    """
    # 1. 解析 prefill
    prefill_rdma: List[Dict] = []
    for p in prefill_files:
        prefill_rdma.extend(parse_prefill_log(p))

    # 2. 解析 decode
    all_reqs, all_readys, all_frees = [], [], []
    for p in decode_files:
        reqs, readys, frees = parse_decode_log(p)
        all_reqs.extend(reqs)
        all_readys.extend(readys)
        all_frees.extend(frees)

    # 3. 构建 (room, session, chunk) -> ChunkExpectation
    chunks: Dict[Tuple[int, str, int], ChunkExpectation] = {}

    # 从 REQ 建立 chunk 与 alloc_id 映射
    for r in all_reqs:
        key = build_chunk_key(r["room"], r["session"], r["chunk"])
        if key not in chunks:
            chunks[key] = ChunkExpectation(
                room=r["room"],
                session=r["session"],
                chunk_idx=r["chunk"],
                is_last=False,  # 稍后从 prefill 更新
            )
        chunks[key].alloc_id = r["alloc_id"]

    # 从 prefill STG-RDMA 补充/更新
    for e in prefill_rdma:
        key = build_chunk_key(e["room"], e["session"], e["chunk_idx"])
        if key not in chunks:
            chunks[key] = ChunkExpectation(
                room=e["room"],
                session=e["session"],
                chunk_idx=e["chunk_idx"],
                is_last=e["is_last"],
            )
        else:
            chunks[key].is_last = e["is_last"]
        chunks[key].prefill_rdma_count += 1
        chunks[key].prefill_tps.add(e["prefill_tp"])

    # 从 decode READY 更新
    for r in all_readys:
        key = build_chunk_key(r["room"], r["session"], r["chunk"])
        if key not in chunks:
            chunks[key] = ChunkExpectation(
                room=r["room"],
                session=r["session"],
                chunk_idx=r["chunk"],
                is_last=False,
            )
        chunks[key].decode_ready_count += 1
        chunks[key].decode_ready_max_group_len = max(
            chunks[key].decode_ready_max_group_len,
            r["group_len"],
        )

    # 从 WM-FREE 更新
    alloc_to_chunk: Dict[Tuple[int, int], Tuple[int, str, int]] = {}
    for (room, session, chunk_idx), c in chunks.items():
        if c.alloc_id is not None:
            alloc_to_chunk[(room, c.alloc_id)] = (room, session, chunk_idx)

    for f in all_frees:
        key = alloc_to_chunk.get((f["room"], f["alloc_id"]))
        if key and key in chunks:
            chunks[key].wm_free_seen = True

    # 4. 验证并收集问题
    issues: List[str] = []
    for key in sorted(chunks.keys()):
        c = chunks[key]
        # 非 last chunk 期望 num_writers；last chunk 以实际 prefill 发送数为准
        expected_writers = (
            expected_num_writers
            if not c.is_last
            else (c.prefill_rdma_count if c.prefill_rdma_count > 0 else expected_num_writers)
        )

        # 检查 prefill 发送数量（仅对非 last 做严格校验）
        if not c.is_last and c.prefill_rdma_count > 0 and c.prefill_rdma_count != expected_writers:
            issues.append(
                f"[PREFILL] room={c.room} chunk={c.chunk_idx} session={c.session}: "
                f"STG-RDMA count={c.prefill_rdma_count}, expected={expected_writers} (is_last={c.is_last})"
            )

        # 检查 decode 是否收齐 READY（非 last 需 group_len=num_writers）
        if not c.is_last and c.decode_ready_max_group_len < expected_writers:
            issues.append(
                f"[READY] room={c.room} chunk={c.chunk_idx} session={c.session}: "
                f"max group_len={c.decode_ready_max_group_len}, expected={expected_writers} "
                f"(可能丢包 CHUNK_READY)"
            )

        # 检查 WM-FREE
        if c.alloc_id is not None and not c.wm_free_seen:
            issues.append(
                f"[WM-FREE] room={c.room} chunk={c.chunk_idx} alloc_id={c.alloc_id} session={c.session}: "
                f"未发现 WM-FREE (可能 scatter 未完成或未释放)"
            )

    return chunks, issues


def run_per_rank_timing(
    prefill_files: List[Path],
    decode_files: List[Path],
) -> List[Dict]:
    """
    收集每个 chunk 各 prefill rank 的：
    - 入队时间（SEND-ENQUEUE，prefill 计算完入队）
    - 准备 send_kv 时间（STAGING_RSP 收到分配）
    - 发送时间（STG-RDMA）
    - 相对最早 rank 的时间差
    """
    all_rdma: List[Dict] = []
    all_staging_rsp: List[Dict] = []
    all_enqueue: List[Dict] = []
    for p in prefill_files:
        rdma, rsp, enq = parse_prefill_log_with_timing(p)
        all_rdma.extend(rdma)
        all_staging_rsp.extend(rsp)
        all_enqueue.extend(enq)

    # 从 decode REQ 获取 (room, chunk) -> session 映射（用于关联 prefill）
    all_reqs: List[Dict] = []
    for p in decode_files:
        reqs, _, _ = parse_decode_log(p)
        all_reqs.extend(reqs)

    room_chunk_to_sessions: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    for r in all_reqs:
        room_chunk_to_sessions[(r["room"], r["chunk"])].add(r["session"])
    for r in all_rdma:
        room_chunk_to_sessions[(r["room"], r["chunk_idx"])].add(r["session"])
    for r in all_staging_rsp:
        room_chunk_to_sessions[(r["room"], r["chunk_idx"])].add(r["session"])

    # 按 (room, session, chunk_idx) 聚合
    # rdma: prefill_tp -> ts
    # staging_rsp: 按时间排序的 ts 列表
    # enqueue: prefill_tp -> ts（SEND-ENQUEUE，chunk_start 需映射到 chunk_idx）
    chunk_data: Dict[Tuple[int, str, int], Dict] = defaultdict(
        lambda: {
            "rdma_by_tp": {},
            "staging_rsp_ts": [],
            "staging_alloc": None,
            "enqueue_by_tp": {},  # prefill_tp -> ts
            "is_last": False,
        }
    )
    for e in all_rdma:
        key = (e["room"], e["session"], e["chunk_idx"])
        chunk_data[key]["rdma_by_tp"][e["prefill_tp"]] = e["ts"]
        chunk_data[key]["is_last"] = e["is_last"]
        if "rdma_ms" in e or "wait_ms" in e or "defers" in e or "enqueue_times" in e:
            if "rdma_metrics_by_tp" not in chunk_data[key]:
                chunk_data[key]["rdma_metrics_by_tp"] = {}
            chunk_data[key]["rdma_metrics_by_tp"][e["prefill_tp"]] = {
                "rdma_ms": e.get("rdma_ms"),
                "wait_ms": e.get("wait_ms"),
                "defers": e.get("defers"),
                "enqueue_times": e.get("enqueue_times"),
            }
    for e in all_staging_rsp:
        key = (e["room"], e["session"], e["chunk_idx"])
        if e["ts"] is not None:
            chunk_data[key]["staging_rsp_ts"].append(e["ts"])
        if chunk_data[key]["staging_alloc"] is None and "offset" in e:
            chunk_data[key]["staging_alloc"] = (
                e["offset"],
                e["round"],
                e["end"],
            )
    # room -> sorted chunk_starts，chunk_idx = index in sorted list
    room_chunk_starts: Dict[int, List[int]] = {}
    for e in all_enqueue:
        room = e["room"]
        if room not in room_chunk_starts:
            room_chunk_starts[room] = sorted(
                set(x["chunk_start"] for x in all_enqueue if x["room"] == room)
            )
    # (room, chunk_idx) -> sessions（enqueue 无 session，需按 room+chunk 匹配）
    room_chunk_to_sessions: Dict[Tuple[int, int], set] = defaultdict(set)
    for (room, session, chunk_idx) in chunk_data:
        room_chunk_to_sessions[(room, chunk_idx)].add(session)
    for e in all_enqueue:
        room, chunk_start = e["room"], e["chunk_start"]
        starts = room_chunk_starts.get(room, [])
        try:
            chunk_idx = starts.index(chunk_start)
        except ValueError:
            continue
        for session in room_chunk_to_sessions.get((room, chunk_idx), set()):
            key = (room, session, chunk_idx)
            if key in chunk_data and e["ts"] is not None:
                chunk_data[key]["enqueue_by_tp"][e["prefill_tp"]] = e["ts"]
    for key, data in chunk_data.items():
        data["staging_rsp_ts"].sort()

    rows: List[Dict] = []
    for (room, session, chunk_idx), data in sorted(chunk_data.items()):
        rdma_by_tp = data["rdma_by_tp"]
        staging_rsp_ts = data["staging_rsp_ts"]
        if not rdma_by_tp:
            continue
        min_send_ts = min(t for t in rdma_by_tp.values() if t is not None)
        rows.append({
            "room": room,
            "session": session,
            "chunk_idx": chunk_idx,
            "is_last": data["is_last"],
            "enqueue_by_tp": data.get("enqueue_by_tp", {}),
            "staging_rsp_ts": staging_rsp_ts,
            "staging_alloc": data.get("staging_alloc"),
            "rdma_by_tp": rdma_by_tp,
            "rdma_metrics_by_tp": data.get("rdma_metrics_by_tp", {}),
            "min_send_ts": min_send_ts,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证 prefill 与 decode 日志的匹配性，检查丢包或未 scatter 情况。"
    )
    parser.add_argument(
        "--log-dir",
        help="直接包含 *_prefill_w*.out 和 *_decode_w*.out 的目录",
    )
    parser.add_argument(
        "--run-dir",
        help="运行目录（含 config.yaml 和 logs/ 子目录）",
    )
    parser.add_argument(
        "--prefill",
        action="append",
        default=[],
        help="Prefill 日志文件，可多次指定",
    )
    parser.add_argument(
        "--decode",
        action="append",
        default=[],
        help="Decode 日志文件，可多次指定",
    )
    parser.add_argument(
        "--num-writers",
        type=int,
        default=4,
        help="期望的 prefill writer 数量（默认 4）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="输出每个 chunk 的详细统计",
    )
    parser.add_argument(
        "--per-rank-timing",
        action="store_true",
        help="打印各 chunk 不同 TP 的准备 send_kv 时间、发送时间及相对最早 rank 的时间差",
    )
    parser.add_argument(
        "--max-staging-ranks",
        type=int,
        default=4,
        help="STAGING_RSP 最多显示的 rank 数（TP4 应为 4；若实际 receipt 数>此值会标注异常）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="--per-rank-timing 时输出文件路径；默认输出到 stdout；若指定 --run-dir 则默认为 run-dir/timing.txt",
    )
    args = parser.parse_args()

    prefill_files: List[Path] = []
    decode_files: List[Path] = []
    run_dir: Optional[Path] = None

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = Path(__file__).resolve().parent.parent / run_dir
        log_dir = discover_log_dir(run_dir)
        prefill_files, decode_files = discover_log_files(log_dir)
    elif args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = Path(__file__).resolve().parent.parent / log_dir
        prefill_files, decode_files = discover_log_files(log_dir)
    elif args.prefill or args.decode:
        prefill_files = [Path(p) for p in args.prefill]
        decode_files = [Path(p) for p in args.decode]
    else:
        parser.error("需要指定 --log-dir、--run-dir 或 --prefill/--decode")

    for p in prefill_files + decode_files:
        if not p.exists():
            print(f"错误: 文件不存在 {p}", file=sys.stderr)
            return 1

    if args.per_rank_timing:
        output_path: Optional[Path] = None
        if args.output:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = Path(__file__).resolve().parent.parent / output_path
        elif run_dir is not None:
            output_path = run_dir / "timing.txt"
        out_file = output_path.open("w", encoding="utf-8") if output_path else None
        try:
            def out(s: str = ""):
                (print(s, file=out_file) if out_file else print(s))

            rows = run_per_rank_timing(prefill_files, decode_files)
            max_ranks = getattr(args, "max_staging_ranks", 4)
            out("=" * 100)
            out("各 chunk 不同 TP 的准备 send_kv 时间、发送时间及相对最早 rank 的时间差")
            out("=" * 100)
            out("说明: 入队 = SEND-ENQUEUE（prefill 计算完并入队 transfer queue）")
            out("      准备 send = STAGING_RSP 收到分配时刻（decode 分配 round/offset，各 rank 收到时间）")
            out("      offset/round/end = 同一 chunk 所有 rank 相同（decode 只分配一次）")
            out("      发送 = STG-RDMA 完成时刻; Δ = 相对最早 rank 的 ms")
            out("      rdma_ms/wait_ms/defers/enqueue_times = 新日志格式（RDMA 耗时、队列等待、defer 次数、入队次数）")
            out(f"      TP4 预期 rank 数={max_ranks}，若实际 receipt 数>此值会标注 [异常]\n")
            out()
            for i, row in enumerate(rows):
                room, session, chunk_idx = row["room"], row["session"], row["chunk_idx"]
                enqueue_by_tp = row.get("enqueue_by_tp", {})
                rdma_by_tp = row["rdma_by_tp"]
                staging_rsp_ts = row["staging_rsp_ts"]
                staging_alloc = row.get("staging_alloc")
                min_send_ts = row["min_send_ts"]
                out(f"--- room={room} chunk={chunk_idx} session={session} is_last={row['is_last']} ---")
                if staging_alloc:
                    off, rnd, end = staging_alloc
                    out(f"  分配 (offset={off} round={rnd} end={end})  ← 同一 req 所有 rank 相同")
                # 入队: SEND-ENQUEUE（prefill 计算完并入队）
                if enqueue_by_tp:
                    min_enq = min(t for t in enqueue_by_tp.values() if t is not None)
                    out("  入队 (SEND-ENQUEUE):")
                    for tp in sorted(enqueue_by_tp.keys()):
                        ts = enqueue_by_tp[tp]
                        delta_ms = (
                            round((ts - min_enq) * 1000, 2)
                            if ts is not None and min_enq is not None
                            else None
                        )
                        delta_str = f"  Δ={delta_ms} ms" if delta_ms is not None else ""
                        out(f"    prefill_tp={tp}: {format_ts(ts)}{delta_str}")
                else:
                    out("  入队 (SEND-ENQUEUE): (无)")
                # 准备 send: STAGING_RSP（按时间序，最多显示 max_ranks 个；若超出则标注异常）
                if staging_rsp_ts:
                    n_total = len(staging_rsp_ts)
                    anomaly = f"  [异常: 实际 {n_total} 个 receipt，预期 {max_ranks}]" if n_total > max_ranks else ""
                    out(f"  准备 send (STAGING_RSP 收到时刻):{anomaly}")
                    for idx, ts in enumerate(staging_rsp_ts[:max_ranks]):
                        out(f"    rank{idx}: {format_ts(ts)}")
                    if n_total > max_ranks:
                        out(f"    ... 共 {n_total} 个（仅显示前 {max_ranks}）")
                else:
                    out("  准备 send (STAGING_RSP): (无)")
                # 发送时间及 Δ，以及 rdma_ms/wait_ms/defers（若有）
                rdma_metrics_by_tp = row.get("rdma_metrics_by_tp", {})
                out("  发送 (STG-RDMA):")
                for tp in sorted(rdma_by_tp.keys()):
                    ts = rdma_by_tp[tp]
                    delta_ms = (
                        round((ts - min_send_ts) * 1000, 2)
                        if ts is not None and min_send_ts is not None
                        else None
                    )
                    delta_str = f"  Δ={delta_ms} ms" if delta_ms is not None else ""
                    m = rdma_metrics_by_tp.get(tp, {})
                    metrics_str = ""
                    if m.get("rdma_ms") is not None or m.get("wait_ms") is not None or m.get("defers") is not None or m.get("enqueue_times") is not None:
                        parts = []
                        if m.get("rdma_ms") is not None:
                            parts.append(f"rdma={m['rdma_ms']}ms")
                        if m.get("wait_ms") is not None:
                            parts.append(f"wait={m['wait_ms']}ms")
                        if m.get("defers") is not None:
                            parts.append(f"defers={m['defers']}")
                        if m.get("enqueue_times") is not None:
                            parts.append(f"enqueue_times={m['enqueue_times']}")
                        metrics_str = "  " + " ".join(parts)
                    out(f"    prefill_tp={tp}: {format_ts(ts)}{delta_str}{metrics_str}")
                out()
        finally:
            if out_file:
                out_file.close()
        if output_path:
            print(f"输出已保存到: {output_path}", file=sys.stderr)
        return 0

    chunks, issues = run_verification(
        prefill_files,
        decode_files,
        expected_num_writers=args.num_writers,
    )

    # 输出
    print(f"解析: prefill={len(prefill_files)} 文件, decode={len(decode_files)} 文件")
    print(f"chunk 总数: {len(chunks)}")
    print()

    if issues:
        print("=" * 60)
        print("发现不匹配/疑似丢包:")
        print("=" * 60)
        for i in issues:
            print(f"  {i}")
        print()
        print(f"共 {len(issues)} 个问题")
    else:
        print("验证通过: 未发现丢包或 WM-FREE 缺失")

    if args.verbose and chunks:
        print()
        print("=" * 60)
        print("各 chunk 统计 (room, session, chunk_idx):")
        print("=" * 60)
        for key in sorted(chunks.keys()):
            c = chunks[key]
            has_issue = (
                (not c.is_last and c.decode_ready_max_group_len < args.num_writers)
                or (c.alloc_id is not None and not c.wm_free_seen)
            )
            status = "ISSUE" if has_issue else "OK"
            print(
                f"  room={c.room} chunk={c.chunk_idx} session={c.session} "
                f"is_last={c.is_last} alloc_id={c.alloc_id} "
                f"rdma={c.prefill_rdma_count} ready_max={c.decode_ready_max_group_len} "
                f"wm_free={c.wm_free_seen} [{status}]"
            )

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
