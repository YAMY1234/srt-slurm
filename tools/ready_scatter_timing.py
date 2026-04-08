#!/usr/bin/env python3
"""
打印每个 chunk 的 ready -> scatter 时间。

ready: decode 收到 STAGING-RECV type=READY 且 group_len 达到期望值（所有 prefill writer 完成）的时刻
scatter: WM-FREE 时刻（scatter 完成、staging 释放）

用法:
  python ready_scatter_timing.py --log-dir <logs_dir>
  python ready_scatter_timing.py --run-dir <outputs/1157890-xxx>
  python ready_scatter_timing.py --decode <decode.out> [--decode <decode2.out> ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")

STAGING_REQ_RE = re.compile(
    r"\[STAGING-RECV\] decode_tp=(?P<decode_tp>\d+) room=(?P<room>\d+) chunk=(?P<chunk>\d+) "
    r"session=(?P<session>\S+) type=REQ alloc_id=(?P<alloc_id>\d+)"
)
STAGING_READY_RE = re.compile(
    r"\[STAGING-RECV\] decode_tp=(?P<decode_tp>\d+) room=(?P<room>\d+) chunk=(?P<chunk>\d+) "
    r"session=(?P<session>\S+) type=READY prefill_rank=(?P<prefill_rank>\S+) group_len=(?P<group_len>\d+)"
)
WM_FREE_RE = re.compile(
    r"\[WM-FREE\] decode_tp=(?P<decode_tp>\d+) alloc_id=(?P<alloc_id>\d+) room=(?P<room>\d+)"
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_ts(text: str) -> Optional[float]:
    m = TS_RE.search(text)
    if not m:
        return None
    dt = datetime.strptime(m.group("ts"), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def discover_log_dir(run_dir: Path) -> Path:
    logs_dir = run_dir / "logs"
    return logs_dir if logs_dir.exists() else run_dir


def discover_decode_files(log_dir: Path) -> List[Path]:
    return sorted(log_dir.glob("*_decode_w*.out"))


def parse_decode_log(path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    reqs, readys, frees = [], [], []
    with path.open("r", errors="ignore") as f:
        for line_no, raw in enumerate(f, 1):
            line = strip_ansi(raw.rstrip("\n"))
            ts = parse_ts(line)
            m = STAGING_REQ_RE.search(line)
            if m:
                g = m.groupdict()
                reqs.append({
                    "ts": ts,
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
                    "ts": ts,
                    "room": int(g["room"]),
                    "session": g["session"],
                    "chunk": int(g["chunk"]),
                    "group_len": int(g["group_len"]),
                    "decode_tp": int(g["decode_tp"]),
                })
                continue
            m = WM_FREE_RE.search(line)
            if m:
                g = m.groupdict()
                frees.append({
                    "ts": ts,
                    "room": int(g["room"]),
                    "alloc_id": int(g["alloc_id"]),
                    "decode_tp": int(g["decode_tp"]),
                })
    return reqs, readys, frees


def run_timing(
    decode_files: List[Path],
    expected_num_writers: int = 4,
) -> List[dict]:
    """
    收集每个 chunk 的 ready_ts, scatter_ts, ready_to_scatter_ms
    """
    all_reqs, all_readys, all_frees = [], [], []
    for p in decode_files:
        reqs, readys, frees = parse_decode_log(p)
        all_reqs.extend(reqs)
        all_readys.extend(readys)
        all_frees.extend(frees)

    # (decode_tp, room, alloc_id) -> (session, chunk)
    alloc_to_chunk: Dict[Tuple[int, int, int], Tuple[str, int]] = {}
    for r in all_reqs:
        key = (r["decode_tp"], r["room"], r["alloc_id"])
        alloc_to_chunk[key] = (r["session"], r["chunk"])

    # (room, session, chunk) -> expected group_len (从 READY 推断最大 group_len)
    chunk_max_group: Dict[Tuple[int, str, int], int] = {}
    for r in all_readys:
        key = (r["room"], r["session"], r["chunk"])
        chunk_max_group[key] = max(chunk_max_group.get(key, 0), r["group_len"])

    # (room, session, chunk) -> ready_ts (收到 group_len=expected 的时刻)
    ready_ts: Dict[Tuple[int, str, int], float] = {}
    for r in all_readys:
        key = (r["room"], r["session"], r["chunk"])
        expected = chunk_max_group.get(key, expected_num_writers)
        if r["group_len"] >= expected and r["ts"] is not None:
            if key not in ready_ts or r["ts"] < ready_ts[key]:
                ready_ts[key] = r["ts"]

    # (room, alloc_id) -> scatter_ts（WM-FREE 可能由不同 decode_tp 打印，用 room+alloc_id 匹配）
    scatter_ts: Dict[Tuple[int, int], float] = {}
    for f in all_frees:
        key = (f["room"], f["alloc_id"])
        if f["ts"] is not None:
            scatter_ts[key] = f["ts"]

    # 构建输出行
    rows: List[dict] = []
    seen = set()
    for (dtp, room, alloc_id), (session, chunk) in alloc_to_chunk.items():
        key = (room, session, chunk)
        if key in seen:
            continue
        seen.add(key)
        expected = chunk_max_group.get(key, expected_num_writers)
        rt = ready_ts.get(key)
        st = scatter_ts.get((room, alloc_id))
        delta_ms = None
        if rt is not None and st is not None and st >= rt:
            delta_ms = round((st - rt) * 1000, 2)
        rows.append({
            "room": room,
            "session": session,
            "chunk": chunk,
            "alloc_id": alloc_id,
            "decode_tp": dtp,
            "expected_group_len": expected,
            "ready_ts": rt,
            "scatter_ts": st,
            "ready_to_scatter_ms": delta_ms,
        })
    return rows


def format_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%H:%M:%S") + f".{dt.microsecond // 1000:03d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="打印每个 chunk 的 ready -> scatter 时间"
    )
    parser.add_argument("--log-dir", help="logs 目录")
    parser.add_argument("--run-dir", help="运行目录（含 logs/）")
    parser.add_argument(
        "--decode",
        action="append",
        default=[],
        help="Decode 日志文件",
    )
    parser.add_argument(
        "--num-writers",
        type=int,
        default=4,
        help="期望的 writer 数量（用于推断 expected group_len）",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="输出 CSV 格式",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="额外输出统计摘要（min/max/avg ready_to_scatter_ms）",
    )
    args = parser.parse_args()

    decode_files: List[Path] = []
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = Path(__file__).resolve().parent.parent / run_dir
        log_dir = discover_log_dir(run_dir)
        decode_files = discover_decode_files(log_dir)
    elif args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = Path(__file__).resolve().parent.parent / log_dir
        decode_files = discover_decode_files(log_dir)
    elif args.decode:
        decode_files = [Path(p) for p in args.decode]
    else:
        parser.error("需要指定 --log-dir、--run-dir 或 --decode")

    for p in decode_files:
        if not p.exists():
            print(f"错误: 文件不存在 {p}", file=sys.stderr)
            return 1

    rows = run_timing(decode_files, expected_num_writers=args.num_writers)
    rows.sort(key=lambda r: (r["room"], r["chunk"]))

    if not rows:
        print("未解析到任何 chunk", file=sys.stderr)
        return 1

    if args.csv:
        print("room,session,chunk,alloc_id,decode_tp,expected_group_len,ready_ts,scatter_ts,ready_to_scatter_ms")
        for r in rows:
            rt = format_ts(r["ready_ts"]) if r["ready_ts"] else ""
            st = format_ts(r["scatter_ts"]) if r["scatter_ts"] else ""
            delta = r["ready_to_scatter_ms"] if r["ready_to_scatter_ms"] is not None else ""
            print(f"{r['room']},{r['session']},{r['chunk']},{r['alloc_id']},{r['decode_tp']},{r['expected_group_len']},{rt},{st},{delta}")
    else:
        print(f"{'room':<22} {'chunk':>5} {'session':<22} {'ready':<12} {'scatter':<12} {'ready->scatter (ms)':>18}")
        print("-" * 95)
        for r in rows:
            rt = format_ts(r["ready_ts"])
            st = format_ts(r["scatter_ts"])
            delta = str(r["ready_to_scatter_ms"]) if r["ready_to_scatter_ms"] is not None else "-"
            print(f"{r['room']:<22} {r['chunk']:>5} {r['session']:<22} {rt:<12} {st:<12} {delta:>18}")

    if args.summary:
        deltas = [r["ready_to_scatter_ms"] for r in rows if r["ready_to_scatter_ms"] is not None]
        if deltas:
            print()
            print("摘要:")
            print(f"  ready->scatter: min={min(deltas):.2f} ms, max={max(deltas):.2f} ms, avg={sum(deltas)/len(deltas):.2f} ms, n={len(deltas)}")
        else:
            print("\n摘要: 无有效 ready->scatter 数据")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
