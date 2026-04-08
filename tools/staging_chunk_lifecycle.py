#!/usr/bin/env python3
"""
从 prefill / decode 的 [STAGING-DBG] 日志整合每个 request(room) 下各 chunk 的生命周期，
并按 TP rank 列出关键时间点与等待分段，标出「最长等待」来自哪一段。

兼容：
  - 新版 sglang：ENQUEUE、PICKUP(queue_wait_s)、READY(waited_s/since_enqueue_s)、
    CHUNK_DONE / LAST_CHUNK_DONE(rdma_s/total_s)
  - 旧版日志：CHUNK_READY_SEND（无 ENQUEUE/PICKUP）、READY 无 since_enqueue_s

用法:
  python staging_chunk_lifecycle.py --log-dir /path/to/logs
  python staging_chunk_lifecycle.py --prefill prefill_w0.out prefill_w1.out --decode decode_w0.out
  python staging_chunk_lifecycle.py --run-dir outputs/1306769-xxx --top-wait 12

decode 行含: CHUNK_READY 首/末、STAGING_RSP SENT、submit_chunk_scatter、alloc_id、
  _free_and_send_watermark；另附 room 级 KVPoll Success / ALL_SUCCESS（非 per-chunk）。

依赖: 仅标准库。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_ts(line: str) -> Optional[float]:
    m = TS_RE.search(line)
    if not m:
        return None
    dt = datetime.strptime(m.group("ts"), "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    return dt.timestamp()


# ---------- Prefill patterns ----------
RE_PREFETCH_PROCEED = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] _prefetch_staging_reqs PROCEED room=(\d+)"
)
RE_STAGING_REQ_SENT = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\] STAGING_REQ sent room=(\d+) chunk=(\d+)"
)
RE_STAGING_RSP_RECV = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] STAGING_RSP RECV room=(\d+) chunk=(\d+) "
    r"offset=(\S+) round=(\S+) end=(\S+) session=(\S+)"
)
RE_ENQUEUE = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] ENQUEUE room=(\d+) chunk_start=(\d+) "
    r"chunk_len=(\d+) is_last=(\S+) shard=(\d+)/(\d+)"
)
RE_PICKUP = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] PICKUP room=(\d+) chunk_start=(\d+) "
    r"chunk_len=(\d+) is_last=(\S+) queue_wait_s=([\d.]+)"
)
# 新版 conn：无 READY 行，PICKUP 直接带 chunk_idx；队列等待用 since_enqueue_s
RE_PICKUP_V2 = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] PICKUP room=(\d+) chunk_start=(\d+) "
    r"chunk_idx=(\d+) c_offset=(\S+) waited_notready_s=([\d.]+) since_enqueue_s=([\d.]+)"
)
RE_READY = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] _do_staging_transfer READY room=(\d+) "
    r"chunk_start=(\d+) chunk_idx=(\d+) c_offset=(\S+) waited_s=([\d.]+)"
)
# 新版可能含 since_enqueue_s=（在 waited_s 之后）
RE_READY_SINCE_ENQ = re.compile(r"since_enqueue_s=([\d.]+)")
RE_NOT_READY = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] _do_staging_transfer NOT_READY room=(\d+) "
    r"chunk_start=(\d+) .*?wait_s=([\d.]+)"
)
RE_CHUNK_DONE = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] CHUNK_DONE room=(\d+) chunk_idx=(\d+) "
    r"rdma_s=([\d.]+) total_s=([\d.]+)"
)
RE_CHUNK_READY_SEND_OLD = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] CHUNK_READY_SEND room=(\d+) chunk_idx=(\d+) "
    r"rdma_s=([\d.]+)"
)
RE_LAST_CHUNK_DONE = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] LAST_CHUNK_DONE room=(\d+) chunk_idx=(\d+) "
    r"rdma_s=([\d.]+) total_s=([\d.]+)"
)
RE_LAST_CHUNK_OLD = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] LAST_CHUNK_RDMA_DONE room=(\d+) "
    r"chunk_idx=(\d+) rdma_s=([\d.]+)"
)
RE_CHUNK_READY_SENT = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] CHUNK_READY sent room=(\d+) chunk=(\d+) "
    r"page_start=(\d+) num_pages=(\d+) prefill_rank=(\d+)"
)
RE_WATERMARK_RECV = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] WATERMARK_RECV session=(\S+) "
    r"prev=\((\S+),(\S+)\) new=\((\S+),(\S+)\)"
)
RE_WATERMARK_BLOCK = re.compile(
    r"\[STAGING-DBG\]\[PREFILL\]\[tp(\d+)\] check_ready WATERMARK_BLOCK "
    r"session=(\S+) chunk_idx=(\d+) .* wait_s=([\d.]+)"
)

# ---------- Decode patterns ----------
RE_DECODE_REG = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] register_decode_req room=(\d+) "
    r"prefill_tp=(\d+)->decode_tp=(\d+)"
)
RE_STAGING_ACCEPT = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] STAGING_REQ ACCEPTED room=(\d+) chunk=(\d+) "
    r"session=(\S+)"
)
RE_STAGING_ALLOC = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] handle_staging_req ALLOC room=(\d+) chunk=(\d+) "
    r"pages=(\d+) session=(\S+)"
)
RE_STAGING_RSP_SENT = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] STAGING_RSP SENT room=(\d+) chunk=(\d+) "
    r"offset=(\S+) round=(\S+) end=(\S+) session=(\S+)"
)
RE_DECODE_CHUNK_READY = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] CHUNK_READY room=(\d+) chunk=(\d+) "
    r"page_start=(\d+) num_pages=(\d+) .* writers=(\d+)/(\d+)"
)
RE_SUBMIT_SCATTER = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] submit_chunk_scatter room=(\d+) chunk=(\d+) "
    r"page_start=(\d+) num_pages=(\d+) alloc_id=(\d+)"
)
RE_WM_FREE = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] _free_and_send_watermark room=(\d+) "
    r"alloc_id=(\d+) post_wm=\((\S+),(\S+)\)"
)
# KVPoll Success：按 room 聚合 prefill 响应，非按 chunk
RE_DECODE_SUCCESS = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] Success room=(\d+) "
    r"from_prefill_rank=(\d+) arrived=(\d+)/(\d+)"
)
RE_DECODE_ALL_SUCCESS = re.compile(
    r"\[STAGING-DBG\]\[DECODE\]\[tp(\d+)\] ALL_SUCCESS room=(\d+) "
    r"is_staging_room=(\S+)"
)


@dataclass
class PrefillRankChunk:
    """单个 (room, chunk_idx) 在某一 prefill TP rank 上的事件。"""

    room: int
    chunk_idx: int
    prefill_tp: int
    session: str = ""

    staging_rsp_recv_ts: Optional[float] = None
    enqueue_ts: Optional[float] = None
    pickup_ts: Optional[float] = None
    queue_wait_s: Optional[float] = None
    ready_ts: Optional[float] = None
    waited_s: Optional[float] = None
    since_enqueue_s: Optional[float] = None
    rdma_s: Optional[float] = None
    total_s: Optional[float] = None
    chunk_ready_sent_ts: Optional[float] = None
    prefill_rank: Optional[int] = None
    chunk_start: Optional[int] = None

    max_not_ready_wait_s: float = 0.0
    max_watermark_block_s: float = 0.0


@dataclass
class DecodeChunk:
    room: int
    chunk_idx: int
    decode_tp: int = 0
    session: str = ""

    first_accept_ts: Optional[float] = None
    first_alloc_ts: Optional[float] = None
    first_rsp_sent_ts: Optional[float] = None
    # 本 chunk 上 decode 收到 CHUNK_READY 的首次/末次（日志即「收到 prefill 侧 chunk ready」）
    first_chunk_ready_ts: Optional[float] = None
    last_chunk_ready_ts: Optional[float] = None
    chunk_ready_ts: List[Tuple[float, int, int]] = field(
        default_factory=list
    )  # (ts, writers_arrived, num_writers)
    writers_full_ts: Optional[float] = None  # writers 收齐（可 scatter）的 decode 时间
    submit_scatter_ts: Optional[float] = None
    alloc_id: Optional[int] = None  # submit_chunk_scatter 打出，用于对齐 watermark
    wm_free_ts: Optional[float] = None  # _free_and_send_watermark（与 alloc_id 匹配）


def discover_logs(log_dir: Path) -> Tuple[List[Path], List[Path]]:
    prefill = sorted(log_dir.glob("*_prefill_w*.out"))
    decode = sorted(log_dir.glob("*_decode_w*.out"))
    return prefill, decode


# 单次扫描 ENQUEUE/PICKUP 后在 parse_prefill_files_v2 末尾按 chunk_start->chunk_idx 合并
_enqueue_events: List[Tuple[float, int, int, int]] = []
_pickup_events: List[Tuple[float, int, int, int, float]] = []
# (ts, room, chunk_idx, tp, since_enqueue_s) — 新版 PICKUP 已含 chunk_idx
_pickup_v2_events: List[Tuple[float, int, int, int, float]] = []
_not_ready_max: Dict[Tuple[int, int, int], float] = {}


def parse_prefill_files_v2(paths: List[Path]) -> Tuple[Dict[Tuple[int, int, int], PrefillRankChunk], Dict[Tuple[int, int], float]]:
    global _enqueue_events, _pickup_events, _pickup_v2_events, _not_ready_max
    _enqueue_events = []
    _pickup_events = []
    _pickup_v2_events = []
    _not_ready_max.clear()
    chunks: Dict[Tuple[int, int, int], PrefillRankChunk] = {}
    staging_req_first: Dict[Tuple[int, int], float] = {}

    def get_pc(room: int, chunk_idx: int, tp: int) -> PrefillRankChunk:
        k = (room, chunk_idx, tp)
        if k not in chunks:
            chunks[k] = PrefillRankChunk(room=room, chunk_idx=chunk_idx, prefill_tp=tp)
        return chunks[k]

    # chunk_start -> chunk_idx 由 READY 行建立
    start_to_idx: Dict[Tuple[int, int], int] = {}

    for path in paths:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if "[STAGING-DBG]" not in line:
                    continue
                line = strip_ansi(line)
                ts = parse_ts(line)
                if ts is None:
                    continue

                m = RE_STAGING_REQ_SENT.search(line)
                if m:
                    room, ch = int(m.group(1)), int(m.group(2))
                    key = (room, ch)
                    staging_req_first[key] = min(staging_req_first.get(key, ts), ts)
                    continue

                m = RE_STAGING_RSP_RECV.search(line)
                if m:
                    tp, room, ch = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    pc = get_pc(room, ch, tp)
                    if not pc.session:
                        pc.session = m.group(7)
                    if pc.staging_rsp_recv_ts is None:
                        pc.staging_rsp_recv_ts = ts
                    continue

                m = RE_ENQUEUE.search(line)
                if m:
                    tp, room, cstart = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    _enqueue_events.append((ts, room, cstart, tp))
                    continue

                m = RE_PICKUP_V2.search(line)
                if m:
                    tp, room, cstart, ch_idx = (
                        int(m.group(1)),
                        int(m.group(2)),
                        int(m.group(3)),
                        int(m.group(4)),
                    )
                    since_enq = float(m.group(7))
                    start_to_idx[(room, cstart)] = ch_idx
                    _pickup_v2_events.append((ts, room, ch_idx, tp, since_enq))
                    continue

                m = RE_PICKUP.search(line)
                if m:
                    tp, room, cstart = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    qw = float(m.group(6))
                    _pickup_events.append((ts, room, cstart, tp, qw))
                    continue

                m = RE_READY.search(line)
                if m:
                    tp, room, cstart, ch_idx = (
                        int(m.group(1)),
                        int(m.group(2)),
                        int(m.group(3)),
                        int(m.group(4)),
                    )
                    start_to_idx[(room, cstart)] = ch_idx
                    waited = float(m.group(6))
                    pc = get_pc(room, ch_idx, tp)
                    pc.chunk_start = cstart
                    pc.ready_ts = ts
                    pc.waited_s = waited
                    m2 = RE_READY_SINCE_ENQ.search(line)
                    if m2:
                        pc.since_enqueue_s = float(m2.group(1))
                    continue

                m = RE_NOT_READY.search(line)
                if m:
                    tp, room, cstart = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    w = float(m.group(4))
                    _not_ready_max[(room, cstart, tp)] = max(
                        _not_ready_max.get((room, cstart, tp), 0.0), w
                    )
                    continue

                m = RE_CHUNK_DONE.search(line)
                if m:
                    tp, room, ch_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    pc = get_pc(room, ch_idx, tp)
                    pc.rdma_s = float(m.group(4))
                    pc.total_s = float(m.group(5))
                    continue

                m = RE_CHUNK_READY_SEND_OLD.search(line)
                if m:
                    tp, room, ch_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    pc = get_pc(room, ch_idx, tp)
                    pc.rdma_s = float(m.group(4))
                    continue

                m = RE_LAST_CHUNK_DONE.search(line)
                if m:
                    tp, room, ch_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    pc = get_pc(room, ch_idx, tp)
                    pc.rdma_s = float(m.group(4))
                    pc.total_s = float(m.group(5))
                    continue

                m = RE_LAST_CHUNK_OLD.search(line)
                if m:
                    tp, room, ch_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    pc = get_pc(room, ch_idx, tp)
                    pc.rdma_s = float(m.group(4))
                    continue

                m = RE_CHUNK_READY_SENT.search(line)
                if m:
                    tp, room, ch_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    prank = int(m.group(6))
                    pc = get_pc(room, ch_idx, tp)
                    pc.chunk_ready_sent_ts = ts
                    pc.prefill_rank = prank
                    continue

    # 合并 ENQUEUE / PICKUP 到 chunk_idx
    for ts, room, cstart, tp in _enqueue_events:
        ch_idx = start_to_idx.get((room, cstart))
        if ch_idx is None:
            continue
        pc = get_pc(room, ch_idx, tp)
        if pc.enqueue_ts is None or ts < pc.enqueue_ts:
            pc.enqueue_ts = ts

    for ts, room, cstart, tp, qw in _pickup_events:
        ch_idx = start_to_idx.get((room, cstart))
        if ch_idx is None:
            continue
        pc = get_pc(room, ch_idx, tp)
        pc.pickup_ts = ts
        pc.queue_wait_s = qw

    for ts, room, ch_idx, tp, since_enq in _pickup_v2_events:
        pc = get_pc(room, ch_idx, tp)
        pc.pickup_ts = ts
        pc.queue_wait_s = since_enq

    for (room, cstart, tp), w in _not_ready_max.items():
        ch_idx = start_to_idx.get((room, cstart))
        if ch_idx is None:
            continue
        pc = get_pc(room, ch_idx, tp)
        pc.max_not_ready_wait_s = max(pc.max_not_ready_wait_s, w)

    return chunks, staging_req_first


def parse_decode_files(
    paths: List[Path],
) -> Tuple[
    Dict[Tuple[int, int], DecodeChunk],
    Dict[int, float],
    Dict[int, float],
]:
    """返回 (decode_chunks, room_first_success_ts, room_all_success_ts)。

    Success / ALL_SUCCESS 为 **room 级**（整请求 KVPoll），与 chunk 无直接一一对应；
    报告中仍按 room 附上，便于与 chunk 时间线对照。
    """
    out: Dict[Tuple[int, int], DecodeChunk] = {}
    room_first_success_ts: Dict[int, float] = {}
    room_all_success_ts: Dict[int, float] = {}
    wm_by_room_alloc: Dict[Tuple[int, int], float] = {}

    def get_dc(room: int, ch: int) -> DecodeChunk:
        k = (room, ch)
        if k not in out:
            out[k] = DecodeChunk(room=room, chunk_idx=ch)
        return out[k]

    for path in paths:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if "[STAGING-DBG]" not in line:
                    continue
                line = strip_ansi(line)
                ts = parse_ts(line)
                if ts is None:
                    continue

                m = RE_STAGING_ACCEPT.search(line)
                if m:
                    dtp, room, ch = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    sess = m.group(4)
                    dc = get_dc(room, ch)
                    dc.decode_tp = dtp
                    dc.session = sess
                    if dc.first_accept_ts is None or ts < dc.first_accept_ts:
                        dc.first_accept_ts = ts
                    continue

                m = RE_STAGING_ALLOC.search(line)
                if m:
                    dtp, room, ch = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    dc = get_dc(room, ch)
                    if dc.first_alloc_ts is None or ts < dc.first_alloc_ts:
                        dc.first_alloc_ts = ts
                    continue

                m = RE_STAGING_RSP_SENT.search(line)
                if m:
                    dtp, room, ch = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    dc = get_dc(room, ch)
                    if dc.first_rsp_sent_ts is None or ts < dc.first_rsp_sent_ts:
                        dc.first_rsp_sent_ts = ts
                    continue

                m = RE_DECODE_CHUNK_READY.search(line)
                if m:
                    dtp, room, ch = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    wa, nw = int(m.group(6)), int(m.group(7))
                    dc = get_dc(room, ch)
                    dc.chunk_ready_ts.append((ts, wa, nw))
                    continue

                m = RE_SUBMIT_SCATTER.search(line)
                if m:
                    dtp, room, ch = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    aid = int(m.group(6))
                    dc = get_dc(room, ch)
                    dc.submit_scatter_ts = ts
                    dc.alloc_id = aid
                    continue

                m = RE_WM_FREE.search(line)
                if m:
                    room = int(m.group(2))
                    aid = int(m.group(3))
                    key = (room, aid)
                    prev = wm_by_room_alloc.get(key)
                    if prev is None or ts < prev:
                        wm_by_room_alloc[key] = ts
                    continue

                m = RE_DECODE_SUCCESS.search(line)
                if m:
                    room = int(m.group(2))
                    prev = room_first_success_ts.get(room)
                    if prev is None or ts < prev:
                        room_first_success_ts[room] = ts
                    continue

                m = RE_DECODE_ALL_SUCCESS.search(line)
                if m:
                    room = int(m.group(2))
                    prev = room_all_success_ts.get(room)
                    if prev is None or ts < prev:
                        room_all_success_ts[room] = ts
                    continue

    # 第一个 writers 收齐 (a>=n) 的时刻；CHUNK_READY 首/末条
    for dc in out.values():
        dc.chunk_ready_ts.sort(key=lambda x: x[0])
        if dc.chunk_ready_ts:
            dc.first_chunk_ready_ts = dc.chunk_ready_ts[0][0]
            dc.last_chunk_ready_ts = dc.chunk_ready_ts[-1][0]
        full_ts: Optional[float] = None
        for t, a, n in dc.chunk_ready_ts:
            if n > 0 and a >= n:
                full_ts = t
                break
        if full_ts is None and dc.chunk_ready_ts:
            max_n = max(x[2] for x in dc.chunk_ready_ts)
            for t, a, n in dc.chunk_ready_ts:
                if n == max_n and a == n:
                    full_ts = t
                    break
        dc.writers_full_ts = full_ts
        if dc.alloc_id is not None:
            wk = (dc.room, dc.alloc_id)
            if wk in wm_by_room_alloc:
                dc.wm_free_ts = wm_by_room_alloc[wk]

    return out, room_first_success_ts, room_all_success_ts


def fmt_ts(t: Optional[float]) -> str:
    if t is None:
        return "-"
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    return dt.strftime("%H:%M:%S") + f".{dt.microsecond // 1000:03d}"


def segment_waits(
    pc: PrefillRankChunk, dc: Optional[DecodeChunk], sr_first: Optional[float]
) -> List[Tuple[str, float]]:
    """返回 (阶段名, 秒) 列表，仅含可算的正值。"""
    segs: List[Tuple[str, float]] = []

    if sr_first is not None and pc.staging_rsp_recv_ts is not None:
        segs.append(("prefetch_STAGING_REQ -> 首条STAGING_RSP_RECV", pc.staging_rsp_recv_ts - sr_first))

    if pc.enqueue_ts is not None and pc.pickup_ts is not None:
        segs.append(("ENQUEUE->PICKUP(队列等待)", pc.pickup_ts - pc.enqueue_ts))

    if pc.queue_wait_s is not None:
        segs.append(("PICKUP 记录的 queue_wait_s 或 since_enqueue_s", pc.queue_wait_s))

    if pc.pickup_ts is not None and pc.ready_ts is not None:
        segs.append(("PICKUP->READY(检查+等水位等)", pc.ready_ts - pc.pickup_ts))

    if pc.waited_s is not None:
        segs.append(("READY 行 waited_s(累计 NOT_READY)", pc.waited_s))

    if pc.since_enqueue_s is not None:
        segs.append(("READY 行 since_enqueue_s", pc.since_enqueue_s))

    if pc.rdma_s is not None:
        segs.append(("staging transfer rdma_s", pc.rdma_s))

    return [(n, v) for n, v in segs if v is not None and v >= 0]


def decode_segment_waits(dc: DecodeChunk) -> List[Tuple[str, float]]:
    """decode 侧分段（与 prefill 行分开统计，避免同一指标重复计入多条 tp）。"""
    segs: List[Tuple[str, float]] = []
    if dc.first_accept_ts is not None and dc.first_chunk_ready_ts is not None:
        segs.append(
            ("decode accept -> 首条 CHUNK_READY(收到 prefill)", dc.first_chunk_ready_ts - dc.first_accept_ts)
        )
    if dc.first_chunk_ready_ts is not None and dc.writers_full_ts is not None:
        segs.append(
            ("decode 首条 CHUNK_READY -> writers 收齐", dc.writers_full_ts - dc.first_chunk_ready_ts)
        )
    if dc.writers_full_ts is not None and dc.submit_scatter_ts is not None:
        segs.append(
            ("decode writers 收齐 -> submit_chunk_scatter", dc.submit_scatter_ts - dc.writers_full_ts)
        )
    if dc.submit_scatter_ts is not None and dc.wm_free_ts is not None:
        segs.append(
            (
                "decode submit_chunk_scatter -> _free_and_send_watermark",
                dc.wm_free_ts - dc.submit_scatter_ts,
            )
        )
    return [(n, v) for n, v in segs if v is not None and v >= 0]


def build_report(
    prefill_chunks: Dict[Tuple[int, int, int], PrefillRankChunk],
    staging_req_first: Dict[Tuple[int, int], float],
    decode_chunks: Dict[Tuple[int, int], DecodeChunk],
    room_first_success_ts: Dict[int, float],
    room_all_success_ts: Dict[int, float],
    top_n: int,
) -> str:
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("Staging chunk 生命周期摘要（[STAGING-DBG]）")
    lines.append("=" * 88)

    # 按 room 分组
    rooms = sorted({k[0] for k in prefill_chunks.keys()} | {k[0] for k in decode_chunks.keys()})

    wait_candidates: List[Tuple[float, str]] = []

    for room in rooms:
        chunk_ids = sorted(
            {k[1] for k in prefill_chunks.keys() if k[0] == room}
            | {k[1] for k in decode_chunks.keys() if k[0] == room}
        )
        for ch in chunk_ids:
            dc = decode_chunks.get((room, ch))
            sr = staging_req_first.get((room, ch))
            tps = sorted({k[2] for k in prefill_chunks.keys() if k[0] == room and k[1] == ch})

            lines.append("")
            lines.append(f"--- room={room} chunk={ch} ---")
            if dc:
                lines.append(
                    f"  [decode tp{dc.decode_tp}] accept={fmt_ts(dc.first_accept_ts)} "
                    f"alloc={fmt_ts(dc.first_alloc_ts)} rsp_sent={fmt_ts(dc.first_rsp_sent_ts)} "
                    f"chunk_ready_first={fmt_ts(dc.first_chunk_ready_ts)} "
                    f"chunk_ready_last={fmt_ts(dc.last_chunk_ready_ts)} "
                    f"writers_full={fmt_ts(dc.writers_full_ts)} "
                    f"scatter={fmt_ts(dc.submit_scatter_ts)} "
                    f"alloc_id={dc.alloc_id} wm_free={fmt_ts(dc.wm_free_ts)}"
                )
                lines.append(
                    f"  [decode room] KVPoll Success(首条)={fmt_ts(room_first_success_ts.get(room))} "
                    f"ALL_SUCCESS={fmt_ts(room_all_success_ts.get(room))} "
                    f"(整请求 room 级，与 chunk 无严格一一对应)"
                )
                dsw = decode_segment_waits(dc)
                for name, sec in dsw:
                    wait_candidates.append(
                        (sec, f"room={room} chunk={ch} decode :: {name}")
                    )
                if dsw:
                    dom = max(dsw, key=lambda x: x[1])
                    lines.append(
                        f"    decode 可算分段中最大: {dom[0]} = {dom[1]:.3f}s"
                    )
            for tp in tps:
                pc = prefill_chunks.get((room, ch, tp))
                if not pc:
                    continue
                lines.append(
                    f"  [prefill tp{tp}] rsp_recv={fmt_ts(pc.staging_rsp_recv_ts)} "
                    f"enqueue={fmt_ts(pc.enqueue_ts)} pickup={fmt_ts(pc.pickup_ts)} "
                    f"ready={fmt_ts(pc.ready_ts)} rdma_s={pc.rdma_s} "
                    f"chunk_ready_sent={fmt_ts(pc.chunk_ready_sent_ts)} "
                    f"rank={pc.prefill_rank}"
                )
                sw = segment_waits(pc, dc, sr)
                for name, sec in sw:
                    wait_candidates.append((sec, f"room={room} chunk={ch} tp={tp} :: {name}"))
                if sw:
                    dom = max(sw, key=lambda x: x[1])
                    lines.append(f"    prefill 可算分段中最大: {dom[0]} = {dom[1]:.3f}s")

    # room 级：KVPoll 首条 Success -> ALL_SUCCESS
    for room in rooms:
        s0 = room_first_success_ts.get(room)
        s1 = room_all_success_ts.get(room)
        if s0 is not None and s1 is not None and s1 >= s0:
            wait_candidates.append(
                (s1 - s0, f"room={room} decode :: KVPoll 首条Success -> ALL_SUCCESS")
            )

    lines.append("")
    lines.append("=" * 88)
    lines.append(f"全局最长等待 Top {top_n}（按分段秒数）")
    lines.append("=" * 88)
    wait_candidates.sort(key=lambda x: -x[0])
    for i, (sec, desc) in enumerate(wait_candidates[:top_n], 1):
        lines.append(f"{i:2d}. {sec:10.3f}s  {desc}")

    if not wait_candidates:
        lines.append("(无足够数据计算分段；请确认日志含 READY/CHUNK_READY_SEND 或 ENQUEUE/PICKUP)")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="整合 STAGING-DBG chunk 生命周期")
    ap.add_argument("--log-dir", type=Path, help="含 *_prefill_w*.out / *_decode_w*.out 的目录")
    ap.add_argument("--prefill", nargs="*", default=[], help="prefill 日志路径")
    ap.add_argument("--decode", nargs="*", default=[], help="decode 日志路径")
    ap.add_argument("--run-dir", type=Path, help="run 目录（使用 run-dir/logs）")
    ap.add_argument("--top-wait", type=int, default=15, help="最长等待 Top N")
    ap.add_argument("--json-out", type=Path, help="导出 JSON")
    ap.add_argument("--text-out", type=Path, help="导出文本报告")
    args = ap.parse_args()

    prefill_paths: List[Path] = [Path(p) for p in args.prefill]
    decode_paths: List[Path] = [Path(p) for p in args.decode]

    if args.run_dir:
        ld = args.run_dir / "logs"
        if ld.is_dir():
            p, d = discover_logs(ld)
            prefill_paths = p
            decode_paths = d
        else:
            prefill_paths = sorted(args.run_dir.glob("*_prefill_w*.out"))
            decode_paths = sorted(args.run_dir.glob("*_decode_w*.out"))

    if args.log_dir:
        p, d = discover_logs(args.log_dir)
        prefill_paths = p
        decode_paths = d

    if not prefill_paths and not decode_paths:
        print("请指定 --log-dir、--run-dir 或 --prefill/--decode", file=sys.stderr)
        return 1

    prefill_chunks, staging_req_first = parse_prefill_files_v2(prefill_paths)
    decode_chunks, room_first_success_ts, room_all_success_ts = parse_decode_files(
        decode_paths
    )

    report = build_report(
        prefill_chunks,
        staging_req_first,
        decode_chunks,
        room_first_success_ts,
        room_all_success_ts,
        args.top_wait,
    )
    print(report)

    if args.text_out:
        args.text_out.write_text(report, encoding="utf-8")

    if args.json_out:
        payload: Dict[str, Any] = {
            "prefill": {
                f"{a}_{b}_{c}": asdict(v)
                for (a, b, c), v in prefill_chunks.items()
            },
            "staging_req_first": {f"{a}_{b}": t for (a, b), t in staging_req_first.items()},
            "decode": {f"{a}_{b}": asdict(v) for (a, b), v in decode_chunks.items()},
            "decode_room_first_kvpoll_success_ts": {
                str(k): v for k, v in room_first_success_ts.items()
            },
            "decode_room_all_success_ts": {
                str(k): v for k, v in room_all_success_ts.items()
            },
        }
        args.json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
