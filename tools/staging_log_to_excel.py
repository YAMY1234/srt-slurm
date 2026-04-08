#!/usr/bin/env python3
"""
Parse staging-related prefill/decode logs and export an Excel workbook.

The workbook is dependency-free and written with the Python standard library.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")

STG_RDMA_RE = re.compile(
    r"\[STG-RDMA\] prefill_tp=(?P<prefill_tp>\d+) room=(?P<room>\d+) -> "
    r"decode_tp=(?P<decode_tp>\d+) session=(?P<session>\S+) round=(?P<round>\d+) "
    r"offset=(?P<offset>\d+) end=(?P<end>\d+) chunk_idx=(?P<chunk_idx>\d+) "
    r"is_last=(?P<is_last>\w+)"
)
WM_DEFER_RE = re.compile(
    r"\[WM-DEFER\] prefill_tp=(?P<prefill_tp>\d+) room=(?P<room>\d+) "
    r"decode_tp=(?P<decode_tp>\d+) session=(?P<session>\S+) "
    r"need_round=(?P<need_round>\d+) need_end=(?P<need_end>\d+) "
    r"cur_wm=\((?P<cur_wm_round>\d+),(?P<cur_wm_tail>\d+)\) "
    r"(?:(?:cnt=(?P<cnt>\d+))|(?:waiting=(?P<waiting_seconds>[0-9.]+)s))"
)
WM_RECV_RE = re.compile(
    r"\[WM-RECV\] prefill_tp=(?P<prefill_tp>\d+) prefill_dp=(?P<prefill_dp>\d+) "
    r"<- decode_tp=(?P<decode_tp>\S+) session=(?P<session>\S+) "
    r"wm=\((?P<wm_round>\d+),(?P<wm_tail>\d+)\) "
    r"prev=\((?P<prev_round>\d+),(?P<prev_tail>\d+)\)"
)
STG_ALLOC_RE = re.compile(
    r"\[STG-ALLOC\] room=(?P<room>\d+) pages=(?P<pages>\d+) chunks=(?P<chunks>\d+) "
    r"ring_before=\(round=(?P<before_round>\d+),head=(?P<before_head>\d+),allocs=(?P<before_allocs>\d+)\) "
    r"ring_after=\(round=(?P<after_round>\d+),head=(?P<after_head>\d+),allocs=(?P<after_allocs>\d+)\) "
    r"chunk_rounds=(?P<chunk_rounds>\{[^}]*\}) wm=\((?P<wm_round>\d+),(?P<wm_tail>\d+)\)"
)
WM_FREE_RE = re.compile(
    r"\[WM-FREE\] decode_tp=(?P<decode_tp>\d+) alloc_id=(?P<alloc_id>\d+) room=(?P<room>\d+) "
    r"wm_before=\((?P<wm_before_round>\d+),(?P<wm_before_tail>\d+)\) "
    r"wm_after=\((?P<wm_after_round>\d+),(?P<wm_after_tail>\d+)\) "
    r"allocs: (?P<allocs_before>\d+)->(?P<allocs_after>\d+) order_head=(?P<order_head>\S+)"
)
WM_SEND_RE = re.compile(
    r"\[WM-SEND\] decode_tp=(?P<decode_tp>\d+) session=(?P<session>\S+) "
    r"wm=\((?P<wm_round>\d+),(?P<wm_tail>\d+)\) room=(?P<room>\d+)"
)
KV_PAIR_RE = re.compile(r"(\w+)=([^\s]+)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_ts(text: str) -> Tuple[Optional[str], Optional[float]]:
    match = TS_RE.search(text)
    if not match:
        return None, None
    ts_text = match.group("ts")
    dt = datetime.strptime(ts_text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    return ts_text, dt.timestamp()


def format_clock(value: Optional[float]) -> str:
    if value is None:
        return ""
    dt = datetime.fromtimestamp(value, tz=timezone.utc)
    return dt.strftime("%H:%M:%S") + f":{dt.microsecond // 1000:03d}"


def compute_time_origin(events: Sequence[Dict[str, object]]) -> Optional[float]:
    timestamps = [
        event.get("timestamp_epoch")
        for event in events
        if event.get("timestamp_epoch") is not None
    ]
    return min(timestamps) if timestamps else None


def elapsed_seconds(value: Optional[float], origin: Optional[float]) -> Optional[float]:
    if value is None or origin is None:
        return None
    return round(value - origin, 6)


def parse_bool(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    return text.lower() == "true"


def to_int(text: Optional[str]) -> Optional[int]:
    if text is None or text == "":
        return None
    return int(text)


def stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def discover_log_dir(run_dir: Path) -> Path:
    logs_dir = run_dir / "logs"
    return logs_dir if logs_dir.exists() else run_dir


def discover_log_files(log_dir: Path) -> List[Path]:
    files = []
    for pattern in ("*_prefill_w*.out", "*_decode_w*.out"):
        files.extend(sorted(log_dir.glob(pattern)))
    return sorted(files)


def source_from_filename(path: Path) -> Tuple[str, str, Optional[int]]:
    name = path.name
    if "_prefill_w" in name:
        source = "prefill"
        host = name.split("_prefill_w", 1)[0]
        worker_text = name.split("_prefill_w", 1)[1].split(".", 1)[0]
        return source, host, int(worker_text)
    if "_decode_w" in name:
        source = "decode"
        host = name.split("_decode_w", 1)[0]
        worker_text = name.split("_decode_w", 1)[1].split(".", 1)[0]
        return source, host, int(worker_text)
    return "unknown", name, None


def parse_key_values(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in KV_PAIR_RE.findall(text):
        result[key] = value.rstrip(",")
    return result


class LogParser:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.events: List[Dict[str, object]] = []
        self.room_offset_to_chunk: Dict[Tuple[int, int], int] = {}
        self.room_to_rid: Dict[int, str] = {}

    def parse(self) -> List[Dict[str, object]]:
        for log_file in discover_log_files(self.log_dir):
            self._parse_file(log_file)
        self.events.sort(
            key=lambda row: (
                row.get("timestamp_epoch") if row.get("timestamp_epoch") is not None else -1.0,
                stringify(row.get("file")),
                row.get("line_no") if row.get("line_no") is not None else 0,
            )
        )
        return self.events

    def _base_row(
        self,
        path: Path,
        line_no: int,
        timestamp_iso: Optional[str],
        timestamp_epoch: Optional[float],
        clean_line: str,
    ) -> Dict[str, object]:
        source, host, worker_id = source_from_filename(path)
        return {
            "timestamp_iso": timestamp_iso,
            "timestamp_epoch": timestamp_epoch,
            "source": source,
            "host": host,
            "worker_id": worker_id,
            "file": path.name,
            "line_no": line_no,
            "raw_message": clean_line,
            "event_type": "",
            "room": None,
            "rid": None,
            "session": None,
            "prefill_tp": None,
            "prefill_dp": None,
            "decode_tp": None,
            "chunk_idx": None,
            "is_last": None,
            "round": None,
            "offset": None,
            "end": None,
            "need_round": None,
            "need_end": None,
            "waiting_seconds": None,
            "cur_wm_round": None,
            "cur_wm_tail": None,
            "wm_round": None,
            "wm_tail": None,
            "prev_round": None,
            "prev_tail": None,
            "alloc_id": None,
            "allocs_before": None,
            "allocs_after": None,
            "order_head": None,
            "pages": None,
            "chunks": None,
            "before_round": None,
            "before_head": None,
            "before_allocs": None,
            "after_round": None,
            "after_head": None,
            "after_allocs": None,
            "chunk_rounds": None,
            "tokens": None,
            "seq_len": None,
            "stg_offset": None,
            "num_chunks": None,
            "engine_rank": None,
            "attn_tp_rank": None,
            "writers": None,
        }

    def _parse_file(self, path: Path) -> None:
        with path.open("r", errors="ignore") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                clean_line = strip_ansi(raw_line.rstrip("\n"))
                timestamp_iso, timestamp_epoch = parse_ts(clean_line)
                row = self._base_row(
                    path, line_no, timestamp_iso, timestamp_epoch, clean_line
                )

                if self._parse_stg_rdma(clean_line, row):
                    self.events.append(row)
                    continue
                if self._parse_wm_defer(clean_line, row):
                    self.events.append(row)
                    continue
                if self._parse_wm_recv(clean_line, row):
                    self.events.append(row)
                    continue
                if self._parse_stg_alloc(clean_line, row):
                    self.events.append(row)
                    continue
                if self._parse_wm_free(clean_line, row):
                    self.events.append(row)
                    continue
                if self._parse_wm_send(clean_line, row):
                    self.events.append(row)
                    continue
                if self._parse_e2e_ok(clean_line, row):
                    self.events.append(row)
                    continue
                if self._parse_post_scatter_ok(clean_line, row):
                    self.events.append(row)
                    continue

    def _parse_stg_rdma(self, text: str, row: Dict[str, object]) -> bool:
        match = STG_RDMA_RE.search(text)
        if not match:
            return False
        groups = match.groupdict()
        room = int(groups["room"])
        offset = int(groups["offset"])
        chunk_idx = int(groups["chunk_idx"])
        row.update(
            {
                "event_type": "STG-RDMA",
                "prefill_tp": int(groups["prefill_tp"]),
                "room": room,
                "decode_tp": int(groups["decode_tp"]),
                "session": groups["session"],
                "round": int(groups["round"]),
                "offset": offset,
                "end": int(groups["end"]),
                "chunk_idx": chunk_idx,
                "is_last": parse_bool(groups["is_last"]),
            }
        )
        self.room_offset_to_chunk[(room, offset)] = chunk_idx
        return True

    def _parse_wm_defer(self, text: str, row: Dict[str, object]) -> bool:
        match = WM_DEFER_RE.search(text)
        if not match:
            return False
        groups = match.groupdict()
        row.update(
            {
                "event_type": "WM-DEFER",
                "prefill_tp": int(groups["prefill_tp"]),
                "room": int(groups["room"]),
                "decode_tp": int(groups["decode_tp"]),
                "session": groups["session"],
                "need_round": int(groups["need_round"]),
                "need_end": int(groups["need_end"]),
                "waiting_seconds": (
                    float(groups["waiting_seconds"])
                    if groups.get("waiting_seconds") is not None
                    else None
                ),
                "cur_wm_round": int(groups["cur_wm_round"]),
                "cur_wm_tail": int(groups["cur_wm_tail"]),
                "order_head": int(groups["cnt"]) if groups.get("cnt") else None,
            }
        )
        return True

    def _parse_wm_recv(self, text: str, row: Dict[str, object]) -> bool:
        match = WM_RECV_RE.search(text)
        if not match:
            return False
        groups = match.groupdict()
        decode_tp_text = groups["decode_tp"]
        row.update(
            {
                "event_type": "WM-RECV",
                "prefill_tp": int(groups["prefill_tp"]),
                "prefill_dp": int(groups["prefill_dp"]),
                "decode_tp": int(decode_tp_text) if decode_tp_text.isdigit() else None,
                "session": groups["session"],
                "wm_round": int(groups["wm_round"]),
                "wm_tail": int(groups["wm_tail"]),
                "prev_round": int(groups["prev_round"]),
                "prev_tail": int(groups["prev_tail"]),
            }
        )
        return True

    def _parse_stg_alloc(self, text: str, row: Dict[str, object]) -> bool:
        match = STG_ALLOC_RE.search(text)
        if not match:
            return False
        groups = match.groupdict()
        row.update(
            {
                "event_type": "STG-ALLOC",
                "room": int(groups["room"]),
                "pages": int(groups["pages"]),
                "chunks": int(groups["chunks"]),
                "before_round": int(groups["before_round"]),
                "before_head": int(groups["before_head"]),
                "before_allocs": int(groups["before_allocs"]),
                "after_round": int(groups["after_round"]),
                "after_head": int(groups["after_head"]),
                "after_allocs": int(groups["after_allocs"]),
                "chunk_rounds": groups["chunk_rounds"],
                "wm_round": int(groups["wm_round"]),
                "wm_tail": int(groups["wm_tail"]),
            }
        )
        return True

    def _parse_wm_free(self, text: str, row: Dict[str, object]) -> bool:
        match = WM_FREE_RE.search(text)
        if not match:
            return False
        groups = match.groupdict()
        row.update(
            {
                "event_type": "WM-FREE",
                "decode_tp": int(groups["decode_tp"]),
                "alloc_id": int(groups["alloc_id"]),
                "room": int(groups["room"]),
                "before_round": int(groups["wm_before_round"]),
                "before_head": int(groups["wm_before_tail"]),
                "after_round": int(groups["wm_after_round"]),
                "after_head": int(groups["wm_after_tail"]),
                "allocs_before": int(groups["allocs_before"]),
                "allocs_after": int(groups["allocs_after"]),
                "order_head": groups["order_head"],
            }
        )
        return True

    def _parse_wm_send(self, text: str, row: Dict[str, object]) -> bool:
        match = WM_SEND_RE.search(text)
        if not match:
            return False
        groups = match.groupdict()
        row.update(
            {
                "event_type": "WM-SEND",
                "decode_tp": int(groups["decode_tp"]),
                "session": groups["session"],
                "wm_round": int(groups["wm_round"]),
                "wm_tail": int(groups["wm_tail"]),
                "room": int(groups["room"]),
            }
        )
        return True

    def _parse_e2e_ok(self, text: str, row: Dict[str, object]) -> bool:
        if "[E2E OK]" not in text:
            return False
        kv = parse_key_values(text.split("[E2E OK]", 1)[1])
        room = to_int(kv.get("room"))
        stg_offset = to_int(kv.get("stg_offset"))
        row.update(
            {
                "event_type": "E2E-OK",
                "rid": kv.get("rid"),
                "room": room,
                "engine_rank": to_int(kv.get("engine_rank")),
                "attn_tp_rank": to_int(kv.get("attn_tp_rank")),
                "prefill_tp": to_int(kv.get("prefill_tp")),
                "decode_tp": to_int(kv.get("decode_tp")),
                "tokens": to_int(kv.get("tokens")),
                "pages": to_int(kv.get("pages")),
                "seq_len": to_int(kv.get("seq_len")),
                "writers": to_int(kv.get("writers")),
                "stg_offset": stg_offset,
                "num_chunks": to_int(kv.get("num_chunks")),
                "chunk_idx": self.room_offset_to_chunk.get((room, stg_offset))
                if room is not None and stg_offset is not None
                else None,
            }
        )
        if room is not None and kv.get("rid"):
            self.room_to_rid[room] = kv["rid"]
        return True

    def _parse_post_scatter_ok(self, text: str, row: Dict[str, object]) -> bool:
        if "[POST-SCATTER OK]" not in text:
            return False
        kv = parse_key_values(text.split("[POST-SCATTER OK]", 1)[1])
        room = to_int(kv.get("room"))
        row.update(
            {
                "event_type": "POST-SCATTER-OK",
                "rid": kv.get("rid"),
                "room": room,
                "prefill_tp": to_int(kv.get("prefill_tp")),
                "decode_tp": to_int(kv.get("decode_tp")),
                "tokens": to_int(kv.get("tokens")),
                "pages": to_int(kv.get("pages")),
            }
        )
        if room is not None and kv.get("rid"):
            self.room_to_rid[room] = kv["rid"]
        return True


def iso_or_empty(value: Optional[float]) -> str:
    if value is None:
        return ""
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def public_event_rows(
    events: Sequence[Dict[str, object]], origin: Optional[float]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ordered_keys = [
        "timestamp_iso",
        "time",
        "elapsed_s",
        "source",
        "host",
        "worker_id",
        "event_type",
        "room",
        "rid",
        "chunk_idx",
        "session",
        "prefill_tp",
        "prefill_dp",
        "decode_tp",
        "round",
        "offset",
        "end",
        "is_last",
        "need_round",
        "need_end",
        "waiting_seconds",
        "cur_wm_round",
        "cur_wm_tail",
        "wm_round",
        "wm_tail",
        "prev_round",
        "prev_tail",
        "alloc_id",
        "allocs_before",
        "allocs_after",
        "order_head",
        "pages",
        "chunks",
        "before_round",
        "before_head",
        "before_allocs",
        "after_round",
        "after_head",
        "after_allocs",
        "chunk_rounds",
        "tokens",
        "seq_len",
        "stg_offset",
        "num_chunks",
        "engine_rank",
        "attn_tp_rank",
        "writers",
        "file",
        "line_no",
        "raw_message",
    ]
    for event in events:
        row = {
            "timestamp_iso": event.get("timestamp_iso"),
            "time": format_clock(event.get("timestamp_epoch")),
            "elapsed_s": elapsed_seconds(event.get("timestamp_epoch"), origin),
        }
        for key in ordered_keys[3:]:
            row[key] = event.get(key)
        rows.append(row)
    return rows


def collapse_sorted(values: Iterable[object]) -> str:
    cleaned = sorted(
        {stringify(value) for value in values if value is not None and stringify(value) != ""}
    )
    return ",".join(cleaned)


def aggregate_chunk_rank(
    events: Sequence[Dict[str, object]], origin: Optional[float]
) -> List[Dict[str, object]]:
    rdma_lookup = build_rdma_chunk_lookup(events)
    groups: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    for event in events:
        if event["event_type"] not in {"STG-RDMA", "WM-DEFER"}:
            continue
        effective_chunk_idx = resolve_chunk_idx(event, rdma_lookup)
        key = (
            stringify(event.get("room")),
            stringify(effective_chunk_idx),
            stringify(event.get("prefill_tp")),
            stringify(event.get("decode_tp")),
            stringify(event.get("session")),
        )
        item = groups.setdefault(
            key,
            {
                "room": event.get("room"),
                "chunk_idx": effective_chunk_idx,
                "prefill_tp": event.get("prefill_tp"),
                "decode_tp": event.get("decode_tp"),
                "session": event.get("session"),
                "stg_round": None,
                "stg_offset": None,
                "stg_end": None,
                "is_last": None,
                "first_wait_ts": None,
                "last_wait_ts": None,
                "wait_event_count": 0,
                "max_need_round": None,
                "max_need_end": None,
                "max_waiting_seconds": None,
                "last_cur_wm_round": None,
                "last_cur_wm_tail": None,
                "rdma_ts": None,
                "rdma_file": None,
                "rdma_line_no": None,
            },
        )
        ts = event.get("timestamp_epoch")
        if event["event_type"] == "WM-DEFER":
            item["first_wait_ts"] = (
                ts if item["first_wait_ts"] is None else min(item["first_wait_ts"], ts)
            )
            item["last_wait_ts"] = (
                ts if item["last_wait_ts"] is None else max(item["last_wait_ts"], ts)
            )
            item["wait_event_count"] += 1
            need_round = event.get("need_round")
            need_end = event.get("need_end")
            if need_round is not None:
                item["max_need_round"] = (
                    need_round
                    if item["max_need_round"] is None
                    else max(item["max_need_round"], need_round)
                )
            if need_end is not None:
                item["max_need_end"] = (
                    need_end
                    if item["max_need_end"] is None
                    else max(item["max_need_end"], need_end)
                )
            waiting_seconds = event.get("waiting_seconds")
            if waiting_seconds is not None:
                item["max_waiting_seconds"] = (
                    waiting_seconds
                    if item["max_waiting_seconds"] is None
                    else max(item["max_waiting_seconds"], waiting_seconds)
                )
            item["last_cur_wm_round"] = event.get("cur_wm_round")
            item["last_cur_wm_tail"] = event.get("cur_wm_tail")
        else:
            item["rdma_ts"] = event.get("timestamp_epoch")
            item["rdma_file"] = event.get("file")
            item["rdma_line_no"] = event.get("line_no")
            item["stg_round"] = event.get("round")
            item["stg_offset"] = event.get("offset")
            item["stg_end"] = event.get("end")
            item["is_last"] = event.get("is_last")

    rows: List[Dict[str, object]] = []
    for item in groups.values():
        first_wait = item["first_wait_ts"]
        rdma_ts = item["rdma_ts"]
        rows.append(
            {
                "room": item["room"],
                "chunk_idx": item["chunk_idx"],
                "prefill_tp": item["prefill_tp"],
                "decode_tp": item["decode_tp"],
                "session": item["session"],
                "wait_start": format_clock(first_wait),
                "wait_end": format_clock(item["last_wait_ts"]),
                "send_time": format_clock(rdma_ts),
                "wait_start_elapsed_s": elapsed_seconds(first_wait, origin),
                "wait_end_elapsed_s": elapsed_seconds(item["last_wait_ts"], origin),
                "send_elapsed_s": elapsed_seconds(rdma_ts, origin),
                "wait_seconds_before_send": (
            round(rdma_ts - first_wait, 6)
            if first_wait is not None and rdma_ts is not None
            else None
                ),
                "wait_event_count": item["wait_event_count"],
                "max_need_round": item["max_need_round"],
                "max_need_end": item["max_need_end"],
                "max_waiting_seconds": item["max_waiting_seconds"],
                "last_cur_wm_round": item["last_cur_wm_round"],
                "last_cur_wm_tail": item["last_cur_wm_tail"],
                "stg_round": item["stg_round"],
                "stg_offset": item["stg_offset"],
                "stg_end": item["stg_end"],
                "is_last": item["is_last"],
                "rdma_file": item["rdma_file"],
                "rdma_line_no": item["rdma_line_no"],
            }
        )
    rows.sort(
        key=lambda row: (
            row.get("room") if row.get("room") is not None else -1,
            row.get("chunk_idx") if row.get("chunk_idx") is not None else -1,
            row.get("prefill_tp") if row.get("prefill_tp") is not None else -1,
        )
    )
    return rows


def aggregate_room_chunk(
    events: Sequence[Dict[str, object]],
    room_to_rid: Dict[int, str],
    origin: Optional[float],
) -> List[Dict[str, object]]:
    rdma_lookup = build_rdma_chunk_lookup(events)
    groups: Dict[Tuple[str, str], Dict[str, object]] = {}
    for event in events:
        room = event.get("room")
        if room is None:
            continue
        chunk_idx = resolve_chunk_idx(event, rdma_lookup)
        if chunk_idx is None and event["event_type"] not in {"STG-ALLOC", "WM-FREE", "WM-SEND"}:
            continue
        key = (stringify(room), stringify(chunk_idx))
        item = groups.setdefault(
            key,
            {
                "room": room,
                "rid": room_to_rid.get(room),
                "chunk_idx": chunk_idx,
                "prefill_tps": set(),
                "decode_tps": set(),
                "sessions": set(),
                "rdma_count": 0,
                "first_wait_ts": None,
                "last_wait_ts": None,
                "wait_event_count": 0,
                "first_rdma_ts": None,
                "last_rdma_ts": None,
                "alloc_pages": None,
                "alloc_chunks": None,
                "max_waiting_seconds": None,
                "first_decode_verify_ts": None,
                "last_decode_verify_ts": None,
            },
        )
        if event.get("prefill_tp") is not None:
            item["prefill_tps"].add(event["prefill_tp"])
        if event.get("decode_tp") is not None:
            item["decode_tps"].add(event["decode_tp"])
        if event.get("session"):
            item["sessions"].add(event["session"])
        ts = event.get("timestamp_epoch")
        event_type = event["event_type"]
        if event_type == "WM-DEFER":
            item["wait_event_count"] += 1
            item["first_wait_ts"] = (
                ts if item["first_wait_ts"] is None else min(item["first_wait_ts"], ts)
            )
            item["last_wait_ts"] = (
                ts if item["last_wait_ts"] is None else max(item["last_wait_ts"], ts)
            )
            waiting_seconds = event.get("waiting_seconds")
            if waiting_seconds is not None:
                item["max_waiting_seconds"] = (
                    waiting_seconds
                    if item["max_waiting_seconds"] is None
                    else max(item["max_waiting_seconds"], waiting_seconds)
                )
        elif event_type == "STG-RDMA":
            item["rdma_count"] += 1
            item["first_rdma_ts"] = (
                ts if item["first_rdma_ts"] is None else min(item["first_rdma_ts"], ts)
            )
            item["last_rdma_ts"] = (
                ts if item["last_rdma_ts"] is None else max(item["last_rdma_ts"], ts)
            )
        elif event_type == "STG-ALLOC" and chunk_idx in (None, ""):
            item["alloc_pages"] = event.get("pages")
            item["alloc_chunks"] = event.get("chunks")
        elif event_type in {"E2E-OK", "POST-SCATTER-OK"}:
            item["first_decode_verify_ts"] = (
                ts
                if item["first_decode_verify_ts"] is None
                else min(item["first_decode_verify_ts"], ts)
            )
            item["last_decode_verify_ts"] = (
                ts
                if item["last_decode_verify_ts"] is None
                else max(item["last_decode_verify_ts"], ts)
            )

    rows: List[Dict[str, object]] = []
    for item in groups.values():
        rows.append(
            {
                "room": item["room"],
                "rid": item["rid"],
                "chunk_idx": item["chunk_idx"],
                "prefill_tps": collapse_sorted(item["prefill_tps"]),
                "decode_tps": collapse_sorted(item["decode_tps"]),
                "sessions": collapse_sorted(item["sessions"]),
                "wait_start": format_clock(item["first_wait_ts"]),
                "wait_end": format_clock(item["last_wait_ts"]),
                "rdma_start": format_clock(item["first_rdma_ts"]),
                "rdma_end": format_clock(item["last_rdma_ts"]),
                "verify_start": format_clock(item["first_decode_verify_ts"]),
                "verify_end": format_clock(item["last_decode_verify_ts"]),
                "wait_start_elapsed_s": elapsed_seconds(item["first_wait_ts"], origin),
                "wait_end_elapsed_s": elapsed_seconds(item["last_wait_ts"], origin),
                "rdma_start_elapsed_s": elapsed_seconds(item["first_rdma_ts"], origin),
                "rdma_end_elapsed_s": elapsed_seconds(item["last_rdma_ts"], origin),
                "verify_start_elapsed_s": elapsed_seconds(item["first_decode_verify_ts"], origin),
                "verify_end_elapsed_s": elapsed_seconds(item["last_decode_verify_ts"], origin),
                "wait_event_count": item["wait_event_count"],
                "max_waiting_seconds": item["max_waiting_seconds"],
                "rdma_count": item["rdma_count"],
                "alloc_pages": item["alloc_pages"],
                "alloc_chunks": item["alloc_chunks"],
            }
        )
    rows.sort(
        key=lambda row: (
            row.get("room") if row.get("room") is not None else -1,
            row.get("chunk_idx") if row.get("chunk_idx") is not None else -1,
        )
    )
    return rows


def aggregate_rooms(
    events: Sequence[Dict[str, object]],
    room_to_rid: Dict[int, str],
    origin: Optional[float],
) -> List[Dict[str, object]]:
    groups: Dict[int, Dict[str, object]] = {}
    for event in events:
        room = event.get("room")
        if room is None:
            continue
        item = groups.setdefault(
            room,
            {
                "room": room,
                "rid": room_to_rid.get(room),
                "prefill_tps": set(),
                "decode_tps": set(),
                "sessions": set(),
                "chunk_indices": set(),
                "first_event_ts": None,
                "last_event_ts": None,
                "wait_event_count": 0,
                "rdma_event_count": 0,
                "wm_send_count": 0,
                "wm_recv_count": 0,
                "wm_free_count": 0,
                "e2e_ok_count": 0,
                "post_scatter_ok_count": 0,
            },
        )
        if event.get("prefill_tp") is not None:
            item["prefill_tps"].add(event["prefill_tp"])
        if event.get("decode_tp") is not None:
            item["decode_tps"].add(event["decode_tp"])
        if event.get("session"):
            item["sessions"].add(event["session"])
        if event.get("chunk_idx") is not None:
            item["chunk_indices"].add(event["chunk_idx"])
        ts = event.get("timestamp_epoch")
        item["first_event_ts"] = (
            ts if item["first_event_ts"] is None else min(item["first_event_ts"], ts)
        )
        item["last_event_ts"] = (
            ts if item["last_event_ts"] is None else max(item["last_event_ts"], ts)
        )
        name = event["event_type"]
        if name == "WM-DEFER":
            item["wait_event_count"] += 1
        elif name == "STG-RDMA":
            item["rdma_event_count"] += 1
        elif name == "WM-SEND":
            item["wm_send_count"] += 1
        elif name == "WM-RECV":
            item["wm_recv_count"] += 1
        elif name == "WM-FREE":
            item["wm_free_count"] += 1
        elif name == "E2E-OK":
            item["e2e_ok_count"] += 1
        elif name == "POST-SCATTER-OK":
            item["post_scatter_ok_count"] += 1

    rows = []
    for item in groups.values():
        chunk_indices = collapse_sorted(item["chunk_indices"])
        rows.append(
            {
                "room": item["room"],
                "rid": item["rid"],
                "event_start": format_clock(item["first_event_ts"]),
                "event_end": format_clock(item["last_event_ts"]),
                "event_start_elapsed_s": elapsed_seconds(item["first_event_ts"], origin),
                "event_end_elapsed_s": elapsed_seconds(item["last_event_ts"], origin),
                "lifespan_seconds": (
            round(item["last_event_ts"] - item["first_event_ts"], 6)
            if item["first_event_ts"] is not None and item["last_event_ts"] is not None
            else None
                ),
                "chunk_count": len(chunk_indices.split(",")) if chunk_indices else 0,
                "chunk_indices": chunk_indices,
                "prefill_tps": collapse_sorted(item["prefill_tps"]),
                "decode_tps": collapse_sorted(item["decode_tps"]),
                "sessions": collapse_sorted(item["sessions"]),
                "wait_event_count": item["wait_event_count"],
                "rdma_event_count": item["rdma_event_count"],
                "wm_send_count": item["wm_send_count"],
                "wm_recv_count": item["wm_recv_count"],
                "wm_free_count": item["wm_free_count"],
                "e2e_ok_count": item["e2e_ok_count"],
                "post_scatter_ok_count": item["post_scatter_ok_count"],
            }
        )
    rows.sort(key=lambda row: row["room"])
    return rows


def build_rdma_chunk_lookup(
    events: Sequence[Dict[str, object]]
) -> Dict[Tuple[object, object, object, object, object, object], object]:
    lookup: Dict[Tuple[object, object, object, object, object, object], object] = {}
    for event in events:
        if event["event_type"] != "STG-RDMA":
            continue
        key = (
            event.get("room"),
            event.get("prefill_tp"),
            event.get("decode_tp"),
            event.get("session"),
            event.get("round"),
            event.get("end"),
        )
        lookup[key] = event.get("chunk_idx")
    return lookup


def resolve_chunk_idx(
    event: Dict[str, object],
    rdma_lookup: Dict[Tuple[object, object, object, object, object, object], object],
):
    if event.get("chunk_idx") is not None:
        return event.get("chunk_idx")
    if event.get("event_type") != "WM-DEFER":
        return event.get("chunk_idx")
    key = (
        event.get("room"),
        event.get("prefill_tp"),
        event.get("decode_tp"),
        event.get("session"),
        event.get("need_round"),
        event.get("need_end"),
    )
    return rdma_lookup.get(key)


def watermark_rows(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    origin = compute_time_origin(events)
    rows = []
    for event in events:
        if event["event_type"] not in {"WM-RECV", "WM-SEND", "WM-FREE"}:
            continue
        rows.append(
            {
                "timestamp_iso": event.get("timestamp_iso"),
                "time": format_clock(event.get("timestamp_epoch")),
                "elapsed_s": elapsed_seconds(event.get("timestamp_epoch"), origin),
                "source": event.get("source"),
                "host": event.get("host"),
                "worker_id": event.get("worker_id"),
                "event_type": event.get("event_type"),
                "room": event.get("room"),
                "session": event.get("session"),
                "prefill_tp": event.get("prefill_tp"),
                "decode_tp": event.get("decode_tp"),
                "wm_round": event.get("wm_round"),
                "wm_tail": event.get("wm_tail"),
                "prev_round": event.get("prev_round"),
                "prev_tail": event.get("prev_tail"),
                "before_round": event.get("before_round"),
                "before_head": event.get("before_head"),
                "after_round": event.get("after_round"),
                "after_head": event.get("after_head"),
                "alloc_id": event.get("alloc_id"),
                "raw_message": event.get("raw_message"),
            }
        )
    return rows


def metadata_rows(log_dir: Path, events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    files = discover_log_files(log_dir)
    counts = defaultdict(int)
    rooms = set()
    sessions = set()
    for event in events:
        counts[event["event_type"]] += 1
        if event.get("room") is not None:
            rooms.add(event["room"])
        if event.get("session"):
            sessions.add(event["session"])
    rows = [
        {"key": "log_dir", "value": str(log_dir)},
        {"key": "log_file_count", "value": len(files)},
        {"key": "event_count", "value": len(events)},
        {"key": "room_count", "value": len(rooms)},
        {"key": "session_count", "value": len(sessions)},
    ]
    for event_type in sorted(counts):
        rows.append({"key": f"event_type:{event_type}", "value": counts[event_type]})
    return rows


def column_name(index: int) -> str:
    letters = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def sanitize_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[:\\/?*\[\]]", "_", name)[:31] or "Sheet"
    candidate = cleaned
    counter = 1
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{cleaned[:31-len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def xml_text(text: str) -> str:
    escaped = escape(text)
    preserve = text != text.strip() or "\n" in text
    if preserve:
        return f'<is><t xml:space="preserve">{escaped}</t></is>'
    return f"<is><t>{escaped}</t></is>"


def build_sheet_xml(rows: List[Dict[str, object]]) -> str:
    if rows:
        headers = list(rows[0].keys())
        for row in rows[1:]:
            for key in row.keys():
                if key not in headers:
                    headers.append(key)
    else:
        headers = ["empty"]
        rows = [{"empty": ""}]

    xml_rows: List[str] = []
    all_rows = [{key: key for key in headers}] + rows
    for row_idx, row in enumerate(all_rows, start=1):
        cells: List[str] = []
        for col_idx, header in enumerate(headers, start=1):
            value = row.get(header)
            if value is None:
                continue
            ref = f"{column_name(col_idx)}{row_idx}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr">{xml_text(stringify(value))}</c>')
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(path: Path, sheets: Sequence[Tuple[str, List[Dict[str, object]]]]) -> None:
    used_names: set[str] = set()
    sheet_entries = []
    for index, (name, rows) in enumerate(sheets, start=1):
        sheet_name = sanitize_sheet_name(name, used_names)
        sheet_entries.append((index, sheet_name, rows))

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for index, _, _ in sheet_entries:
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    for index, sheet_name, _ in sheet_entries:
        workbook.append(
            f'<sheet name="{escape(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
        )
    workbook.extend(["</sheets>", "</workbook>"])

    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index, _, _ in sheet_entries:
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    workbook_rels.append("</Relationships>")

    package_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>staging log analysis</dc:title>"
        "<dc:creator>staging_log_to_excel.py</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>'
        "</cp:coreProperties>"
    )

    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Python</Application>"
        "</Properties>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", package_rels)
        zf.writestr("xl/workbook.xml", "".join(workbook))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        for index, _, rows in sheet_entries:
            zf.writestr(f"xl/worksheets/sheet{index}.xml", build_sheet_xml(rows))


def resolve_paths(args) -> Tuple[Path, Path]:
    script_root = Path(__file__).resolve().parent.parent
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = script_root / run_dir
        log_dir = discover_log_dir(run_dir)
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = script_root / log_dir
    if not log_dir.exists():
        raise FileNotFoundError(f"log directory does not exist: {log_dir}")
    output = Path(args.output) if args.output else log_dir / "staging_timeline.xlsx"
    if not output.is_absolute():
        output = script_root / output
    return log_dir, output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse staging-related logs and export an Excel workbook."
    )
    parser.add_argument(
        "--run-dir",
        help="Run directory that contains config.yaml and a logs/ subdirectory.",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory that directly contains *_prefill_w*.out and *_decode_w*.out files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output .xlsx path. Defaults to <log-dir>/staging_timeline.xlsx",
    )
    args = parser.parse_args()

    if not args.run_dir and not args.log_dir:
        parser.error("one of --run-dir or --log-dir is required")

    log_dir, output_path = resolve_paths(args)
    parser_obj = LogParser(log_dir)
    events = parser_obj.parse()
    if not events:
        print(f"No staging events found under {log_dir}", file=sys.stderr)
        return 1
    origin = compute_time_origin(events)

    sheets = [
        (
            "README",
            [
                {
                    "sheet": "README",
                    "description": "Workbook generated from staging-related log lines only.",
                },
                {"sheet": "metadata", "description": "Run-level counters and file stats."},
                {"sheet": "events", "description": "Raw parsed events, one row per log line."},
                {
                    "sheet": "chunk_rank",
                    "description": "Prefill chunk timeline keyed by room + chunk_idx + prefill_tp.",
                },
                {
                    "sheet": "room_chunk",
                    "description": "Chunk summary merged across TP ranks.",
                },
                {"sheet": "rooms", "description": "Request-level summary keyed by room."},
                {"sheet": "watermarks", "description": "All watermark send/recv/free events."},
            ],
        ),
        ("metadata", metadata_rows(log_dir, events)),
        ("events", public_event_rows(events, origin)),
        ("chunk_rank", aggregate_chunk_rank(events, origin)),
        ("room_chunk", aggregate_room_chunk(events, parser_obj.room_to_rid, origin)),
        ("rooms", aggregate_rooms(events, parser_obj.room_to_rid, origin)),
        ("watermarks", watermark_rows(events)),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(output_path, sheets)
    print(f"Wrote {output_path}")
    print(f"Events: {len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
