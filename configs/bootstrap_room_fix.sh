#!/usr/bin/env bash
# setup_script for srtslurm: applies bootstrap_room sign-bit fix + Dynamo 0.8.1 compat shim.
#
# Place this in the srtslurm configs/ directory on the cluster:
#   cp bootstrap_room_fix.sh /mnt/lustre01/users/slurm-shared/ishan/srt-slurm/configs/
#
# Then in your recipe YAML:
#   setup_script: bootstrap_room_fix.sh
#
# Runs inside the container before workers start. SGLang source is at /sgl-workspace/sglang.

set -euo pipefail

SGLANG_ROOT="/sgl-workspace/sglang"

echo "[bootstrap_room_fix] Patching SGLang at ${SGLANG_ROOT} ..."

# Use Python for all patches — sed is too fragile for multi-line edits
python3 << 'PYTHON_PATCH'
import os, sys

sglang_root = "/sgl-workspace/sglang"

# --- 1. decode.py: mask sign bit when computing expected_room ---
decode_file = os.path.join(sglang_root, "python/sglang/srt/disaggregation/decode.py")
with open(decode_file) as f:
    content = f.read()

old_decode = """\
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )"""

new_decode = """\
        raw_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )
        expected_room = raw_room & 0x7FFFFFFFFFFFFFFF"""

if old_decode in content:
    content = content.replace(old_decode, new_decode)
    with open(decode_file, "w") as f:
        f.write(content)
    print("[bootstrap_room_fix] decode.py patched")
elif "raw_room = (" in content:
    print("[bootstrap_room_fix] decode.py already patched, skipping")
else:
    print("[bootstrap_room_fix] WARNING: decode.py pattern not found", file=sys.stderr)

# --- 2. utils.py: mask sign bit when storing bootstrap_room ---
utils_file = os.path.join(sglang_root, "python/sglang/srt/disaggregation/utils.py")
with open(utils_file) as f:
    content = f.read()

old_utils = """\
        # Store bootstrap_room for validation on decode side
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            req.bootstrap_room if req.bootstrap_room is not None else 0
        )"""

new_utils = """\
        # Store bootstrap_room for validation on decode side
        room_val = req.bootstrap_room if req.bootstrap_room is not None else 0
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            room_val & 0x7FFFFFFFFFFFFFFF
        )"""

if old_utils in content:
    content = content.replace(old_utils, new_utils)
    with open(utils_file, "w") as f:
        f.write(content)
    print("[bootstrap_room_fix] utils.py patched")
elif "room_val = req.bootstrap_room" in content:
    print("[bootstrap_room_fix] utils.py already patched, skipping")
else:
    print("[bootstrap_room_fix] WARNING: utils.py pattern not found", file=sys.stderr)

# --- 3. utils/__init__.py: add Dynamo 0.8.1 compat shim ---
init_file = os.path.join(sglang_root, "python/sglang/srt/utils/__init__.py")
with open(init_file) as f:
    content = f.read()

if "maybe_wrap_ipv6_address" not in content:
    content += '''

# Re-export network utilities for backward compatibility with Dynamo 0.8.1
from sglang.srt.utils.network import (
    get_local_ip_auto,
    get_zmq_socket,
    get_zmq_socket_on_host,
)


def maybe_wrap_ipv6_address(address: str) -> str:
    """Compatibility shim for Dynamo 0.8.1 (removed in dev-0401)."""
    import ipaddress
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ipaddress.AddressValueError:
        return address
'''
    with open(init_file, "w") as f:
        f.write(content)
    print("[bootstrap_room_fix] utils/__init__.py patched")
else:
    print("[bootstrap_room_fix] utils/__init__.py already patched, skipping")

# --- 4. sglang.srt.tracing package stub for Dynamo 0.8.1 ---
tracing_dir = os.path.join(sglang_root, "python/sglang/srt/tracing")
if not os.path.isdir(tracing_dir):
    os.makedirs(tracing_dir, exist_ok=True)
    with open(os.path.join(tracing_dir, "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(tracing_dir, "trace.py"), "w") as f:
        f.write('''\
"""Compatibility shim: sglang.srt.tracing was moved to sglang.srt.observability in dev-0401.
Provides no-op stubs for Dynamo 0.8.1 compatibility."""

def trace_set_remote_propagate_context(base64_str):
    pass

def trace_set_thread_info(*args, **kwargs):
    pass
''')
    print("[bootstrap_room_fix] tracing/ package stub created")
else:
    print("[bootstrap_room_fix] tracing/ already exists, skipping")

# --- 5. Fix dynamo stream_output -> incremental_streaming_output for dev-0401 SGLang ---
# Old dynamo sets server_args.stream_output=True but dev-0401 SGLang renamed it to
# incremental_streaming_output. Patch dynamo's args.py to use the correct flag name.
import glob
args_candidates = glob.glob("/sgl-workspace/dynamo/components/src/dynamo/sglang/args.py") + \
                  glob.glob("/usr/local/lib/python3.12/dist-packages/dynamo/sglang/args.py")
for args_file in args_candidates:
    try:
        with open(args_file) as f:
            content = f.read()
        if "stream_output" in content and "incremental_streaming_output" not in content:
            content = content.replace("stream_output", "incremental_streaming_output")
            with open(args_file, "w") as f:
                f.write(content)
            print(f"[bootstrap_room_fix] Patched stream_output -> incremental_streaming_output in {args_file}")
        else:
            print(f"[bootstrap_room_fix] {args_file} already correct or not applicable, skipping")
    except Exception as e:
        print(f"[bootstrap_room_fix] Could not patch {args_file}: {e}", file=sys.stderr)

print("[bootstrap_room_fix] Done.")
PYTHON_PATCH