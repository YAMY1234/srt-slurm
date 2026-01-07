# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment configuration constants and setup."""

import logging
import os

# Base network configurations
ETCD_CLIENT_PORT_BASE = 2379
ETCD_PEER_PORT_BASE = 2380
NATS_PORT_BASE = 4222
DIST_INIT_PORT = 29500
ETCD_LISTEN_ADDR = "http://0.0.0.0"

# Port offset multiplier (each job gets a unique offset based on job ID)
PORT_OFFSET_MULTIPLIER = 10


def get_port_offset() -> int:
    """
    Get port offset based on SLURM_JOB_ID to avoid port conflicts
    when multiple jobs run on the same node.
    
    Returns:
        Port offset (0 if not running under SLURM, otherwise based on job ID)
    """
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    try:
        # Use modulo to keep offset reasonable (max 1000 different offsets)
        offset = (int(job_id) % 100) * PORT_OFFSET_MULTIPLIER
        return offset
    except ValueError:
        return 0


def get_etcd_client_port() -> int:
    """Get ETCD client port with job-specific offset."""
    return ETCD_CLIENT_PORT_BASE + get_port_offset()


def get_etcd_peer_port() -> int:
    """Get ETCD peer port with job-specific offset."""
    return ETCD_PEER_PORT_BASE + get_port_offset()


def get_nats_port() -> int:
    """Get NATS port with job-specific offset."""
    return NATS_PORT_BASE + get_port_offset()


def setup_env(master_ip: str):
    """Setup NATS and ETCD environment variables."""
    etcd_port = get_etcd_client_port()
    nats_port = get_nats_port()
    
    nats_server = f"nats://{master_ip}:{nats_port}"
    etcd_endpoints = f"http://{master_ip}:{etcd_port}"

    os.environ["NATS_SERVER"] = nats_server
    os.environ["ETCD_ENDPOINTS"] = etcd_endpoints

    logging.info(f"set NATS_SERVER: {nats_server}")
    logging.info(f"set ETCD_ENDPOINTS: {etcd_endpoints}")
    logging.info(f"Port offset for this job: {get_port_offset()} (SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')})")
