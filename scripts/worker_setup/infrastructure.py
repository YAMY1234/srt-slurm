# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure setup for NATS, ETCD, Nginx, and frontend workers."""

import logging
import os

from .command import install_dynamo_wheels
from .environment import ETCD_LISTEN_ADDR, get_etcd_client_port, get_etcd_peer_port, get_nats_port
from .utils import run_command, wait_for_etcd


def setup_head_prefill_node(prefill_host_ip: str) -> None:
    """Setup NATS and ETCD on the prefill host node."""
    nats_port = get_nats_port()
    etcd_client_port = get_etcd_client_port()
    etcd_peer_port = get_etcd_peer_port()
    
    # Get unique data directory based on SLURM_JOB_ID to avoid conflicts
    job_id = os.environ.get("SLURM_JOB_ID", "default")
    etcd_data_dir = f"/tmp/etcd_data_{job_id}"
    
    # Clean up any existing data directory to ensure fresh start
    import shutil
    if os.path.exists(etcd_data_dir):
        logging.info(f"Cleaning up existing etcd data directory: {etcd_data_dir}")
        shutil.rmtree(etcd_data_dir, ignore_errors=True)
    
    logging.info(f"Starting nats server on node {prefill_host_ip} (port: {nats_port})")
    nats_cmd = f"/configs/nats-server -js -p {nats_port}"
    nats_process = run_command(nats_cmd, background=True)
    if not nats_process:
        raise RuntimeError("Failed to start nats-server")

    logging.info(f"Starting etcd server on node {prefill_host_ip} (client: {etcd_client_port}, peer: {etcd_peer_port}, data-dir: {etcd_data_dir})")
    etcd_cmd = (
        f"/configs/etcd --listen-client-urls {ETCD_LISTEN_ADDR}:{etcd_client_port} "
        f"--advertise-client-urls http://{prefill_host_ip}:{etcd_client_port} "
        f"--listen-peer-urls {ETCD_LISTEN_ADDR}:{etcd_peer_port} "
        f"--initial-cluster default=http://{prefill_host_ip}:{etcd_peer_port} "
        f"--data-dir {etcd_data_dir}"
    )

    etcd_process = run_command(etcd_cmd, background=True)
    if not etcd_process:
        raise RuntimeError("Failed to start etcd")


def setup_nginx_worker(nginx_config: str) -> int:
    """Setup nginx load balancer"""
    logging.info("Setting up nginx load balancer")

    if not nginx_config or not os.path.exists(nginx_config):
        raise ValueError(f"Nginx config file not found: {nginx_config}")

    nginx_cmd = f"apt-get update && apt-get install -y nginx && nginx -c {nginx_config} && sleep 86400"
    return run_command(nginx_cmd)


def setup_frontend_worker(worker_idx: int, master_ip: str, gpu_type: str) -> int:
    """Setup a frontend worker"""
    logging.info(f"Setting up frontend worker {worker_idx}")
    
    etcd_client_port = get_etcd_client_port()

    # First frontend (worker_idx 0) also sets up NATS/ETCD
    if worker_idx == 0:
        setup_head_prefill_node(master_ip)
        if not wait_for_etcd(f"http://{master_ip}:{etcd_client_port}"):
            raise RuntimeError("Failed to connect to etcd")
    else:
        logging.info(f"Setting up additional frontend worker {worker_idx}")
        if not wait_for_etcd(f"http://{master_ip}:{etcd_client_port}"):
            raise RuntimeError("Failed to connect to etcd")

    # Install dynamo from PyPI
    install_dynamo_wheels(gpu_type)

    # Run frontend
    frontend_cmd = "python3 -m dynamo.frontend --http-port=8000"
    return run_command(frontend_cmd)
