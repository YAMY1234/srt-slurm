# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main orchestration script for benchmark sweeps.

This script is called from within the sbatch job and coordinates:
1. Starting head node infrastructure (NATS, etcd)
2. Starting backend workers (prefill/decode/agg)
3. Starting frontends and nginx
4. Running benchmarks
5. Cleanup
"""

import argparse
import functools
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from srtctl.cli.mixins import BenchmarkStageMixin, FrontendStageMixin, PostProcessStageMixin, WorkerStageMixin
from srtctl.core.config import load_config
from srtctl.core.health import wait_for_port
from srtctl.core.processes import (
    ManagedProcess,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.slurm import get_port_offset, get_slurm_job_id, start_srun_process
from srtctl.core.status import JobStage, JobStatus, StatusReporter
from srtctl.core.topology import Endpoint, NodePortAllocator, Process
from srtctl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class SweepOrchestrator(WorkerStageMixin, FrontendStageMixin, BenchmarkStageMixin, PostProcessStageMixin):
    """Main orchestrator for benchmark sweeps.

    Usage:
        config = load_config(config_path)  # Returns typed SrtConfig
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config, runtime)
        exit_code = orchestrator.run()
    """

    config: SrtConfig
    runtime: RuntimeContext

    @property
    def backend(self):
        """Access the backend config (implements BackendProtocol)."""
        return self.config.backend

    @functools.cached_property
    def endpoints(self) -> list[Endpoint]:
        """Compute endpoint allocation topology (cached).

        This is the single source of truth for endpoint assignments.
        """
        r = self.config.resources
        return self.backend.allocate_endpoints(
            num_prefill=r.num_prefill,
            num_decode=r.num_decode,
            num_agg=r.num_agg,
            gpus_per_prefill=r.gpus_per_prefill,
            gpus_per_decode=r.gpus_per_decode,
            gpus_per_agg=r.gpus_per_agg,
            gpus_per_node=r.gpus_per_node,
            available_nodes=self.runtime.nodes.worker,
        )

    @functools.cached_property
    def backend_processes(self) -> list[Process]:
        """Compute physical process topology from endpoints (cached)."""
        # DYN_SYSTEM_PORT is parsed as i16 by dynamo runtime, so keep ports < 32768.
        # Also avoid collisions across concurrent jobs by offsetting from the job id.
        #
        # Note: srtctl allocates one sys_port per Process and increments sequentially from base_sys_port.
        # Therefore, base_sys_port must reserve a sufficiently large consecutive port window per job
        # to avoid collisions with other jobs running concurrently.
        #
        # Use get_port_offset() for consistency with other services (NATS, etcd, frontend).
        # get_port_offset returns 0-990 in steps of 10, giving 100 slots.
        # Each slot needs ~200 ports for sys_port allocation, so we multiply offset by 20.
        port_offset = get_port_offset(self.runtime.job_id)
        sys_port_stride = 200  # Reserved consecutive sys ports per job.
        base_sys_port = 9000 + (port_offset * 20)  # Range: 9000-28800, step 200

        port_allocator: NodePortAllocator | None = None
        if self.config.frontend.type == "sglang" and getattr(self.backend, "type", None) == "sglang":
            prefill_cfg: dict[str, object] = {}
            try:
                prefill_cfg = self.backend.get_config_for_mode("prefill")  # type: ignore[assignment]
            except Exception:
                prefill_cfg = {}

            user_bootstrap_port = prefill_cfg.get("disaggregation-bootstrap-port")
            if user_bootstrap_port is not None:
                try:
                    base_bootstrap_port = int(user_bootstrap_port)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid disaggregation-bootstrap-port=%r; falling back to default bootstrap port allocation",
                        user_bootstrap_port,
                    )
                else:
                    port_allocator = NodePortAllocator(base_bootstrap_port=base_bootstrap_port)

        processes = self.backend.endpoints_to_processes(
            self.endpoints,
            base_sys_port=base_sys_port,
            port_allocator=port_allocator,
        )
        if len(processes) > sys_port_stride:
            logger.warning(
                "This job allocates %d processes, which may exceed the reserved sys_port window (%d). "
                "Consider increasing sys_port_stride to reduce cross-job collision risk.",
                len(processes),
                sys_port_stride,
            )
        return processes

    def start_head_infrastructure(self, registry: ProcessRegistry) -> ManagedProcess | None:
        """Start head node infrastructure when required by the chosen frontend.

        Dynamo frontend requires NATS+etcd for discovery/control planes.
        SGLang frontend uses direct worker connections and does not require these services.
        """
        if self.config.frontend.type != "dynamo":
            logger.info(
                "Skipping head node infrastructure (frontend.type=%s does not require NATS/etcd)",
                self.config.frontend.type,
            )
            return None

        infra_node = self.runtime.nodes.infra
        logger.info("Starting infrastructure services (NATS, etcd)")
        logger.info("Infra node: %s", infra_node)

        setup_script = Path(__file__).parent / "setup_head.py"
        if not setup_script.exists():
            raise RuntimeError(f"setup_head.py not found at {setup_script}")

        setup_script_container = Path("/tmp/setup_head.py")
        infra_log = self.runtime.log_dir / "infra.out"

        cmd = [
            "python3",
            str(setup_script_container),
            "--name",
            self.config.name,
            "--log-dir",
            str(self.runtime.log_dir),
        ]

        mounts = dict(self.runtime.container_mounts)
        mounts[setup_script] = setup_script_container
        # Mount host /tmp to container /host-tmp for etcd/nats data on local storage
        # This ensures etcd WAL writes go to fast local disk, not network storage
        mounts[Path("/tmp")] = Path("/host-tmp")

        proc = start_srun_process(
            command=cmd,
            nodelist=[infra_node],
            output=str(infra_log),
            container_image=str(self.runtime.container_image),
            container_mounts=mounts,
        )

        managed = ManagedProcess(
            name="infra_services",
            popen=proc,
            log_file=infra_log,
            node=infra_node,
            critical=True,
        )

        port_offset = get_port_offset(self.runtime.job_id)
        nats_port = 4222 + port_offset
        etcd_port = 2379 + port_offset
        logger.info("Port offset for this job: %d (job_id: %s)", port_offset, self.runtime.job_id)

        # 300s timeout to handle slow container imports on first run
        logger.info("Waiting for NATS (port %d) on %s...", nats_port, infra_node)
        if not wait_for_port(infra_node, nats_port, timeout=300):
            raise RuntimeError("NATS failed to start")
        logger.info("NATS is ready")

        logger.info("Waiting for etcd (port %d) on %s...", etcd_port, infra_node)
        if not wait_for_port(infra_node, etcd_port, timeout=300):
            raise RuntimeError("etcd failed to start")
        logger.info("etcd is ready")

        return managed

    def _print_connection_info(self) -> None:
        """Print srun commands for connecting to nodes."""
        container_args = f"--container-image={self.runtime.container_image}"
        mounts_str = ",".join(f"{src}:{dst}" for src, dst in self.runtime.container_mounts.items())
        if mounts_str:
            container_args += f" --container-mounts={mounts_str}"

        public_port = self.runtime.frontend_port

        logger.info("")
        logger.info("=" * 60)
        logger.info("Connection Commands")
        logger.info("=" * 60)
        logger.info("Frontend URL: http://%s:%d", self.runtime.nodes.head, public_port)
        logger.info("")
        logger.info("To connect to head node (%s):", self.runtime.nodes.head)
        logger.info(
            "  srun %s --jobid %s -w %s --overlap --pty bash",
            container_args,
            self.runtime.job_id,
            self.runtime.nodes.head,
        )

        # Print worker node connection commands
        for node in self.runtime.nodes.worker:
            if node != self.runtime.nodes.head:
                logger.info("")
                logger.info("To connect to worker node (%s):", node)
                logger.info(
                    "  srun %s --jobid %s -w %s --overlap --pty bash",
                    container_args,
                    self.runtime.job_id,
                    node,
                )

        logger.info("=" * 60)
        logger.info("")

    def run(self) -> int:
        """Run the complete sweep."""
        # Create status reporter (fire-and-forget, no-op if not configured)
        reporter = StatusReporter.from_config(self.config.reporting, self.runtime.job_id)
        reporter.report_started(self.config, self.runtime)

        logger.info("Sweep Orchestrator")
        logger.info("Job ID: %s", self.runtime.job_id)
        logger.info("Run name: %s", self.runtime.run_name)
        logger.info("Config: %s", self.config.name)
        logger.info("Infra node: %s", self.runtime.nodes.infra)
        logger.info("Head node: %s", self.runtime.nodes.head)
        logger.info("Worker nodes: %s", ", ".join(self.runtime.nodes.worker))
        if self.config.profiling.enabled:
            logger.info("Profiling: %s", self.config.profiling.type)

        registry = ProcessRegistry(job_id=self.runtime.job_id)
        stop_event = threading.Event()
        setup_signal_handlers(stop_event, registry)
        start_process_monitor(stop_event, registry)

        exit_code = 1

        try:
            # Stage 1: Head infrastructure (NATS, etcd)
            reporter.report(JobStatus.STARTING, JobStage.HEAD_INFRASTRUCTURE, "Starting head infrastructure")
            head_proc = self.start_head_infrastructure(registry)
            if head_proc is not None:
                registry.add_process(head_proc)

            # Stage 2: Workers
            reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
            worker_procs = self.start_all_workers()
            registry.add_processes(worker_procs)

            # Stage 3: Frontend
            reporter.report(JobStatus.FRONTEND, JobStage.FRONTEND, "Starting frontend")
            frontend_procs = self.start_frontend(registry)
            for proc in frontend_procs:
                registry.add_process(proc)

            self._print_connection_info()

            # Stage 4: Benchmark (status reported AFTER health check passes)
            exit_code = self.run_benchmark(registry, stop_event, reporter)

        except Exception as e:
            logger.exception("Error during sweep: %s", e)
            reporter.report(JobStatus.FAILED, JobStage.CLEANUP, str(e))
            exit_code = 1

        finally:
            logger.info("Cleanup")
            reporter.report_completed(exit_code)
            stop_event.set()
            registry.cleanup()
            if exit_code != 0:
                registry.print_failure_details()
            # Run post-processing (AI analysis if enabled)
            self.run_postprocess(exit_code)

        return exit_code


def main():
    """Main entry point."""
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Run benchmark sweep")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    setup_logging()

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)

        config = load_config(config_path)

        # Check for setup_script override from CLI (passed via env var)
        setup_script_override = os.environ.get("SRTCTL_SETUP_SCRIPT")
        if setup_script_override:
            logger.info("Setup script override: %s", setup_script_override)
            config = replace(config, setup_script=setup_script_override)

        job_id = get_slurm_job_id()
        if not job_id:
            logger.error("Not running in SLURM (SLURM_JOB_ID not set)")
            sys.exit(1)

        # Type narrowing: job_id is str after the check above
        assert job_id is not None
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        exit_code = orchestrator.run()

        sys.exit(exit_code)

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
