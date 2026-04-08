# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest runner for unit/integration tests inside the container."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("pytest")
class PytestRunner(BenchmarkRunner):
    """Run pytest inside the container.

    Does not require a running server. Useful for unit tests that
    verify staging buffer logic, head-slice computation, etc.

    Required config fields:
        - benchmark.test_path: Path to test file or directory (relative to /sgl-workspace/sglang)

    Optional config fields:
        - benchmark.pytest_args: Additional pytest arguments (default: "-v")
    """

    needs_server = False

    @property
    def name(self) -> str:
        return "Pytest"

    @property
    def script_path(self) -> str:
        return "python3"

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        if not getattr(config.benchmark, "test_path", None):
            errors.append("benchmark.test_path is required for pytest runner")
        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        test_path = b.test_path
        pytest_args = getattr(b, "pytest_args", "-v") or "-v"

        return [
            "python3", "-m", "pytest",
            f"/sgl-workspace/sglang/{test_path}",
            *pytest_args.split(),
        ]
