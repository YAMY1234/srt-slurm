# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K-Bench runner (original bench_sglang.py style).

Unlike the 'gsm8k' type which uses sglang.test.run_eval (Chat Completions API),
this runner uses /v1/completions (raw text, no chat template) by default,
matching the original sglang/benchmark/gsm8k/bench_sglang.py behavior.

Set benchmark.use_chat_api: true to switch to /v1/chat/completions.

Optional config fields:
    - benchmark.num_examples: Number of questions (default: 1319)
    - benchmark.num_shots: Few-shot examples (default: 5)
    - benchmark.max_tokens: Max new tokens per response (default: 512)
    - benchmark.num_threads: Concurrent requests (default: 64)
    - benchmark.temperature: Sampling temperature (default: 0.0)
    - benchmark.top_p: Nucleus sampling threshold (default: 1.0)
    - benchmark.use_chat_api: Use /v1/chat/completions instead (default: false)
    - benchmark.platinum: Use GSM8K Platinum dataset (default: false)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("gsm8k-bench")
class GSM8KBenchRunner(BenchmarkRunner):
    """GSM8K benchmark via /v1/completions (no chat template by default)."""

    @property
    def name(self) -> str:
        return "GSM8K-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/gsm8k-bench/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "gsm8k-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        return []

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        use_chat_api = "true" if getattr(b, "use_chat_api", False) else ""
        platinum = "true" if getattr(b, "platinum", False) else ""

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.num_examples or 1319),
            str(b.num_shots or 5),
            str(b.max_tokens or 512),
            str(b.num_threads or 64),
            str(getattr(b, "temperature", None) or "0.0"),
            str(getattr(b, "top_p", None) or "1.0"),
            use_chat_api,
            platinum,
        ]
