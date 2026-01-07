#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPQA evaluation script using sglang.test.run_eval with gpqa

# Set HF_TOKEN to avoid rate limiting when downloading tokenizer
# You can override this by setting HF_TOKEN environment variable before running
if [ -z "$HF_TOKEN" ]; then
    # Default token - replace with your own or set HF_TOKEN env var
    export HF_TOKEN="${HF_TOKEN:-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}"
    echo "Warning: HF_TOKEN not set. Using default token. Set HF_TOKEN env var to use your own."
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

head_node="localhost"
head_port=8000
# Model name must match served-model-name in sglang config
# Use MODEL_NAME env var if set, otherwise default to nvidia FP4 model
model_name="${MODEL_NAME:-nvidia/DeepSeek-R1-0528-NVFP4-v2}"

# Parse arguments from SLURM job
n_prefill=$1
n_decode=$2
# prefill_gpus and decode_gpus are parsed for argument position consistency with other benchmarks
# shellcheck disable=SC2034
prefill_gpus=$3
# shellcheck disable=SC2034
decode_gpus=$4
num_examples=${5:-198}  # Default: 198
max_tokens=${6:-16384}   # Default: 16384 (R1 needs more tokens for thinking)
repeat=${7:-1}          # Default: 1
num_threads=${8:-512}   # Default: 512
thinking_mode=${9:-deepseek-r1} # Default: deepseek-r1

echo "GPQA Benchmark Config: num_examples=${num_examples}; max_tokens=${max_tokens}; repeat=${repeat}; num_threads=${num_threads}; thinking-mode=${thinking_mode}"

# Source utilities for wait_for_model
source /scripts/utils/benchmark_utils.sh

wait_for_model_timeout=1500 # 25 minutes
wait_for_model_check_interval=5 # check interval -> 5s
wait_for_model_report_interval=60 # wait_for_model report interval -> 60s

wait_for_model $head_node $head_port $n_prefill $n_decode $wait_for_model_check_interval $wait_for_model_timeout $wait_for_model_report_interval

# Create results directory
result_dir="/logs/accuracy"
mkdir -p $result_dir

echo "Running GPQA evaluation..."

# Set OPENAI_API_KEY if not set
if [ -z "$OPENAI_API_KEY" ]; then
    export OPENAI_API_KEY="EMPTY"
fi

# Run the evaluation
# Note: --thinking-mode removed because dynamo frontend doesn't support chat_template_kwargs
python3 -m sglang.test.run_eval \
    --base-url "http://${head_node}:${head_port}" \
    --model ${model_name} \
    --eval-name gpqa \
    --num-examples ${num_examples} \
    --max-tokens ${max_tokens} \
    --repeat ${repeat} \
    --num-threads ${num_threads}

# Copy the result file from /tmp to our logs directory
# The result file is named gpqa_{model_name}.json
result_file=$(ls -t /tmp/gpqa_*.json 2>/dev/null | head -n1)

if [ -f "$result_file" ]; then
    cp "$result_file" "$result_dir/"
    echo "Results saved to: $result_dir/$(basename $result_file)"
else
    echo "Warning: Could not find result file in /tmp"
fi

echo "GPQA evaluation complete"
