#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GSM8K accuracy evaluation
# Expects: endpoint [num_examples] [max_tokens] [num_threads] [num_shots]

set -e

ENDPOINT=$1
NUM_EXAMPLES=${2:-1319}
MAX_TOKENS=${3:-16384}
NUM_THREADS=${4:-512}
NUM_SHOTS=${5:-5}

echo "GSM8K Config: endpoint=${ENDPOINT}; num_examples=${NUM_EXAMPLES}; max_tokens=${MAX_TOKENS}; num_threads=${NUM_THREADS}; num_shots=${NUM_SHOTS}"

# Create results directory
result_dir="/logs/accuracy"
mkdir -p "$result_dir"

# Set OPENAI_API_KEY if not set
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "Running GSM8K evaluation..."

# Note: --model is omitted to auto-detect from server
python3 -m sglang.test.run_eval \
    --base-url "${ENDPOINT}" \
    --eval-name gsm8k \
    --num-examples "${NUM_EXAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --num-threads "${NUM_THREADS}" \
    --num-shots "${NUM_SHOTS}"

# Copy result file
result_file=$(ls -t /tmp/gsm8k_*.json 2>/dev/null | head -n1)
if [ -f "$result_file" ]; then
    cp "$result_file" "$result_dir/"
    echo "Results saved to: $result_dir/$(basename "$result_file")"
else
    echo "Warning: Could not find result file in /tmp"
fi

# Copy HTML report if exists
html_file=$(ls -t /tmp/gsm8k_*.html 2>/dev/null | head -n1)
if [ -f "$html_file" ]; then
    cp "$html_file" "$result_dir/"
fi

echo "GSM8K evaluation complete"

