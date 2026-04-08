#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GSM8K benchmark evaluation (original bench_sglang.py style)
# Uses /v1/completions by default (no chat template), or /v1/chat/completions with --use-chat-api
# Expects: endpoint [num_questions] [num_shots] [max_new_tokens] [parallel] [temperature] [top_p] [use_chat_api] [platinum]

set -e

ENDPOINT=$1
NUM_QUESTIONS=${2:-1319}
NUM_SHOTS=${3:-5}
MAX_NEW_TOKENS=${4:-512}
PARALLEL=${5:-64}
TEMPERATURE=${6:-0.0}
TOP_P=${7:-1.0}
USE_CHAT_API=${8:-}
PLATINUM=${9:-}

# Auto-detect model name from /v1/models endpoint
MODEL_NAME=$(curl -s "${ENDPOINT}/v1/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
if [ -z "${MODEL_NAME}" ]; then
    MODEL_NAME="default"
    echo "Warning: Could not auto-detect model name, using default: ${MODEL_NAME}"
fi

echo "GSM8K-Bench Config: endpoint=${ENDPOINT}; model=${MODEL_NAME}; num_questions=${NUM_QUESTIONS}; num_shots=${NUM_SHOTS}; max_new_tokens=${MAX_NEW_TOKENS}; parallel=${PARALLEL}; temperature=${TEMPERATURE}; top_p=${TOP_P}; use_chat_api=${USE_CHAT_API:-false}; platinum=${PLATINUM:-false}"

# Build command
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cmd="python3 ${SCRIPT_DIR}/bench_gsm8k.py \
    --endpoint ${ENDPOINT} \
    --model ${MODEL_NAME} \
    --num-questions ${NUM_QUESTIONS} \
    --num-shots ${NUM_SHOTS} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --parallel ${PARALLEL} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P}"

if [ "${USE_CHAT_API}" = "true" ]; then
    cmd="${cmd} --use-chat-api"
fi

if [ "${PLATINUM}" = "true" ]; then
    cmd="${cmd} --platinum"
fi

echo "Executing: ${cmd}"
eval "${cmd}"

echo "GSM8K-Bench evaluation complete"
