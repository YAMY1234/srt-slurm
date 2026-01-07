#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup HuggingFace cache directory to use shared storage
# This avoids re-downloading datasets for every benchmark run
setup_hf_cache() {
    local shared_cache="${1:-/configs/hf_cache}"
    
    # Skip if HF_HOME is already set
    if [ -n "$HF_HOME" ]; then
        echo "Using existing HF_HOME: $HF_HOME"
        return 0
    fi
    
    # Try to use shared cache directory
    if [ -d "$shared_cache" ] || mkdir -p "$shared_cache" 2>/dev/null; then
        export HF_HOME="$shared_cache"
        export HF_DATASETS_CACHE="$shared_cache/datasets"
        export HUGGINGFACE_HUB_CACHE="$shared_cache/hub"
        echo "Using shared HuggingFace cache: $HF_HOME"
        
        # Create subdirectories if they don't exist
        mkdir -p "$HF_DATASETS_CACHE" 2>/dev/null || true
        mkdir -p "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || true
    else
        echo "Warning: Could not access shared cache $shared_cache, using default cache"
    fi
    
    # Set longer timeout for HuggingFace requests (default is 10s which can timeout)
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
}

wait_for_model() {

    local model_host=$1
    local model_port=$2
    local n_prefill=${3:-1}
    local n_decode=${4:-1}
    local poll=${5:-1}
    local timeout=${6:-600}
    local report_every=${7:-60}

    local health_addr="http://${model_host}:${model_port}/health"
    local models_addr="http://${model_host}:${model_port}/v1/models"
    echo "Polling ${health_addr} every ${poll} seconds to check whether ${n_prefill} prefills and ${n_decode} decodes are alive"

    local start_ts=$(date +%s)
    local report_ts=$(date +%s)
    
    # Stability check: require consecutive successful checks to ensure service is stable
    local stability_required=6  # Require 6 consecutive successful checks (30 seconds with 5s poll)
    local stability_count=0

    while :; do
        # Curl timeout - our primary use case here is to launch it at the first node (localhost), so no timeout is needed.
        curl_result=$(curl ${health_addr} 2>/dev/null)
        # Python path - Use of `check_server_health.py` is self-constrained outside of any packaging.
        check_result=$(python3 /scripts/utils/check_server_health.py $n_prefill $n_decode <<< $curl_result)
        if [[ $check_result == *"Model is ready."* ]]; then
            # Additional check: verify endpoints are actually available by calling /v1/models
            # Check that the response contains model data (not just HTTP 200 with empty list)
            models_response=$(curl -s ${models_addr} 2>/dev/null)
            models_http_code=$(curl -s -o /dev/null -w "%{http_code}" ${models_addr} 2>/dev/null)
            
            # Check HTTP 200 and response contains "data" with at least one model
            if [[ "$models_http_code" == "200" ]] && [[ "$models_response" == *'"data"'* ]] && [[ "$models_response" == *'"id"'* ]]; then
                stability_count=$((stability_count + 1))
                echo "Service ready check passed ($stability_count/$stability_required)"
                
                if [[ $stability_count -ge $stability_required ]]; then
                    echo $check_result
                    echo "Service stable: passed $stability_required consecutive checks"
                    echo "Models response: $models_response"
                    return 0
                fi
            else
                # Reset stability counter if check fails
                if [[ $stability_count -gt 0 ]]; then
                    echo "Service became unavailable, resetting stability counter (was $stability_count)"
                fi
                stability_count=0
                echo "Instances ready but endpoints not available yet (HTTP $models_http_code), waiting..."
            fi
        else
            # Reset stability counter if health check fails
            if [[ $stability_count -gt 0 ]]; then
                echo "Health check failed, resetting stability counter (was $stability_count)"
            fi
            stability_count=0
        fi

        time_now=$(date +%s)
        if [[ $((time_now - start_ts)) -ge $timeout ]]; then
            echo "Model did not get healthy in ${timeout} seconds"
            exit 2;
        fi

        if [[ $((time_now - report_ts)) -ge $report_every ]]; then
            echo $check_result
            report_ts=$time_now
        fi

        sleep $poll
    done
}