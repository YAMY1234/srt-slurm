#!/bin/bash
# ============================================================
# SGLang Disaggregation Benchmark Commands Summary
# ============================================================
# Usage: Copy and paste the command you want to run
# All commands should be run from: ~/srt-slurm
# ============================================================

cd /home/yangminl/srt-slurm
streamlit run analysis/dashboard/app.py

# export to excel
python analysis/export_to_excel.py

python scripts/plot_batch_metrics.py /lustre/fsw/coreai_comparch_trtllm/yangminl/srt-slurm/outputs/1047098-1p1d-dep4/logs

# Qwen3.5-397B-A17B-FP8 on GB200

# AGG 模式 - 单节点 TP4 聚合推理
# PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/agg.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/agg-65536.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/agg-32768.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/agg-65536-no-parser.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/agg-81920-no-parser.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/agg-acc.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/agg-acc-symmem.yaml

# fp8-agg-sweep-nobuffer
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-agg-sweep-nobuffer/agg-tp4-trtllm.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-agg-sweep-nobuffer/agg-tep4-trtllm.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-agg-sweep-nobuffer/agg-dep4-trtllm.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-agg-sweep-nobuffer/agg-tp4-trtllm-nobs.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-agg-sweep-nobuffer/agg-tp4-trtllm-symmem.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-agg-sweep-nobuffer/agg-tep4-trtllm-symmem.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-agg-sweep-nobuffer/agg-dep4-trtllm-symmem.yaml


# 1P1D 普通模式 - PD分离, TP4, 无EP
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/1p1d.yaml

# 1P1D + MTP 精度验证 (GPQA) - Disagg + EAGLE speculative decoding
# 无 prefix caching (disable-radix-cache + no_buffer)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/1p1d-mtp-acc.yaml
# 有 prefix caching (extra_buffer + radix cache) - 已知低精度问题: GSM8K 0.71/0.61
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/1p1d-mtp-acc-prefixcache.yaml

# 1P1D DeepEP + DeepGemm - 多节点 TP8/EP8 (需要 rebuild-deepep.sh 修改 kNumMaxTopK)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/1p1d-deepep-deepgemm.yaml --setup-script rebuild-deepep.sh

# 1P1D + DEP4 + Acc
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/1p1d-dep-acc.yaml

# overflow
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/repro-int32-overflow.yaml

# FP8 Sweep
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-sweep/agg-tp4-trtllm.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-sweep/agg-tep4-trtllm.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-sweep/agg-dep4-trtllm.yaml

# Debug: KV/Mamba slice transfer validation (GSM8K)
# Debug A: AGG + TP4/DP4/dp_attn (baseline, no disagg) -> 0.970
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/debug/debug-a-agg-dp4.yaml
# Debug B: 1P1D + TP4/DP4/dp_attn, 1+1 nodes -> 0.010 (before fix) -> 0.995 (after fix)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/debug/debug-b-1p1d-dp4.yaml
# Debug C: 1P1D + TP8/DP8/dp_attn, 1+2 nodes -> 0.010 (before fix) -> 0.995 (after fix)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/debug/debug-c-1p1d-dp8-2node.yaml

# FP8 Disagg Sweep - 1P1D, 1k1k sa-bench, concurrency sweep
# 1P1D + TP4 prefill + TP4 decode (pure TP, no EP)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-disagg-sweep/1p1d-tp4.yaml
# 1P1D + TP4 prefill + TEP4 decode (TP4+EP4, no dp-attn)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-disagg-sweep/1p1d-tep4.yaml
# 1P1D + TP4 prefill + DEP4 decode (DP4+TP4+EP4, dp-attn)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-disagg-sweep/1p1d-dep4.yaml
# 1P1D + DEP4 prefill + DEP4 decode (homogeneous, no TP slice transfer overhead)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-disagg-sweep/1p1d-dep4dep4.yaml
# 1P1D + TP4 prefill + TP4 decode + MTP (EAGLE spec dec v2)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-disagg-sweep/1p1d-tp4-mtp.yaml
# 1P1D + TP4 prefill + TP4 decode (pure TP, no EP) - NIXL transfer backend
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/fp8-disagg-sweep/1p1d-tp4-nixl.yaml

# NVFP4
# Repro sglang#19383: TopKTopPSamplingFromProbs illegal memory access (max-running-requests=512)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5-nvfp4/repro-19383-sampling-crash.yaml
# Verify: bypass flashinfer kernel with --sampling-backend pytorch (quick test)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5-nvfp4/repro-19383-pytorch-sampling.yaml
# Verify fix: rebuild sgl_kernel with flashinfer sampling bounds-check fix
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5-nvfp4/repro-19383-with-fix.yaml

# Profile - Qwen3.5 Torch Profiling (prefill step 10-20, decode step 10-20)
# Disagg 1P1D profiles
# 1P1D + TP4 prefill + TP4 decode (pure TP, no EP)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/profile/1p1d-tp4-profile.yaml
# 1P1D + TP4 prefill + TEP4 decode (TP4+EP4, no dp-attn)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/profile/1p1d-tep4-profile.yaml
# 1P1D + TP4 prefill + DEP4 decode (DP4+TP4+EP4, dp-attn)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/profile/1p1d-dep4-profile.yaml
# 1P1D + TP4 prefill + TP4 decode + MTP (EAGLE spec dec)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/profile/1p1d-tp4-mtp-profile.yaml
# Agg profiles
# Aggregated TP4 (pure TP, no EP) + symm-mem
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/profile/agg-tp4-trtllm-symmem-profile.yaml
# Aggregated TEP4 (TP4+EP4, no dp-attn) + symm-mem
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/profile/agg-tep4-trtllm-symmem-profile.yaml
# Aggregated DEP4 (DP4+TP4+EP4, dp-attn) + symm-mem
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/qwen3.5/profile/agg-dep4-trtllm-symmem-profile.yaml

# ============================================================
# 128k8k Accuracy Tests (LongBench-v2)
# ============================================================
# Test configurations for GB300 FP4 128k context accuracy evaluation
# Container: dev-1212, requires --setup-script reinstall-flashinfer.sh

# Test 1: PP4 + DEP4 - 基准配置
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test1-top-of-curve.yaml --setup-script reinstall-flashinfer.sh

# Test 2: PP4 + DEP4 + chunking - chunked-prefill测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test2-mid-curve-pt2-chunking.yaml --setup-script reinstall-flashinfer.sh

# Test 3: PP4 + DEP8 - EP8大并行
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test3-mid-curve-pt3-ep8.yaml --setup-script reinstall-flashinfer.sh

# Test 4: PP4 + 2 x DEP4 - 低延迟多worker
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test4-low-latency.yaml --setup-script reinstall-flashinfer.sh

# Test 5: PP4 + DEP4 + kvcache - KV-Cache Reuse测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test5-top-of-curve-kvcache.yaml --setup-script reinstall-flashinfer.sh

# Test 6: PP4 + DEP4 + MTP - MTP开启测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test6-top-of-curve-mtp.yaml --setup-script reinstall-flashinfer.sh

# Test 7: PP4 + DEP8 + MTP + kvcache - EP8 + MTP + KVCache
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test7-mid-curve-pt3-mtp-kvcache.yaml --setup-script reinstall-flashinfer.sh

# Submit all 7 tests at once:
# for f in recipes/gb300-fp4/128k8k_acc/test*.yaml; do
#   echo "Submitting: $f"
#   PYTHONPATH=src python -m srtctl.cli.submit apply -f "$f" --setup-script reinstall-flashinfer.sh
# done

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k/longbenchv2.yaml --setup-script reinstall-flashinfer.sh




# Test 1: PP4 + DEP4 - 基准配置
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test1-top-of-curve.yaml

# Test 2: PP4 + DEP4 + chunking - chunked-prefill测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test2-mid-curve-pt2-chunking.yaml

# Test 3: PP4 + DEP8 - EP8大并行
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test3-mid-curve-pt3-ep8.yaml

# Test 4: PP4 + 2 x DEP4 - 低延迟多worker
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test4-low-latency.yaml

# Test 5: PP4 + DEP4 + kvcache - KV-Cache Reuse测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test5-top-of-curve-kvcache.yaml

# Test 6: PP4 + DEP4 + MTP - MTP开启测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test6-top-of-curve-mtp.yaml

# Test 7: PP4 + DEP8 + MTP + kvcache - EP8 + MTP + KVCache
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test7-mid-curve-pt3-mtp-kvcache.yaml

# Test 6: PP4 + DEP4 + MTP - MTP开启测试 (V2)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_acc/test6-top-of-curve-mtp_v2.yaml

# SGLang Router Profile
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/sglang_router_profile.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/sglang_router_profile_no_spec_v2.yaml

# 1P1D MTP V2
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1p_1d_mtp_v2.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-fp4/agg_longbenchv2.yaml


PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/symmem_perf/1-top-of-curve-without-sym.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/symmem_perf/1-top-of-curve.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-v1-symm/1p_1d_mtp_v1_nochunk.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-v1-symm/pp_tp_mtp1_longbenchv2-symmem.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-v1-symm/1p_1d_mtp_v1_nochunk-r4.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-v1-symm/1p_1d_nochunk-r4.yaml

# /home/yangminl/srt-slurm/recipes/symmem-debug/1p_1d_mtp_v2_tp4_nochunk.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/symmem-debug/1p_1d_mtp_v2_tp4_nochunk.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/symmem-debug/1p_1d_mtp_v2_tp4_nochunk-nocudagraph.yaml


# /home/yangminl/srt-slurm/recipes/gb300-mtp-perf/pp_tp_base.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-perf/pp_tp_base.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-perf/pp_tp_mtp1.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-perf/pp_tp_mtp2.yaml

# /home/yangminl/srt-slurm/recipes/gb300-mtp-gpqa
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-gpqa/pp_tp_base.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-gpqa/pp_tp_mtp1.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/gb300-mtp-gpqa/pp_tp_mtp2.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipes/mainbranch-1012/pp_tp_mtp2.yaml


PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/1p_1d_mtp_v1_nochunk.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/1p_1d_mtp_v2_8192.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/1p_1d_mtp_v2_nochunk.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/1p_1d_mtp_v2_tp4_8192.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/1p_1d_mtp_v2_tp4_nochunk.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/1p_1d_nochunk.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/agg_longbenchv2-8192.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/agg_longbenchv2-nomtp-nosymmem.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/agg_longbenchv2-nosymmem-mtpv1.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/agg_longbenchv2-nosymmem.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipes/gb300-mtp-verify-withsymmem/agg_longbenchv2.yaml


# ls -la /cm/local/apps/slurm/current/bin/sbatch 2>&1; echo "---"; echo $PATH | head -c 500
# export PATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-mtp.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-mtp-small.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-mtp-no-deepgemm.yaml  --setup-script gb300-fp4-mtp-setup.sh


PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-mtp-acc.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-mtp-gsm8k.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-nomtp-acc.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-nomtp-acc-gsm8k.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-mtp-small-acc.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/max-tpt-mtp-small-nomtp-acc.yaml  --setup-script gb300-fp4-mtp-setup.sh

# /home/yangminl/srt-slurm/recipes/gb300-baizhou-debug folder all:
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhou-debug/max-tpt-mtp-small-nomtp-acc-noeplb.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhou-debug/max-tpt-mtp-small-nomtp-acc-noeplb-nodeepep.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhou-debug/max-tpt-mtp-small-nomtp-acc-noeplb-nodeepep-nocudsl.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhou-debug/max-tpt-mtp-small-nomtp-acc-noeplb-cutlass.yaml  --setup-script gb300-fp4-mtp-setup.sh

# collect expert distribution
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/collect-expert-distribution-8k1k.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/collect-expert-distribution-1k8k.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-baizhoubranch/collect-expert-distribution-1k1k.yaml  --setup-script gb300-fp4-mtp-setup.sh


# 1-top-of-curve
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_cutlass/1-top-of-curve.yaml --setup-script gb300-fp4-mtp-setup.sh

# 2-mid-curve-pt1
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_cutlass/2-mid-curve-pt1.yaml --setup-script gb300-fp4-mtp-setup.sh

# 3-mid-curve-pt2
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_cutlass/3-mid-curve-pt2.yaml --setup-script gb300-fp4-mtp-setup.sh

# 4-mid-curve-pt3
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_cutlass/4-mid-curve-pt3.yaml --setup-script gb300-fp4-mtp-setup.sh

# 5-low-latency
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_cutlass/5-low-latency.yaml --setup-script gb300-fp4-mtp-setup.sh

# GB300 profile:
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k/2-mid-curve-pt1-profile.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k/2-mid-curve-pt1-cutlass-profile.yaml --setup-script gb300-fp4-mtp-setup.sh

# 128k8k cutlass prefill
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k/2-mid-curve-pt1-cutlass-prefill.yaml --setup-script gb300-fp4-mtp-setup.sh


PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen1_tep4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen1_tep8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen7_tep8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen13_tep4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen14_tep2_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx2_ctx_pp4_gen11_tep2_batch4_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen8_tep2_batch2_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx3_ctx_pp4_gen1_dep8_batch32_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx5_ctx_pp4_gen1_dep32_batch4_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx7_ctx_pp4_gen1_dep16_batch32_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx8_ctx_pp4_gen1_dep32_batch16_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx8_ctx_pp4_gen1_dep32_batch8_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

# low latency
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tep4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tep2_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen7_tep8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen7_tp8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

# 1p2d
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen2_tp4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tp8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

# 128k8k MTP 配置
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx3_ctx_pp4_gen1_dep8_batch16_eplb0_mtp1.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx5_ctx_pp4_gen1_dep16_batch8_eplb0_mtp2.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx5_ctx_pp4_gen1_dep32_batch2_eplb256_mtp3.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx6_ctx_pp4_gen1_dep16_batch16_eplb0_mtp1.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx7_ctx_pp4_gen1_dep32_batch4_eplb256_mtp3.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx7_ctx_pp4_gen2_dep16_batch4_eplb0_mtp3.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx8_ctx_pp4_gen1_dep16_batch32_eplb0_mtp1.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx8_ctx_pp4_gen1_dep32_batch16_eplb256_mtp1.yaml --setup-script gb300-fp4-mtp-setup.sh


# 1. ctx8 - middle curve (1288k_ratematch)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx8_ctx_pp4_gen1_dep32_batch8_eplb0_mtp0.yaml --setup-script gb200-fp4-setup.sh

# 2. ctx3 - high throughput (1288k_ratematch)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/1288k_ratematch/ctx3_ctx_pp4_gen1_dep8_batch32_eplb0_mtp0.yaml --setup-script gb200-fp4-setup.sh

# 3. ctx5 - middle curve with MTP (128k8k_mtp)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx5_ctx_pp4_gen1_dep16_batch8_eplb0_mtp2.yaml --setup-script gb200-fp4-setup.sh

# srt-slurm/recipes/gb300-fp4/128k8k_mtp/ctx7_ctx_pp4_gen2_dep16_batch4_eplb0_mtp3.yaml
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx7_ctx_pp4_gen2_dep16_batch4_eplb0_mtp3.yaml --setup-script gb200-fp4-setup.sh

# Extra experiments Feb 5th 2026

# GB200 tune - larger bs (160)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/gb300_blog/ctx8_ctx_pp4_gen1_dep32_batch160_eplb0_mtp0.yaml --setup-script gb200-fp4-setup.sh

# GB200 run with MTP1
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/gb300_blog/ctx8_ctx_pp4_gen1_dep16_batch32_eplb0_mtp1.yaml --setup-script gb200-fp4-setup.sh

# GB200 MTP2 - batch16
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/gb300_blog/ctx8_ctx_pp4_gen1_dep32_batch16_eplb0_mtp2.yaml --setup-script gb200-fp4-setup.sh

# GB200 MTP2 - batch8
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/gb300_blog/ctx8_ctx_pp4_gen1_dep32_batch8_eplb0_mtp2.yaml --setup-script gb200-fp4-setup.sh

# GB200 no MTP - ctx7
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/gb300_blog/ctx7_ctx_pp4_gen1_dep16_batch16_eplb0_mtp0.yaml --setup-script gb200-fp4-setup.sh

# kernel comp
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_kernel_comp/2-mid-curve-pt1-profile.yaml --setup-script gb200-fp4-setup.sh
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_kernel_comp/2-mid-curve-pt1-cutlass-profile.yaml --setup-script gb200-fp4-setup.sh

# kernel comp - chunked prefill (non-cutlass, flashinfer_trtllm)

# --- GB200 chunked ---
# chunk 16k (GB200)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_kernel_comp/2-mid-curve-pt1-profile-chunked-16k.yaml --setup-script gb200-fp4-setup.sh

# chunk 32k (GB200)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_kernel_comp/2-mid-curve-pt1-profile-chunked-32k.yaml --setup-script gb200-fp4-setup.sh

# chunk 64k (GB200)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_kernel_comp/2-mid-curve-pt1-profile-chunked-64k.yaml --setup-script gb200-fp4-setup.sh

# --- GB300 chunked ---
# chunk 16k (GB300)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_kernel_comp/2-mid-curve-pt1-profile-chunked-16k-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 32k (GB300)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_kernel_comp/2-mid-curve-pt1-profile-chunked-32k-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 64k (GB300)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_kernel_comp/2-mid-curve-pt1-profile-chunked-64k-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# MTP verification
python -m srtctl.cli.submit apply -f recipes/mtp_verification/max-tpt-2p1d-mtp.yaml --setup-script gb200-fp4-setup.sh
# /lustre/fsw/coreai_comparch_trtllm/yangminl/srt-slurm/recipes/mtp_verification/ctx8_ctx_pp4_gen1_dep16_batch32_eplb0_mtp1.yaml
python -m srtctl.cli.submit apply -f recipes/mtp_verification/ctx8_ctx_pp4_gen1_dep16_batch32_eplb0_mtp1.yaml --setup-script gb200-fp4-setup.sh

# Prefill experiments
# 1. concurrency1-cutlass
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0_concurrency1-cutlass.yaml --setup-script gb200-fp4-setup.sh

# 2. concurrency1
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0_concurrency1.yaml --setup-script gb200-fp4-setup.sh

# 3. chunked-cutlass
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-cutlass.yaml --setup-script gb200-fp4-setup.sh

# 4. chunked
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked.yaml --setup-script gb200-fp4-setup.sh

# GB300 (128k8k_prefill_300) - 4个文件

# 1. concurrency1-gb300
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0_concurrency1-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# 2. concurrency1-cutlass-gb300
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0_concurrency1-cutlass-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# 3. chunked-gb300
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# 4. chunked-cutlass-gb300
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-cutlass-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh


# MTPGB200:
# 128k8k MTP 配置
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx3_ctx_pp4_gen1_dep8_batch16_eplb0_mtp1.yaml --setup-script gb200-fp4-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx5_ctx_pp4_gen1_dep16_batch8_eplb0_mtp2.yaml --setup-script gb200-fp4-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx5_ctx_pp4_gen1_dep32_batch2_eplb256_mtp3.yaml --setup-script gb200-fp4-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx6_ctx_pp4_gen1_dep16_batch16_eplb0_mtp1.yaml --setup-script gb200-fp4-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx7_ctx_pp4_gen1_dep32_batch4_eplb256_mtp3.yaml --setup-script gb200-fp4-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx7_ctx_pp4_gen2_dep16_batch4_eplb0_mtp3.yaml --setup-script gb200-fp4-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx8_ctx_pp4_gen1_dep16_batch32_eplb0_mtp1.yaml --setup-script gb200-fp4-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_mtp/ctx8_ctx_pp4_gen1_dep32_batch16_eplb256_mtp1.yaml --setup-script gb200-fp4-setup.sh

# ============================================================
# Prefill Chunking Comparison (Static & Dynamic Chunking)
# ============================================================
# Only non-cutlass (flashinfer_trtllm) versions
# Comparing chunk sizes: 8k, 16k, 32k(existing), 64k
# With and without dynamic chunking

# --- GB200 (128k8k_prefill) - Static Chunking ---

# chunk 2k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-2k.yaml --setup-script gb200-fp4-setup.sh

# chunk 4k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-4k.yaml --setup-script gb200-fp4-setup.sh

# chunk 8k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-8k.yaml --setup-script gb200-fp4-setup.sh

# chunk 16k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-16k.yaml --setup-script gb200-fp4-setup.sh

# chunk 32k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked.yaml --setup-script gb200-fp4-setup.sh

# chunk 64k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-64k.yaml --setup-script gb200-fp4-setup.sh

# --- GB200 (128k8k_prefill) - Dynamic Chunking ---

# chunk 8k (dynamic)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-8k-dynchunk.yaml --setup-script gb200-fp4-setup.sh

# chunk 16k (dynamic)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-16k-dynchunk.yaml --setup-script gb200-fp4-setup.sh

# chunk 32k (dynamic)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-32k-dynchunk.yaml --setup-script gb200-fp4-setup.sh

# chunk 64k (dynamic)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-64k-dynchunk.yaml --setup-script gb200-fp4-setup.sh

# --- GB300 (128k8k_prefill_300) - Static Chunking ---

# chunk 2k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-2k-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 4k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-4k-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 8k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-8k-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 16k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-16k-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 32k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 64k (static)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-64k-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# --- GB300 (128k8k_prefill_300) - Dynamic Chunking ---

# chunk 8k (dynamic)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-8k-dynchunk-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 16k (dynamic)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-16k-dynchunk-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 32k (dynamic)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-32k-dynchunk-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh

# chunk 64k (dynamic)
python -m srtctl.cli.submit apply -f recipes/gb300-fp4/128k8k_prefill_300/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked-64k-dynchunk-gb300.yaml --setup-script gb300-fp4-mtp-setup.sh
