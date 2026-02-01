#!/bin/bash
# ============================================================
# SGLang Disaggregation Benchmark Commands Summary
# ============================================================
# Usage: Copy and paste the command you want to run
# All commands should be run from: ~/srt-slurm
# ============================================================

cd /home/yangminl/srt-slurm
streamlit run analysis/dashboard/app.py

# ============================================================
# 128k8k Accuracy Tests (LongBench-v2)
# ============================================================
# Test configurations for GB300 FP4 128k context accuracy evaluation
# Container: dev-1212, requires --setup-script reinstall-flashinfer.sh

# Test 1: PP4 + DEP4 - 基准配置
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test1-top-of-curve.yaml --setup-script reinstall-flashinfer.sh

# Test 2: PP4 + DEP4 + chunking - chunked-prefill测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test2-mid-curve-pt2-chunking.yaml --setup-script reinstall-flashinfer.sh

# Test 3: PP4 + DEP8 - EP8大并行
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test3-mid-curve-pt3-ep8.yaml --setup-script reinstall-flashinfer.sh

# Test 4: PP4 + 2 x DEP4 - 低延迟多worker
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test4-low-latency.yaml --setup-script reinstall-flashinfer.sh

# Test 5: PP4 + DEP4 + kvcache - KV-Cache Reuse测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test5-top-of-curve-kvcache.yaml --setup-script reinstall-flashinfer.sh

# Test 6: PP4 + DEP4 + MTP - MTP开启测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test6-top-of-curve-mtp.yaml --setup-script reinstall-flashinfer.sh

# Test 7: PP4 + DEP8 + MTP + kvcache - EP8 + MTP + KVCache
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test7-mid-curve-pt3-mtp-kvcache.yaml --setup-script reinstall-flashinfer.sh

# Submit all 7 tests at once:
# for f in recipies/gb300-fp4/128k8k_acc/test*.yaml; do
#   echo "Submitting: $f"
#   PYTHONPATH=src python -m srtctl.cli.submit apply -f "$f" --setup-script reinstall-flashinfer.sh
# done

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k/longbenchv2.yaml --setup-script reinstall-flashinfer.sh




# Test 1: PP4 + DEP4 - 基准配置
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test1-top-of-curve.yaml

# Test 2: PP4 + DEP4 + chunking - chunked-prefill测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test2-mid-curve-pt2-chunking.yaml

# Test 3: PP4 + DEP8 - EP8大并行
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test3-mid-curve-pt3-ep8.yaml

# Test 4: PP4 + 2 x DEP4 - 低延迟多worker
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test4-low-latency.yaml

# Test 5: PP4 + DEP4 + kvcache - KV-Cache Reuse测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test5-top-of-curve-kvcache.yaml

# Test 6: PP4 + DEP4 + MTP - MTP开启测试
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test6-top-of-curve-mtp.yaml

# Test 7: PP4 + DEP8 + MTP + kvcache - EP8 + MTP + KVCache
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test7-mid-curve-pt3-mtp-kvcache.yaml

# Test 6: PP4 + DEP4 + MTP - MTP开启测试 (V2)
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_acc/test6-top-of-curve-mtp_v2.yaml

# SGLang Router Profile
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/sglang_router_profile.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/sglang_router_profile_no_spec_v2.yaml

# 1P1D MTP V2
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1p_1d_mtp_v2.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-fp4/agg_longbenchv2.yaml


PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/symmem_perf/1-top-of-curve-without-sym.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/symmem_perf/1-top-of-curve.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-v1-symm/1p_1d_mtp_v1_nochunk.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-v1-symm/pp_tp_mtp1_longbenchv2-symmem.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-v1-symm/1p_1d_mtp_v1_nochunk-r4.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-v1-symm/1p_1d_nochunk-r4.yaml

# /home/yangminl/srt-slurm/recipies/symmem-debug/1p_1d_mtp_v2_tp4_nochunk.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/symmem-debug/1p_1d_mtp_v2_tp4_nochunk.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/symmem-debug/1p_1d_mtp_v2_tp4_nochunk-nocudagraph.yaml


# /home/yangminl/srt-slurm/recipies/gb300-mtp-perf/pp_tp_base.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-perf/pp_tp_base.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-perf/pp_tp_mtp1.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-perf/pp_tp_mtp2.yaml

# /home/yangminl/srt-slurm/recipies/gb300-mtp-gpqa
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-gpqa/pp_tp_base.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-gpqa/pp_tp_mtp1.yaml
PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/gb300-mtp-gpqa/pp_tp_mtp2.yaml

PYTHONPATH=src python -m srtctl.cli.submit apply -f recipies/mainbranch-1012/pp_tp_mtp2.yaml


PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/1p_1d_mtp_v1_nochunk.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/1p_1d_mtp_v2_8192.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/1p_1d_mtp_v2_nochunk.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/1p_1d_mtp_v2_tp4_8192.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/1p_1d_mtp_v2_tp4_nochunk.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/1p_1d_nochunk.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/agg_longbenchv2-8192.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/agg_longbenchv2-nomtp-nosymmem.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/agg_longbenchv2-nosymmem-mtpv1.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/agg_longbenchv2-nosymmem.yaml

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f /home/yangminl/srt-slurm/recipies/gb300-mtp-verify-withsymmem/agg_longbenchv2.yaml


# ls -la /cm/local/apps/slurm/current/bin/sbatch 2>&1; echo "---"; echo $PATH | head -c 500
# export PATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-mtp.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-mtp-small.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-mtp-no-deepgemm.yaml  --setup-script gb300-fp4-mtp-setup.sh


PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-mtp-acc.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-mtp-gsm8k.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-nomtp-acc.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-nomtp-acc-gsm8k.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-mtp-small-acc.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/max-tpt-mtp-small-nomtp-acc.yaml  --setup-script gb300-fp4-mtp-setup.sh

# /home/yangminl/srt-slurm/recipies/gb300-baizhou-debug folder all:
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhou-debug/max-tpt-mtp-small-nomtp-acc-noeplb.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhou-debug/max-tpt-mtp-small-nomtp-acc-noeplb-nodeepep.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhou-debug/max-tpt-mtp-small-nomtp-acc-noeplb-nodeepep-nocudsl.yaml  --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhou-debug/max-tpt-mtp-small-nomtp-acc-noeplb-cutlass.yaml  --setup-script gb300-fp4-mtp-setup.sh

# collect expert distribution
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/collect-expert-distribution-8k1k.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/collect-expert-distribution-1k8k.yaml  --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-baizhoubranch/collect-expert-distribution-1k1k.yaml  --setup-script gb300-fp4-mtp-setup.sh


# 1-top-of-curve
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_cutlass/1-top-of-curve.yaml --setup-script gb300-fp4-mtp-setup.sh

# 2-mid-curve-pt1
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_cutlass/2-mid-curve-pt1.yaml --setup-script gb300-fp4-mtp-setup.sh

# 3-mid-curve-pt2
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_cutlass/3-mid-curve-pt2.yaml --setup-script gb300-fp4-mtp-setup.sh

# 4-mid-curve-pt3
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_cutlass/4-mid-curve-pt3.yaml --setup-script gb300-fp4-mtp-setup.sh

# 5-low-latency
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_cutlass/5-low-latency.yaml --setup-script gb300-fp4-mtp-setup.sh

# GB300 profile:
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k/2-mid-curve-pt1-profile.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k/2-mid-curve-pt1-cutlass-profile.yaml --setup-script gb300-fp4-mtp-setup.sh

# 128k8k cutlass prefill
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k/2-mid-curve-pt1-cutlass-prefill.yaml --setup-script gb300-fp4-mtp-setup.sh


PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen1_tep4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen1_tep8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen7_tep8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen13_tep4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen14_tep2_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx2_ctx_pp4_gen11_tep2_batch4_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx1_ctx_pp4_gen8_tep2_batch2_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx3_ctx_pp4_gen1_dep8_batch32_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx5_ctx_pp4_gen1_dep32_batch4_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx7_ctx_pp4_gen1_dep16_batch32_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx8_ctx_pp4_gen1_dep32_batch16_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/ctx8_ctx_pp4_gen1_dep32_batch8_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

# low latency
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tep4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tp4_batch1_eplb0_mtp0-chunked.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tep2_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen7_tep8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen7_tp8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

# 1p2d
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen2_tp4_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/1288k_ratematch/low_latency/ctx1_ctx_pp4_gen1_tp8_batch1_eplb0_mtp0.yaml --setup-script gb300-fp4-mtp-setup.sh

# 128k8k MTP 配置
PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_mtp/ctx3_ctx_pp4_gen1_dep8_batch16_eplb0_mtp1.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_mtp/ctx5_ctx_pp4_gen1_dep16_batch8_eplb0_mtp2.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_mtp/ctx5_ctx_pp4_gen1_dep32_batch2_eplb256_mtp3.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_mtp/ctx6_ctx_pp4_gen1_dep16_batch16_eplb0_mtp1.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_mtp/ctx7_ctx_pp4_gen1_dep32_batch4_eplb256_mtp3.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_mtp/ctx7_ctx_pp4_gen2_dep16_batch4_eplb0_mtp3.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_mtp/ctx8_ctx_pp4_gen1_dep16_batch32_eplb0_mtp1.yaml --setup-script gb300-fp4-mtp-setup.sh

PYTHONPATH=/cm/local/apps/slurm/current/bin:/cm/local/apps/slurm/current/sbin:$PATH python -m srtctl.cli.submit apply -f recipies/gb300-fp4/128k8k_mtp/ctx8_ctx_pp4_gen1_dep32_batch16_eplb256_mtp1.yaml --setup-script gb300-fp4-mtp-setup.sh