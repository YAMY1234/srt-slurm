参考一下 /home/yangminl/srt-slurm/recipies/gb300-fp4/1288k_ratematch，我需要创建一个类似的一组yaml到本目录
具体的配置需要参考：
name	max global bs	concurrency	tps/user	tps/gpu	AR	MTP
ctx1_ctx_pp4_gen1_tep8_batch1_eplb0_mtp3	1	2	257.18	21.43	2.99	3
ctx1_ctx_pp4_gen1_tep8_batch2_eplb0_mtp3	2	4	221.955	36.99	3.06	3
ctx1_ctx_pp4_gen6_tep8_batch1_eplb0_mtp3	1	12	229.7683	26.51	3.17	3
ctx1_ctx_pp4_gen4_tep8_batch2_eplb0_mtp3	2	16	193.5825	43.02	3.11	3
ctx1_ctx_pp4_gen8_tep2_batch1_eplb0_mtp3	1	16	187.2225	74.89	3.04	3
ctx1_ctx_pp4_gen5_tep2_batch2_eplb0_mtp3	2	20	148.688	106.21	3.05	3
ctx5_ctx_pp4_gen1_dep32_batch2_eplb256_mtp3	64	128	121.7397	149.83	2.98	3
ctx7_ctx_pp4_gen1_dep32_batch4_eplb256_mtp3	128	256	86.5141	184.56	2.95	3
ctx7_ctx_pp4_gen2_dep16_batch4_eplb0_mtp3	64	256	86.5548	184.65	2.93	3
ctx5_ctx_pp4_gen1_dep16_batch8_eplb0_mtp2	128	256	61.3834	218.25	2.6	2
ctx8_ctx_pp4_gen1_dep32_batch16_eplb256_mtp1	512	1024	25.7158	205.73	1.97	1
ctx6_ctx_pp4_gen1_dep16_batch16_eplb0_mtp1	256	512	37.4528	239.7	1.97	1
ctx3_ctx_pp4_gen1_dep8_batch16_eplb0_mtp1	128	256	37.7598	241.66	1.97	1
ctx8_ctx_pp4_gen1_dep16_batch32_eplb0_mtp1	512	1024	25.0067	266.74	1.97	1

类似的，ctxN表示N个prefill worker，genN表示N个decode worker
dep32表示deepep + cutedsl + tp32 ep32 dp32（这个具体参数可以参考1288k_ratematch下面已有的）
global bs用于决定decode的batch size（对于gen1 depXX而言，decode的max running requests和global bs相同）

MTP是重点：
mtp的额外配置需要参考 /home/yangminl/srt-slurm/recipies/gb200-fp4/1k1k/max-tpt-mtp.yaml 里面的配置（只需要考虑decode不需要考虑prefill）

MTP需要额外添加的参数包括：
      # MTP
      speculative-algorithm: "EAGLE"
      speculative-num-steps: 2
      speculative-eagle-topk: 1
      speculative-num-draft-tokens: 3
      speculative-moe-runner-backend: "deep_gemm"
      speculative-moe-a2a-backend: "deepep"

      enable-single-batch-overlap: true

以及环境变量里面的：
    SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH: "1" # Used in older sglang version （我估计已经有了）
    SGLANG_ENABLE_SPEC_V2: "1"
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH: "1"
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO: "1"

我们目前只需要考虑DEP的情况，因此所有TEP的配置你都可以先不创建，我们先创建所有DEP的即可