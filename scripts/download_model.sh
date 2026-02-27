#!/bin/bash
# ============================================================================
# HuggingFace 模型下载脚本 - 通过 SLURM 计算节点运行
# ============================================================================
# 用法: ./download_model.sh <model_id> [output_dir]
#
# 示例:
#   ./download_model.sh Qwen/Qwen3.5-397B-A17B-FP8
#   ./download_model.sh Qwen/Qwen3.5-397B-A17B-FP8 /custom/path
#   ./download_model.sh deepseek-ai/DeepSeek-R1
#
# 输出: /lustre/fsw/coreai_comparch_trtllm/yangminl/models/<model_name>/
# ============================================================================

set -e

# 配置
ACCOUNT="coreai_comparch_trtllm"
PARTITION="gb200-backfill"
TIME_LIMIT="08:00:00"
DEFAULT_MODEL_DIR="/lustre/fsw/coreai_comparch_trtllm/yangminl/models"
LOG_DIR="/lustre/fsw/coreai_comparch_trtllm/yangminl/logs"
CONTAINER="/lustre/fsw/coreai_comparch_trtllm/yangminl/containers/lmsysorg+sglang+dev.sqsh"

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 错误: 请提供 HuggingFace 模型 ID"
    echo ""
    echo "用法: $0 <model_id> [output_dir]"
    echo ""
    echo "示例:"
    echo "  $0 Qwen/Qwen3.5-397B-A17B-FP8"
    echo "  $0 deepseek-ai/DeepSeek-R1"
    echo "  $0 Qwen/Qwen3.5-397B-A17B-FP8 /custom/output/path"
    exit 1
fi

MODEL_ID="$1"

# 从 model_id 提取模型名: Qwen/Qwen3.5-397B-A17B-FP8 -> Qwen3.5-397B-A17B-FP8
MODEL_NAME=$(basename "$MODEL_ID")

# 输出目录
if [ -n "$2" ]; then
    OUTPUT_DIR="$2"
else
    OUTPUT_DIR="${DEFAULT_MODEL_DIR}/${MODEL_NAME}"
fi

# 创建目录
mkdir -p "$DEFAULT_MODEL_DIR" "$LOG_DIR"

echo "============================================"
echo "🤗 HuggingFace 模型下载工具"
echo "============================================"
echo "模型:   $MODEL_ID"
echo "输出:   $OUTPUT_DIR"
echo "容器:   $CONTAINER"
echo "分区:   $PARTITION"
echo "时限:   $TIME_LIMIT"
echo "============================================"

# 检查容器是否存在
if [ ! -f "$CONTAINER" ]; then
    echo "❌ 容器文件不存在: $CONTAINER"
    echo "   请先运行 scripts/download_container.sh 下载容器"
    exit 1
fi

# 检查目录是否已存在且非空
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "⚠️  警告: 目录已存在且非空: $OUTPUT_DIR"
    read -p "是否继续 (会断点续传)? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

# 创建临时 SLURM 脚本
TEMP_SCRIPT=$(mktemp /tmp/hf_download_XXXXXX.sh)

cat > "$TEMP_SCRIPT" << 'OUTER_EOF'
#!/bin/bash
#SBATCH --job-name=ACCOUNT_PH-hf-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=TIME_LIMIT_PH
#SBATCH --account=ACCOUNT_PH
#SBATCH --partition=PARTITION_PH
#SBATCH --output=LOG_DIR_PH/hf_download_%j.log

echo "============================================"
echo "🤗 开始下载 HuggingFace 模型"
echo "============================================"
echo "模型:   MODEL_ID_PH"
echo "输出:   OUTPUT_DIR_PH"
echo "容器:   CONTAINER_PH"
echo "节点:   $(hostname)"
echo "时间:   $(date)"
echo "============================================"

mkdir -p "OUTPUT_DIR_PH"

echo ""
echo "📥 正在通过容器下载模型..."
srun --nodes=1 --ntasks=1 \
    --container-image="CONTAINER_PH" \
    --container-mounts="OUTPUT_DIR_PH:OUTPUT_DIR_PH" \
    --no-container-mount-home \
    bash -c "
        echo 'Container Python: '\$(python3 --version)
        echo 'huggingface-cli: '\$(which huggingface-cli)
        huggingface-cli download \
            'MODEL_ID_PH' \
            --local-dir 'OUTPUT_DIR_PH' \
            --local-dir-use-symlinks False
    "

# 检查结果
if [ $? -eq 0 ] && [ -d "OUTPUT_DIR_PH" ] && [ "$(ls -A "OUTPUT_DIR_PH")" ]; then
    echo ""
    echo "============================================"
    echo "✅ 下载成功！"
    echo "============================================"
    echo "模型文件:"
    ls -lh "OUTPUT_DIR_PH/" | head -20
    echo ""
    du -sh "OUTPUT_DIR_PH"
    echo ""
    echo "📝 在 srtslurm.yaml model_paths 中添加:"
    echo ""
    echo "model_paths:"
    echo "  MODEL_NAME_PH: \"OUTPUT_DIR_PH\""
    echo ""
else
    echo ""
    echo "============================================"
    echo "❌ 下载失败！"
    echo "============================================"
    exit 1
fi

echo "完成时间: $(date)"
OUTER_EOF

# 替换占位符
sed -i "s|CONTAINER_PH|${CONTAINER}|g" "$TEMP_SCRIPT"
sed -i "s|ACCOUNT_PH|${ACCOUNT}|g" "$TEMP_SCRIPT"
sed -i "s|PARTITION_PH|${PARTITION}|g" "$TEMP_SCRIPT"
sed -i "s|TIME_LIMIT_PH|${TIME_LIMIT}|g" "$TEMP_SCRIPT"
sed -i "s|LOG_DIR_PH|${LOG_DIR}|g" "$TEMP_SCRIPT"
sed -i "s|MODEL_ID_PH|${MODEL_ID}|g" "$TEMP_SCRIPT"
sed -i "s|MODEL_NAME_PH|${MODEL_NAME}|g" "$TEMP_SCRIPT"
sed -i "s|OUTPUT_DIR_PH|${OUTPUT_DIR}|g" "$TEMP_SCRIPT"

# 提交作业
echo ""
echo "📤 提交 SLURM 作业..."
JOB_ID=$(sbatch --parsable "$TEMP_SCRIPT")

# 清理临时文件
rm -f "$TEMP_SCRIPT"

echo ""
echo "============================================"
echo "✅ 作业已提交！"
echo "============================================"
echo "Job ID: $JOB_ID"
echo ""
echo "📊 查看状态:"
echo "  squeue -j $JOB_ID"
echo ""
echo "📋 查看日志:"
echo "  tail -f ${LOG_DIR}/hf_download_${JOB_ID}.log"
echo ""
echo "📁 完成后检查文件:"
echo "  ls -lh ${OUTPUT_DIR}/"
echo "  du -sh ${OUTPUT_DIR}"
echo ""
echo "📝 完成后更新 srtslurm.yaml:"
echo "  model_paths:"
echo "    ${MODEL_NAME}: \"${OUTPUT_DIR}\""
echo "============================================"
