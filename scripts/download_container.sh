#!/bin/bash
# ============================================================================
# Docker 镜像下载脚本 - 通过 SLURM 计算节点运行 enroot import
# ============================================================================
# 用法: ./download_container.sh <docker_image> [output_name]
#
# 示例:
#   ./download_container.sh lmsysorg/sglang:dev
#   ./download_container.sh lmsysorg/sglang:v0.5.7
#   ./download_container.sh lmsysorg/sglang:v0.5.7 sglang-v0.5.7
#
# 输出: /lustre/fsw/coreai_comparch_trtllm/yangminl/containers/<name>.sqsh
# ============================================================================

set -e

# 配置
ACCOUNT="coreai_comparch_trtllm"
PARTITION="gb200-backfill"  # 可改成 gb300-backfill
TIME_LIMIT="02:00:00"
OUTPUT_DIR="/lustre/fsw/coreai_comparch_trtllm/yangminl/containers"
LOG_DIR="/lustre/fsw/coreai_comparch_trtllm/yangminl/logs"

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 错误: 请提供 Docker 镜像名称"
    echo ""
    echo "用法: $0 <docker_image> [output_name]"
    echo ""
    echo "示例:"
    echo "  $0 lmsysorg/sglang:dev"
    echo "  $0 lmsysorg/sglang:v0.5.7"
    echo "  $0 lmsysorg/sglang:v0.5.7 sglang-v0.5.7"
    exit 1
fi

DOCKER_IMAGE="$1"

# 生成输出文件名
if [ -n "$2" ]; then
    OUTPUT_NAME="$2"
else
    # 自动从 docker 镜像名生成: lmsysorg/sglang:v0.5.7 -> lmsysorg+sglang+v0.5.7
    OUTPUT_NAME=$(echo "$DOCKER_IMAGE" | tr '/:' '+')
fi

OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.sqsh"

# 创建输出目录
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================"
echo "🐳 Docker 镜像下载工具"
echo "============================================"
echo "镜像: $DOCKER_IMAGE"
echo "输出: $OUTPUT_FILE"
echo "分区: $PARTITION"
echo "============================================"

# 检查文件是否已存在
if [ -f "$OUTPUT_FILE" ]; then
    echo "⚠️  警告: 文件已存在: $OUTPUT_FILE"
    read -p "是否覆盖? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

# 创建临时 SLURM 脚本
TEMP_SCRIPT=$(mktemp /tmp/enroot_import_XXXXXX.sh)

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${ACCOUNT}-dev.enroot-import
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/enroot_import_%j.log

echo "============================================"
echo "🐳 开始导入 Docker 镜像"
echo "============================================"
echo "镜像: ${DOCKER_IMAGE}"
echo "输出: ${OUTPUT_FILE}"
echo "节点: \$(hostname)"
echo "时间: \$(date)"
echo "============================================"

# 删除旧文件（如果存在）
rm -f "${OUTPUT_FILE}"

# 运行 enroot import
echo ""
echo "📥 正在下载并转换镜像..."
enroot import -o "${OUTPUT_FILE}" docker://${DOCKER_IMAGE}

# 检查结果
if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "============================================"
    echo "✅ 导入成功！"
    echo "============================================"
    ls -lh "${OUTPUT_FILE}"
    echo ""
    echo "📝 在 srtslurm.yaml 中添加:"
    echo ""
    echo "containers:"
    echo "  ${OUTPUT_NAME}: \"${OUTPUT_FILE}\""
    echo ""
else
    echo ""
    echo "============================================"
    echo "❌ 导入失败！"
    echo "============================================"
    exit 1
fi

echo "完成时间: \$(date)"
EOF

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
echo "  tail -f ${LOG_DIR}/enroot_import_${JOB_ID}.log"
echo ""
echo "📁 完成后检查文件:"
echo "  ls -lh ${OUTPUT_FILE}"
echo "============================================"
