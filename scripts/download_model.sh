#!/bin/bash
# ============================================================================
# HuggingFace model download script - runs via SLURM compute node
# ============================================================================
# Usage: ./download_model.sh <model_id> [output_dir]
#
# Examples:
#   ./download_model.sh Qwen/Qwen3.5-397B-A17B-FP8
#   ./download_model.sh Qwen/Qwen3.5-397B-A17B-FP8 /custom/path
#   ./download_model.sh deepseek-ai/DeepSeek-R1
#
# Output: /lustre/fsw/coreai_comparch_trtllm/yangminl/models/<model_name>/
# ============================================================================

set -e

# Configuration
ACCOUNT="coreai_comparch_trtllm"
PARTITION="gb200-backfill"
TIME_LIMIT="08:00:00"
DEFAULT_MODEL_DIR="/lustre/fsw/coreai_comparch_trtllm/yangminl/models"
LOG_DIR="/lustre/fsw/coreai_comparch_trtllm/yangminl/logs"
CONTAINER="/lustre/fsw/coreai_comparch_trtllm/yangminl/containers/lmsysorg+sglang+dev.sqsh"

# Check arguments
if [ -z "$1" ]; then
    echo "Error: Please provide a HuggingFace model ID"
    echo ""
    echo "Usage: $0 <model_id> [output_dir]"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen3.5-397B-A17B-FP8"
    echo "  $0 deepseek-ai/DeepSeek-R1"
    echo "  $0 Qwen/Qwen3.5-397B-A17B-FP8 /custom/output/path"
    exit 1
fi

MODEL_ID="$1"

# Extract model name from model_id: Qwen/Qwen3.5-397B-A17B-FP8 -> Qwen3.5-397B-A17B-FP8
MODEL_NAME=$(basename "$MODEL_ID")

# Output directory
if [ -n "$2" ]; then
    OUTPUT_DIR="$2"
else
    OUTPUT_DIR="${DEFAULT_MODEL_DIR}/${MODEL_NAME}"
fi

# Create directories
mkdir -p "$DEFAULT_MODEL_DIR" "$LOG_DIR"

echo "============================================"
echo "HuggingFace Model Download Tool"
echo "============================================"
echo "Model:   $MODEL_ID"
echo "Output:   $OUTPUT_DIR"
echo "Container:   $CONTAINER"
echo "Partition:   $PARTITION"
echo "Time limit:   $TIME_LIMIT"
echo "============================================"

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "Container file not found: $CONTAINER"
    echo "   Please run scripts/download_container.sh first to download the container"
    exit 1
fi

# Check if directory already exists and is non-empty
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "Warning: Directory already exists and is non-empty: $OUTPUT_DIR"
    read -p "Continue (will resume download)? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
fi

# Create temporary SLURM script
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
echo "Starting HuggingFace model download"
echo "============================================"
echo "Model:   MODEL_ID_PH"
echo "Output:   OUTPUT_DIR_PH"
echo "Container:   CONTAINER_PH"
echo "Node:   $(hostname)"
echo "Time:   $(date)"
echo "============================================"

mkdir -p "OUTPUT_DIR_PH"

echo ""
echo "Downloading model via container..."
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

# Check result
if [ $? -eq 0 ] && [ -d "OUTPUT_DIR_PH" ] && [ "$(ls -A "OUTPUT_DIR_PH")" ]; then
    echo ""
    echo "============================================"
    echo "Download succeeded!"
    echo "============================================"
    echo "Model files:"
    ls -lh "OUTPUT_DIR_PH/" | head -20
    echo ""
    du -sh "OUTPUT_DIR_PH"
    echo ""
    echo "Add to srtslurm.yaml model_paths:"
    echo ""
    echo "model_paths:"
    echo "  MODEL_NAME_PH: \"OUTPUT_DIR_PH\""
    echo ""
else
    echo ""
    echo "============================================"
    echo "Download failed!"
    echo "============================================"
    exit 1
fi

echo "Completed at: $(date)"
OUTER_EOF

# Replace placeholders
sed -i "s|CONTAINER_PH|${CONTAINER}|g" "$TEMP_SCRIPT"
sed -i "s|ACCOUNT_PH|${ACCOUNT}|g" "$TEMP_SCRIPT"
sed -i "s|PARTITION_PH|${PARTITION}|g" "$TEMP_SCRIPT"
sed -i "s|TIME_LIMIT_PH|${TIME_LIMIT}|g" "$TEMP_SCRIPT"
sed -i "s|LOG_DIR_PH|${LOG_DIR}|g" "$TEMP_SCRIPT"
sed -i "s|MODEL_ID_PH|${MODEL_ID}|g" "$TEMP_SCRIPT"
sed -i "s|MODEL_NAME_PH|${MODEL_NAME}|g" "$TEMP_SCRIPT"
sed -i "s|OUTPUT_DIR_PH|${OUTPUT_DIR}|g" "$TEMP_SCRIPT"

# Submit job
echo ""
echo "Submitting SLURM job..."
JOB_ID=$(sbatch --parsable "$TEMP_SCRIPT")

# Clean up temporary files
rm -f "$TEMP_SCRIPT"

echo ""
echo "============================================"
echo "Job submitted!"
echo "============================================"
echo "Job ID: $JOB_ID"
echo ""
echo "View status:"
echo "  squeue -j $JOB_ID"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_DIR}/hf_download_${JOB_ID}.log"
echo ""
echo "Check files after completion:"
echo "  ls -lh ${OUTPUT_DIR}/"
echo "  du -sh ${OUTPUT_DIR}"
echo ""
echo "Update srtslurm.yaml after completion:"
echo "  model_paths:"
echo "    ${MODEL_NAME}: \"${OUTPUT_DIR}\""
echo "============================================"
