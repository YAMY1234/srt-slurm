#!/bin/bash
# ============================================================================
# Docker image download script - runs enroot import via SLURM compute node
# ============================================================================
# Usage: ./download_container.sh <docker_image> [output_name]
#
# Examples:
#   ./download_container.sh lmsysorg/sglang:dev
#   ./download_container.sh lmsysorg/sglang:v0.5.7
#   ./download_container.sh lmsysorg/sglang:v0.5.7 sglang-v0.5.7
#
# Output: /lustre/fsw/coreai_comparch_trtllm/yangminl/containers/<name>.sqsh
# ============================================================================

set -e

# Configuration
ACCOUNT="coreai_comparch_trtllm"
PARTITION="gb200-backfill"  # can be changed to gb300-backfill
TIME_LIMIT="02:00:00"
OUTPUT_DIR="/lustre/fsw/coreai_comparch_trtllm/yangminl/containers"
LOG_DIR="/lustre/fsw/coreai_comparch_trtllm/yangminl/logs"

# Check arguments
if [ -z "$1" ]; then
    echo "Error: Please provide a Docker image name"
    echo ""
    echo "Usage: $0 <docker_image> [output_name]"
    echo ""
    echo "Examples:"
    echo "  $0 lmsysorg/sglang:dev"
    echo "  $0 lmsysorg/sglang:v0.5.7"
    echo "  $0 lmsysorg/sglang:v0.5.7 sglang-v0.5.7"
    exit 1
fi

DOCKER_IMAGE="$1"

# Generate output file name
if [ -n "$2" ]; then
    OUTPUT_NAME="$2"
else
    # Auto-generate from docker image name: lmsysorg/sglang:v0.5.7 -> lmsysorg+sglang+v0.5.7
    OUTPUT_NAME=$(echo "$DOCKER_IMAGE" | tr '/:' '+')
fi

OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.sqsh"

# Create output directory
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================"
echo "Docker Image Download Tool"
echo "============================================"
echo "Image: $DOCKER_IMAGE"
echo "Output: $OUTPUT_FILE"
echo "Partition: $PARTITION"
echo "============================================"

# Check if file already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Warning: File already exists: $OUTPUT_FILE"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
fi

# Create temporary SLURM script
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
echo "Starting Docker image import"
echo "============================================"
echo "Image: ${DOCKER_IMAGE}"
echo "Output: ${OUTPUT_FILE}"
echo "Node: \$(hostname)"
echo "Time: \$(date)"
echo "============================================"

# Remove old file (if exists)
rm -f "${OUTPUT_FILE}"

# Run enroot import
echo ""
echo "Downloading and converting image..."
enroot import -o "${OUTPUT_FILE}" docker://${DOCKER_IMAGE}

# Check result
if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "============================================"
    echo "Import succeeded!"
    echo "============================================"
    ls -lh "${OUTPUT_FILE}"
    echo ""
    echo "Add to srtslurm.yaml:"
    echo ""
    echo "containers:"
    echo "  ${OUTPUT_NAME}: \"${OUTPUT_FILE}\""
    echo ""
else
    echo ""
    echo "============================================"
    echo "Import failed!"
    echo "============================================"
    exit 1
fi

echo "Completed at: \$(date)"
EOF

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
echo "  tail -f ${LOG_DIR}/enroot_import_${JOB_ID}.log"
echo ""
echo "Check files after completion:"
echo "  ls -lh ${OUTPUT_FILE}"
echo "============================================"
