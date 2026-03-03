#!/bin/bash
set -e

BRANCH="gb300_blog"

cd /sgl-workspace/sglang
git remote remove origin
git remote add origin https://github.com/YAMY1234/sglang.git
git fetch origin
git checkout origin/${BRANCH}

# ========== Set up pip cache to shared directory ==========
# All nodes share this cache to avoid redundant downloads
export PIP_CACHE_DIR=/configs/.pip-cache
mkdir -p "$PIP_CACHE_DIR"
echo "Using pip cache directory: $PIP_CACHE_DIR"

# PEP 668: Allow pip to install system-wide in container
PIP_ARGS="--break-system-packages"

echo "=== GB300 Setup: Upgrading FlashInfer for sm_103a support ==="

# Uninstall existing flashinfer packages
echo "Uninstalling existing flashinfer packages..."
pip uninstall -y flashinfer-python $PIP_ARGS 2>/dev/null || true
pip uninstall -y flashinfer $PIP_ARGS 2>/dev/null || true
pip uninstall -y flashinfer-jit-cache $PIP_ARGS 2>/dev/null || true

# Install latest FlashInfer with CUDA 13.0 support (flashinfer_cutedsl)
echo "Installing latest FlashInfer..."
pip install flashinfer-python==0.6.1 flashinfer-cubin==0.6.1 $PIP_ARGS

# Verify installation
echo "Verifying FlashInfer installation..."
python -c "import flashinfer; print(f'FlashInfer version: {flashinfer.__version__}')" || echo "Warning: Could not verify FlashInfer version"

echo "=== FlashInfer upgrade complete ==="

# Install sgl-kernel 0.3.21 for CUDA 13.0 (aarch64)
echo "Installing sgl-kernel 0.3.21..."
pip install https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_aarch64.whl --force-reinstall $PIP_ARGS
echo "sgl-kernel 0.3.21 installed"

# Install cutlass-dsl 4.3.4
pip uninstall -y nvidia-cutlass-dsl-libs-base $PIP_ARGS 2>/dev/null || true
pip install nvidia-cutlass-dsl==4.3.4 $PIP_ARGS --force-reinstall

echo "=== GB300 Setup complete ==="

