#!/usr/bin/env bash
# Prepare the LRTransfer environment and training data.
#
# Usage:
#   ./prepare.sh
#
# Downloads the full FineWeb100B dataset (1030 chunks, each 100M tokens,
# ~200MB on disk — ~200GB total).
#
# Environment overrides:
#   ENV_NAME    Conda environment name (default: lr-transfer)
#   PYTHON_VER  Python version for the environment (default: 3.12)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-lr-transfer}"
PYTHON_VER="${PYTHON_VER:-3.12}"
NUM_CHUNKS=1030

echo "==> Project dir : ${SCRIPT_DIR}"
echo "==> Conda env   : ${ENV_NAME} (python ${PYTHON_VER})"
echo "==> Data chunks : ${NUM_CHUNKS}"

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first:" >&2
    echo "       https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

# Make 'conda activate' available in this non-interactive shell.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "==> Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "==> Creating conda env '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y
fi

conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------------------------------
echo "==> Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# 3. Training data (FineWeb100B, GPT-2 tokens)
# ---------------------------------------------------------------------------
echo "==> Downloading FineWeb100B data (${NUM_CHUNKS} chunks)..."
echo "    (already-downloaded shards are skipped automatically)"
python "${SCRIPT_DIR}/data/cached_fineweb100B.py" "${NUM_CHUNKS}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo "==> Setup complete."
echo "    Activate the environment with:  conda activate ${ENV_NAME}"
echo "    Launch training with:           torchrun --nproc_per_node=8 train.py"
