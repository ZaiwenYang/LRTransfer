#!/usr/bin/env bash
# Resume a run from its latest checkpoint and linearly decay the learning
# rate to zero until the total iteration count reaches NUM_ITERS.
#
# Usage:
#   ./decay.sh <NUM_ITERS> [NUM_GPUS] <CKPT_ROOT>
#
#   NUM_ITERS   Total iteration count to train to. The checkpoint at step
#               NUM_ITERS - 5000 is resumed, and the LR decays linearly to
#               zero over the final 5000 steps.
#   NUM_GPUS    Number of GPUs on this node (default: 8).
#   CKPT_ROOT   Output directory of a previous run.sh run. The checkpoint is
#               loaded from <CKPT_ROOT>/models/, and the base learning rate
#               is read from <CKPT_ROOT>/config.yaml.
#
# Outputs are written to <CKPT_ROOT>_decay<NUM_ITERS> so the original run's
# config.yaml / metrics.csv are not overwritten. All other hyperparameters
# match run.sh (global batch size fixed at 512).

set -euo pipefail

WARMDOWN_ITERS=5000

if [[ $# -eq 2 ]]; then
    NUM_ITERS="$1"
    NUM_GPUS=8
    CKPT_ROOT="${2%/}"
elif [[ $# -eq 3 ]]; then
    NUM_ITERS="$1"
    NUM_GPUS="$2"
    CKPT_ROOT="${3%/}"
else
    echo "Usage: $0 <NUM_ITERS> [NUM_GPUS] <CKPT_ROOT>" >&2
    exit 1
fi

BATCH_SIZE=512
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "ERROR: NUM_GPUS=${NUM_GPUS} must evenly divide batch_size=${BATCH_SIZE}" >&2
    exit 1
fi
DEVICE_BATCH_SIZE=$(( BATCH_SIZE / NUM_GPUS ))

# --- Locate the checkpoint at step NUM_ITERS - WARMDOWN_ITERS -------------
if (( NUM_ITERS <= WARMDOWN_ITERS )); then
    echo "ERROR: NUM_ITERS=${NUM_ITERS} must be greater than WARMDOWN_ITERS=${WARMDOWN_ITERS}" >&2
    exit 1
fi
STEP=$(( NUM_ITERS - WARMDOWN_ITERS ))

# checkpoints are named state_step%06d.pt (zero-padded to 6 digits)
RESUME_CKPT="${CKPT_ROOT}/models/state_step$(printf '%06d' "${STEP}").pt"
if [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${RESUME_CKPT}" >&2
    echo "Available checkpoints in ${CKPT_ROOT}/models/:" >&2
    ls "${CKPT_ROOT}/models/" >&2 || true
    exit 1
fi

# --- Recover the base learning rate from the original run's config --------
# config.yaml stores optimizer_lr after the 2^(-p) transform has been applied,
# so it is passed straight through WITHOUT --use_power_learning_rate.
CONFIG_YAML="${CKPT_ROOT}/config.yaml"
if [[ ! -f "${CONFIG_YAML}" ]]; then
    echo "ERROR: ${CONFIG_YAML} not found" >&2
    exit 1
fi
LR="$(awk '$1 == "optimizer_lr:" {print $2}' "${CONFIG_YAML}")"
if [[ -z "${LR}" ]]; then
    echo "ERROR: optimizer_lr not found in ${CONFIG_YAML}" >&2
    exit 1
fi

OUT_ROOT="${CKPT_ROOT}_decay${NUM_ITERS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Resume ckpt      : ${RESUME_CKPT} (step ${STEP})"
echo "==> Total iterations : ${NUM_ITERS}  (decay over ${WARMDOWN_ITERS} steps)"
echo "==> Base LR          : ${LR}"
echo "==> GPUs             : ${NUM_GPUS}"
echo "==> Device batch size: ${DEVICE_BATCH_SIZE}  (global batch = ${BATCH_SIZE})"
echo "==> Output dir       : ${OUT_ROOT}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${SCRIPT_DIR}/train.py" \
    --optimizer adam \
    --optimizer_lr "${LR}" \
    --optimizer_mom 0.95 \
    --weight_decay 0.1 \
    --num_iterations "${NUM_ITERS}" \
    --n_layer 12 \
    --n_head 20 \
    --n_embd 2560 \
    --batch_size "${BATCH_SIZE}" \
    --device_batch_size "${DEVICE_BATCH_SIZE}" \
    --val_loss_every 100 \
    --warmup_iters 1000 \
    --warmdown_iters "${WARMDOWN_ITERS}" \
    --save_every 5000 \
    --log_root "${OUT_ROOT}" \
    --resume_ckpt "${RESUME_CKPT}"
