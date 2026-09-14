#!/usr/bin/env bash
# Launch a single-node AdamW training run.
#
# Usage:
#   ./run.sh <POWER_LR> <NUM_GPUS> [CKPT_ROOT]
#
#   POWER_LR    Power learning rate exponent p; effective LR is 2^(-p)
#               (passed as --optimizer_lr with --use_power_learning_rate).
#   NUM_GPUS    Number of GPUs on this node.
#   CKPT_ROOT   Directory for run outputs; checkpoints are saved to
#               <CKPT_ROOT>/models/ (default: logs). NOTE: no per-run
#               subdirectory is created, so give each run its own CKPT_ROOT
#               to avoid overwriting a previous run's outputs.
#
# The global batch size is fixed at 512 sequences regardless of GPU count
# (train.py accumulates gradients as needed).

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <POWER_LR> <NUM_GPUS> [CKPT_ROOT]" >&2
    exit 1
fi

POWER_LR="$1"
NUM_GPUS="$2"
CKPT_ROOT="${3:-logs}"

BATCH_SIZE=512
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "ERROR: NUM_GPUS=${NUM_GPUS} must evenly divide batch_size=${BATCH_SIZE}" >&2
    exit 1
fi
DEVICE_BATCH_SIZE=$(( BATCH_SIZE / NUM_GPUS ))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Power LR         : ${POWER_LR}  (effective LR = 2^-${POWER_LR})"
echo "==> GPUs             : ${NUM_GPUS}"
echo "==> Device batch size: ${DEVICE_BATCH_SIZE}  (global batch = ${BATCH_SIZE})"
echo "==> Checkpoint root  : ${CKPT_ROOT}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${SCRIPT_DIR}/train.py" \
    --optimizer adam \
    --use_power_learning_rate \
    --optimizer_lr "${POWER_LR}" \
    --optimizer_mom 0.95 \
    --weight_decay 0.1 \
    --num_iterations 200000 \
    --n_layer 12 \
    --n_head 20 \
    --n_embd 2560 \
    --batch_size "${BATCH_SIZE}" \
    --device_batch_size "${DEVICE_BATCH_SIZE}" \
    --val_loss_every 100 \
    --warmup_iters 1000 \
    --warmdown_iters 1000 \
    --save_every 5000 \
    --log_root "${CKPT_ROOT}"
