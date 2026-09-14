# LR Transfer

Training code for studying learning rate transfer in GPT-2 style models, with support for multiple optimizers including AdamW, [Muon](https://kellerjordan.github.io/posts/muon/), and [AdamH](https://tinyurl.com/muonh). Built on top of the open-source [nanoGPT](https://github.com/karpathy/nanogpt) codebase.

> **Note on Muon.** Our current work is **not** related to the [Muon optimizer](https://kellerjordan.github.io/posts/muon/). However, in the early stage of this project we explored Muon as well, and the corresponding code paths (the `muon` optimizer choice and all `--backend*` and `--layer_wise_lr` flags) have been left in the script for reference. They are **not used** in the experiments reported in our work, and can be safely ignored when reproducing our results.
 

## Features

- Three optimizer choices: `adam`, `muon`, `adamh`
- WSD, cosine, sine-wave, and triangle-wave learning rate schedules
- Optional 2-power learning rate parametrization (`--use_power_learning_rate`)
- Checkpointing with full training resumption (model + optimizer states)
- CSV metrics logging (`metrics.csv`) as the primary source for data analysis
- Optional [Weights & Biases](https://wandb.ai/) and [SwanLab](https://swanlab.cn/) experiment tracking

## Environment

First, create and activate a Python 3.12 conda environment:

```shell
conda create -n lr-transfer python=3.12 -y
conda activate lr-transfer
```

Then install the dependencies:

```shell
pip install -r requirements.txt
```

## Prepare Data

We use FineWeb100B for training. The data is downloaded from an [open-source repo](https://huggingface.co/datasets/kjj0/fineweb100B-gpt2/tree/main) on Hugging Face. Fetch the data by:

```shell
python data/cached_fineweb100B.py 1030
```

By default, training expects the following paths (configurable via `--input_bin` and `--input_val_bin`):

- Training shards: `data/fineweb100B/fineweb_train_*.bin`
- Validation shards: `data/fineweb100B/fineweb_val_*.bin`

## Simple Running Command

Launch a single-node, 8-GPU training run with default hyperparameters:

```shell
torchrun --standalone --nproc_per_node=8 train.py
```

## Command-Line Arguments

### Data

| Argument | Default | Description |
| --- | --- | --- |
| `--input_bin` | `data/fineweb100B/fineweb_train_*.bin` | Glob pattern for training data shards |
| `--input_val_bin` | `data/fineweb100B/fineweb_val_*.bin` | Glob pattern for validation data shards |

### Optimization

| Argument | Default | Description |
| --- | --- | --- |
| `--optimizer` | `muon` | Which optimizer to use: `adam`, `muon`, or `adamh` |
| `--embed_lr` | `0.0036` | Learning rate for embedding / `lm_head` parameters |
| `--optimizer_lr` | `0.02` | Peak/base learning rate for the main optimizer |
| `--optimizer_mom` | `0.95` | Momentum (or first beta) for the main optimizer |
| `--use_power_learning_rate` | `False` | If set, the effective LR is `2 ** (-optimizer_lr)` |
| `--batch_size` | `512` | Global batch size in sequences (across all devices) |
| `--device_batch_size` | `64` | Per-device batch size in sequences |
| `--sequence_length` | `1024` | Sequence length in tokens |
| `--num_iterations` | `2000` | Total number of training iterations |
| `--warmup_iters` | `250` | Number of linear warmup iterations for every schedule |
| `--warmdown_iters` | `1450` | Number of linear warmdown iterations for WSD |
| `--lr_schedule` | `wsd` | Schedule: `wsd`, `cosine`, `sine`, or `triangle` |
| `--final_lr` | `0.0` | Final learning rate for the cosine schedule |
| `--lr_amplitude` | `0.0` | Amplitude `A` for sine and triangle schedules |
| `--lr_period` | `1000` | Period `T` in iterations for sine and triangle schedules |
| `--weight_decay` | `0` | Weight decay coefficient |
| `--grad_clip` | `1.0` | Gradient clipping value (0 disables clipping) |

### Muon-Specific

| Argument | Default | Description |
| --- | --- | --- |
| `--backend` | `newtonschulz5` | Backend for the Muon orthogonalization step |
| `--backend_steps` | `10` | Number of Newton-Schulz iterations |
| `--layer_wise_lr` | `Keller-Jordan` | Layer-wise learning rate schedule for Muon |

### Model

| Argument | Default | Description |
| --- | --- | --- |
| `--vocab_size` | `50304` | Vocabulary size |
| `--n_layer` | `12` | Number of transformer layers |
| `--n_head` | `6` | Number of attention heads |
| `--n_embd` | `768` | Embedding dimension |
| `--untie_embeddings` | `False` | Give the token embedding and `lm_head` separate weights instead of sharing one tensor |
| `--target_norms` | `None` | JSON file containing initialization weight norms; applied after `--init_ckpt` when both are provided |

### Evaluation & Logging

| Argument | Default | Description |
| --- | --- | --- |
| `--val_loss_every` | `100` | Run validation every N steps (`0` = only at the end) |
| `--val_tokens` | `10485760` | Number of validation tokens per evaluation |
| `--save_every` | `5000` | Checkpoint frequency in steps (`0` = only at the end) |
| `--use_wandb` | `False` | Enable W&B logging (flag; no value required) |
| `--wandb_project` | `LRTransfer` | W&B project name |
| `--wandb_run_name` | `None` | W&B experiment name (defaults to `run_id`) |
| `--wandb_mode` | `online` | W&B mode: `online`, `offline`, or `disabled` |
| `--use_swanlab` | `False` | Enable SwanLab logging; can be used together with `--use_wandb` |
| `--swanlab_project` | `LRTransfer` | SwanLab project name |
| `--swanlab_run_name` | `None` | SwanLab experiment name (defaults to `run_id`) |
| `--swanlab_mode` | `online` | SwanLab mode: `online`, `offline`, or `local` |

Both integrations are opt-in: pass `--use_wandb` and/or `--use_swanlab` without
a boolean value. Once enabled, both default to `online`. Log in with `wandb login`
or `swanlab login` before online runs. Both SDKs are imported only when enabled
on DDP rank 0; if an enabled SDK is missing, training exits with an import error.
Both experiment names default to the same timestamp-based `run_id` and can be
overridden with the corresponding `--*_run_name` argument.

W&B logs are stored under `<log_root>/wandb` by default, and SwanLab logs under
`<log_root>/swanlab`. Both are finalized at the end of training.
Mode choices remain SDK-specific: W&B offers `disabled`, while
SwanLab offers `local` for its optional dashboard.

To use W&B offline:

```shell
torchrun --standalone --nproc_per_node=8 train.py --use_wandb --wandb_mode offline
```

### SwanLab

SwanLab logs the same metrics as W&B, from DDP rank 0 only: training loss,
per-parameter gradient/optimizer/update/weight norms, correlations, and gradient
clipping norm on every training step; validation loss, learning rate, and timing
at validation steps. All command-line arguments are saved as the experiment configuration.

Two additional metrics are recorded for each parameter tensor after every optimizer
step (the tensor is treated as one flattened vector for L2 norms):

- `Approximate ELR/<parameter_name>`: `lr_t * optimizer_norm_t / ||w_t||_2`.
  The denominator is the **pre-update** weight norm. `lr_t` is the actual learning
  rate used by that parameter's optimizer group, captured before the scheduler
  advances. `optimizer_norm_t` is the existing reconstructed optimizer-direction
  norm, with decoupled weight decay removed. A zero LR gives zero Approximate ELR
  for nonzero weights.
- `Normalized Weight Change/<parameter_name>`:
  `||w_{t+1} / ||w_{t+1}||_2 - w_t / ||w_t||_2||_2`.

`Approximate ELR/average` and `Normalized Weight Change/average` are arithmetic
means over parameter tensors, excluding parameters in `wte` and `lm_head` modules
(including names prefixed by DDP or `torch.compile`). Each tensor has equal weight,
regardless of its number of elements. Embedding parameters still have individual
metrics; tied weights appear once, as in `model.named_parameters()`.

A zero weight norm makes the corresponding normalized metric undefined, recorded
as `NaN`; undefined values in included parameters propagate to the average. An empty
set of eligible parameters also yields `NaN`. At step 0, no update metrics exist yet.
W&B and SwanLab receive identical metrics; CSV stores these metrics at the existing
validation logging frequency, using the latest completed optimizer update.

```shell
# Offline logging (no login required)
torchrun --standalone --nproc_per_node=8 train.py \
    --use_swanlab --swanlab_mode offline --swanlab_run_name adam-lr7 \
    --optimizer adam --use_power_learning_rate --optimizer_lr 7 \
    --log_root logs/adam-lr7

# Online logging (default; log in once before launching training)
swanlab login
torchrun --standalone --nproc_per_node=8 train.py \
    --use_swanlab \
    --swanlab_project LRTransfer --swanlab_run_name gpt-2
```

SwanLab is included in `requirements.txt`; in an existing environment, install it
with `pip install swanlab`. Logs are stored in `<log_root>/swanlab`. Training and validation metrics
for the same completed step are submitted together, so each metric is logged only
once per step. Restoring `--resume_ckpt` creates a new tracking experiment whose
steps start at the restored checkpoint step.

For the optional local dashboard, install `pip install 'swanlab[dashboard]'`,
train with `--swanlab_mode local`, then run `swanlab watch <log_root>/swanlab`.
The `offline` mode stores data for later synchronization; `local` also writes
dashboard data. See the [SwanLab mode documentation](https://docs.swanlab.cn/api/py-init.html).

### Checkpoints

| Argument | Default | Description |
| --- | --- | --- |
| `--init_ckpt` | `None` | Path to a `.pt` checkpoint used only for initial model weights; training starts fresh at step 0 |
| `--resume_ckpt` | `None` | Path to a `.pt` checkpoint to resume from (restores model, optimizers, and step) |

## Example Commands

### Learning Rate Schedules

All schedules begin with linear warmup for `warmup_iters`. In the formulas
below, `t` counts optimizer steps after warmup:

- `wsd`: stable peak LR followed by linear warmdown.
- `cosine`: `final_lr + 0.5 * (optimizer_lr - final_lr) * (1 + cos(pi * t / (num_iterations - warmup_iters)))`.
- `sine`: `optimizer_lr + lr_amplitude * sin(2 * pi * (t + warmup_iters) / lr_period)`.
- `triangle`: `optimizer_lr + lr_amplitude * (2 / pi) * asin(sin(2 * pi * (t + warmup_iters) / lr_period))`.

The resulting multiplier is also applied to the embedding optimizer, preserving
the existing ratio between its learning rate and the main optimizer learning
rate. `final_lr` and `lr_amplitude` are absolute learning rates, including when
`--use_power_learning_rate` is used for `optimizer_lr`.

```shell
# Cosine from 0.02 to 0.001
torchrun --standalone --nproc_per_node=8 train.py \
    --lr_schedule cosine --optimizer_lr 0.02 --final_lr 0.001

# Sine wave centered at 0.02, amplitude 0.005, period 1000 steps
torchrun --standalone --nproc_per_node=8 train.py \
    --lr_schedule sine --optimizer_lr 0.02 \
    --lr_amplitude 0.005 --lr_period 1000

# Triangle wave with the same center, amplitude, and period
torchrun --standalone --nproc_per_node=8 train.py \
    --lr_schedule triangle --optimizer_lr 0.02 \
    --lr_amplitude 0.005 --lr_period 1000
```


### Train with AdamW

```shell
torchrun --standalone --nproc_per_node=8 train.py \
    --num_iterations 200000 \
    --n_layer 12 \ 
    --n_head 4 \
    --n_embd 512 \
    --optimizer adam \
    --use_power_learning_rate \
    --optimizer_lr 7 \
    --embed_lr 0.0036 \
    --weight_decay 0.1 \
    --val_loss_every 100 \
    --warmup_iters 1000 \
    --warmdown_iters 5000 \
    --save_every 5000
```

### Train with AdamH

```shell
torchrun --standalone --nproc_per_node=8 train.py \
    --num_iterations 200000 \
    --n_layer 12 \ 
    --n_head 4 \
    --n_embd 512 \
    --optimizer adamh \
    --use_power_learning_rate \
    --optimizer_lr 7 \
    --embed_lr 0.0036 \
    --val_loss_every 100 \
    --warmup_iters 1000 \
    --warmdown_iters 5000 \
    --save_every 5000
```

### Resume from a Checkpoint

The schedule and data loader resume at the checkpoint's saved global step.
`--num_iterations` remains the final global step, not the number of additional
steps. Repeat the original model, optimizer, batch/data, and schedule arguments,
and use a new `--log_root` to preserve the source run:

```shell
torchrun --standalone --nproc_per_node=8 train.py \
    --resume_ckpt ckpts/sine/models/state_step001000.pt \
    --num_iterations 200000 \
    --lr_schedule sine \
    --optimizer_lr 0.02 \
    --lr_amplitude 0.005 \
    --lr_period 1000 \
    --log_root ckpts/sine_resumed
```

### Initialize a Fresh Run from Checkpoint Weights

`--init_ckpt` loads only the checkpoint's model weights. The optimizer,
learning-rate schedule, data position, and step are initialized as a fresh run.
It cannot be combined with `--resume_ckpt`.

```shell
torchrun --standalone --nproc_per_node=8 train.py \
    --init_ckpt logs/source/models/state_step001000.pt \
    --num_iterations 200000
```

### Initialize with Target Weight Norms

The JSON file may include architecture metadata followed by the complete target
table. Each value is the norm of one whole tensor:

```json
{
  "metadata": {
    "vocab_size": 50304,
    "n_layer": 12,
    "n_head": 4,
    "n_embd": 512
  },
  "target_norms": {
    "transformer.wte.weight": 574.3865,
    "transformer.h.0.attn.c_q.weight": 31.3868
  }
}
```

The abbreviated table above is illustrative; the actual file must cover every
unique trainable parameter. By default, do not add `lm_head.weight`, because it
is tied to `transformer.wte.weight`. With `--untie_embeddings`, both
`transformer.wte.weight` and `lm_head.weight` are independent and must have
entries. Target norms only apply to fresh initialization and cannot be combined
with `--resume_ckpt`. When `--init_ckpt` is also provided, the checkpoint weights
are loaded first and each tensor is then rescaled to its target norm.

```shell
torchrun --standalone --nproc_per_node=8 train.py \
    --n_layer 12 \
    --n_head 4 \
    --n_embd 512 \
    --target_norms configs/target_norms_512.json
```

To rescale checkpoint initialization weights while preserving each tensor's
direction, combine the two fresh-initialization options:

```shell
torchrun --standalone --nproc_per_node=8 train.py \
    --init_ckpt logs/source/models/state_step001000.pt \
    --target_norms configs/target_norms_512.json
```

## Output Layout

Each run creates a timestamped directory under `logs/`:

```
logs/
└── 20260101_120000/              # run_id = YYYYMMDD_HHMMSS
    ├── config.yaml               # all hyperparameters used
    ├── logs.txt                  # full source code dump + nvidia-smi + step-by-step log
    ├── metrics.csv               # per-step metrics (train/val loss, norms, LR, etc.)
    └── models/
        ├── state_step000250.pt
        ├── state_step000500.pt
        └── ...
```

The `metrics.csv` columns are: `step, phase, train_loss, val_loss, training_time_ms, step_avg_time_ms, learning_rate, grad_norms_json, optimizer_norms_json, update_norms_json, weight_norms_json, correlations_json`.

The update metrics append four columns: `approximate_elrs_json`,
`normalized_weight_changes_json`, `approximate_elr_average`, and
`normalized_weight_change_average`. They are blank before the first optimizer
update (including an evaluation immediately after restoring a checkpoint).

## Acknowledgments

This codebase builds on [nanoGPT](https://github.com/karpathy/nanogpt) by Andrej Karpathy and the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) by Keller Jordan.
