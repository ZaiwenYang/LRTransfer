# LR Transfer

Training code for studying learning rate transfer in GPT-2 style models, with support for multiple optimizers including AdamW, [Muon](https://kellerjordan.github.io/posts/muon/), and [AdamH](https://tinyurl.com/muonh). Built on top of the open-source [nanoGPT](https://github.com/karpathy/nanogpt) codebase.

> **Note on Muon.** Our current work is **not** related to the [Muon optimizer](https://kellerjordan.github.io/posts/muon/). However, in the early stage of this project we explored Muon as well, and the corresponding code paths (the `muon` optimizer choice, all `--backend*` and `--layer_wise_lr` flags, and the Newton-Schulz error / singular-value diagnostics) have been left in the script for reference. They are **not used** in the experiments reported in our work, and can be safely ignored when reproducing our results.
 

## Features

- Three optimizer choices: `adam`, `muon`, `adamh`
- WSD, cosine, sine-wave, and triangle-wave learning rate schedules
- Optional 2-power learning rate parametrization (`--use_power_learning_rate`)
- Checkpointing with full training resumption (model + optimizer states)
- CSV metrics logging (`metrics.csv`) as the primary source for data analysis, plus automatic loss curve plotting
- Optional [Weights & Biases](https://wandb.ai/) integration  for live experiment monitoring

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
| `--target_norms` | `None` | JSON file containing initialization weight norms; applied after `--init_ckpt` when both are provided |

### Evaluation & Logging

| Argument | Default | Description |
| --- | --- | --- |
| `--val_loss_every` | `100` | Run validation every N steps (`0` = only at the end) |
| `--val_tokens` | `10485760` | Number of validation tokens per evaluation |
| `--save_every` | `5000` | Checkpoint frequency in steps (`0` = only at the end) |
| `--use_wandb` | `False` | Enable Weights & Biases Monitor. We often don't use wandb, as we collect more detailed data in a `metrics.csv`. |
| `--wandb_project` | `LRTransfer` | W&B project name |
| `--wandb_run_name` | `gpt-2` | W&B run name (falls back to `run_id` if `None`) |

### Backup & Checkpoints

| Argument | Default | Description |
| --- | --- | --- |
| `--enable_backup` | `False` | Copy logs and W&B run folder to `--backup_root` after training |
| `--backup_root` | `logs` | Destination root for backups |
| `--write_file` | `final_val_loss.txt` | Append the final validation loss to this file |
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
    --save_every 5000 \
    --write_file final_val_loss.txt
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
    --save_every 5000 \
    --write_file final_val_loss.txt
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
unique trainable parameter. Do not add `lm_head.weight`, because it is tied to
`transformer.wte.weight`. Target norms only apply to fresh initialization and
cannot be combined with `--resume_ckpt`. When `--init_ckpt` is also provided,
the checkpoint weights are loaded first and each tensor is then rescaled to its
target norm.

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
    ├── losses.png                # train/val loss curves (auto-generated at the end)
    └── models/
        ├── state_step000250.pt
        ├── state_step000500.pt
        └── ...
```

The `metrics.csv` columns are: `step, phase, train_loss, val_loss, total_ns_error, max_singular, min_singular, avg_singular, training_time_ms, step_avg_time_ms, learning_rate, grad_norms_json, optimizer_norms_json, update_norms_json, weight_norms_json, correlations_json`.

The final validation loss is also appended to the file specified by `--write_file` (default `final_val_loss.txt`) along with the optimizer settings, making it easy to monitor the experiment.

## Acknowledgments

This codebase builds on [nanoGPT](https://github.com/karpathy/nanogpt) by Andrej Karpathy and the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) by Keller Jordan.
