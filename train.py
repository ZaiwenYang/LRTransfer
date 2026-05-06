import os
import sys
import argparse
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import re
from dataclasses import dataclass
from datetime import datetime
import yaml

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import socket

# Import our modules
from models.nanoGPT import GPT, GPTConfig
from optim.muon import Muon
from optim.adamh import AdamH

from tqdm import tqdm

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be limited to console/file")

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self, start_step=0):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

        for _ in range(start_step):
            self.current_position += self.B * self.T * self.num_processes
            if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
                self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 model')
    
    # Data hyperparameters
    parser.add_argument('--input_bin', type=str, default='data/fineweb100B/fineweb_train_*.bin',
                        help='input .bin to train on')
    parser.add_argument('--input_val_bin', type=str, default='data/fineweb100B/fineweb_val_*.bin',
                        help='input .bin to eval validation loss on') 
    
    # Optimization hyperparameters
    parser.add_argument('--optimizer', type=str, default='muon',
                        choices=['adam', 'muon', 'adamh'],
                        help='optimizer to use: adam, muon, or adamh')
    parser.add_argument('--batch_size', type=int, default=8*64,
                        help='batch size, in sequences, across all devices')
    parser.add_argument('--device_batch_size', type=int, default=64,
                        help='batch size, in sequences, per device')
    parser.add_argument('--sequence_length', type=int, default=1024,
                        help='sequence length, in tokens')
    parser.add_argument('--num_iterations', type=int, default=2000,
                        help='number of iterations to run')
    parser.add_argument('--warmup_iters', type=int, default=250,
                        help='number of warmup iterations')
    parser.add_argument('--warmdown_iters', type=int, default=1450,
                        help='number of iterations of linear warmdown for triangular or trapezoidal schedule')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')

    # Muon Setting
    parser.add_argument('--backend', type=str, default='newtonschulz5',
                        help='backend for Muon optimizer')
    parser.add_argument('--embed_lr', type=float, default=0.0036,
                        help='learning rate for embedding layers')
    parser.add_argument('--optimizer_lr', type=float, default=0.02,
                        help='learning rate for Muon optimizer')
    parser.add_argument('--optimizer_mom', type=float, default=0.95,
                        help='momentum for Muon optimizer')
    parser.add_argument('--backend_steps', type=int, default=10,
                        help='number of backend steps for Muon optimizer')
    parser.add_argument('--use_power_learning_rate', action='store_true',
                        help="whether to use power learning rate for Muon optimizer")
    parser.add_argument('--layer_wise_lr', type=str, default='Keller-Jordan',
                        help='layer-wise learning rate schedule for Muon optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='gradient clipping value')
           
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=50304,
                        help='vocabulary size')
    parser.add_argument('--n_layer', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--n_head', type=int, default=6,
                        help='number of attention heads')
    parser.add_argument('--n_embd', type=int, default=768,
                        help='embedding dimension')
    
    # Evaluation and logging hyperparameters
    parser.add_argument('--val_loss_every', type=int, default=125,
                        help='every how many steps to evaluate val loss? 0 for only at the end')
    parser.add_argument('--val_tokens', type=int, default=10485760,
                        help='how many tokens of validation data?')
    parser.add_argument('--save_every', type=int, default=250,
                        help='every how many steps to save the checkpoint? 0 for only at the end')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='whether to use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='LRTransfer',
                        help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='gpt-2',
                        help='wandb run name, if None will use the run_id')

    # Backup options
    parser.add_argument('--enable_backup', action='store_true',
                        help='enable backup of logs and wandb run folder to backup_root')
    parser.add_argument('--backup_root', type=str, default='logs',
                        help='root folder to copy logs to when --enable_backup is set')
    parser.add_argument("--write_file", type=str, default="final_val_loss.txt",
                        help="File to show the final val loss")
    
    # Resume training
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='path to checkpoint to resume training from')
    
    return parser.parse_args()


def custom_adam(args, raw_model, ddp_rank, ddp_world_size):
    optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.embed_lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay, fused=True)
    optimizer2 = torch.optim.AdamW(raw_model.transformer.h.parameters(), lr=args.optimizer_lr, betas=(args.optimizer_mom, 0.95),
                                   weight_decay=args.weight_decay, fused=True)
    optimizers = [optimizer1, optimizer2]

    return optimizers

def custom_adamh(args, raw_model, ddp_rank, ddp_world_size):
    optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.embed_lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay, fused=True)
    optimizer2 = AdamH(
        raw_model.transformer.h.parameters(),
        lr=args.optimizer_lr,
        betas=(args.optimizer_mom, 0.95),
        weight_decay=0.0,
        radius=-1.0,          
        amsgrad=False,
        fused=True
    )
    optimizers = [optimizer1, optimizer2]

    return optimizers


def custom_muon(args, raw_model, ddp_rank, ddp_world_size):
    optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.optimizer_lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay, fused=True)
    optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.optimizer_lr, momentum=args.optimizer_mom, backend=args.backend, backend_steps=args.backend_steps, layerwise_lr=args.layer_wise_lr,
                                    rank=ddp_rank, world_size=ddp_world_size)
    optimizers = [optimizer1, optimizer2]

    return optimizers


def optim_choice():
    return {
        'adam': custom_adam,
        'muon': custom_muon,
        'adamh': custom_adamh
    }

# -----------------------------------------------------------------------------
# Main training function

def main():
    args = parse_args()
    if args.use_power_learning_rate:
        print("Using 2-power learning rate for optimizers")
        args.optimizer_lr = 2 ** (-args.optimizer_lr)
        print(f"2-power learning rate: {args.optimizer_lr}")
        
    
    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
    
    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    if master_process:
        print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

    # create model
    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    model = GPT(model_config)
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)
    
    # Load model state from checkpoint early if resuming (before DDP and compile)
    start_step = 0
    ckpt_optimizers_state = None
    if args.resume_ckpt:
        print(f"Loading model state from checkpoint {args.resume_ckpt}...")
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        start_step = ckpt.get('step', 0)
        ckpt_optimizers_state = ckpt.get('optimizers', None)
        print(f"Restored model state, resuming from step {start_step}")
    
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    
    # print model size
    if master_process:
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f'Model has {n_params/1e6:.2f} million parameters.')

    # init the optimizer(s)
    optimizers = optim_choice()[args.optimizer](args, raw_model, ddp_rank, ddp_world_size)
    
    # Restore optimizer states from checkpoint if available
    if ckpt_optimizers_state is not None and args.resume_ckpt:
        assert len(ckpt_optimizers_state) == len(optimizers), \
            f"Number of optimizers in checkpoint ({len(ckpt_optimizers_state)}) " \
            f"does not match current setup ({len(optimizers)})"
        for i, opt_state in enumerate(ckpt_optimizers_state):
            optimizers[i].load_state_dict(opt_state)
        print(f"Restored optimizer states for {len(optimizers)} optimizers")
    
    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return decay_ratio
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
    if args.resume_ckpt and start_step > 0:
        for scheduler in schedulers:
            for _ in range(start_step):
                scheduler.step()

    # begin logging
    run_id = None
    logdir = None
    logfile = None
    if master_process:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = 'logs/%s/' % run_id
        modeldir = os.path.join(logdir, 'models')
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(modeldir, exist_ok=True)
        config_dict = {
            'optimizer': args.optimizer,
            'batch_size': args.batch_size,
            'device_batch_size': args.device_batch_size,
            'sequence_length': args.sequence_length,
            'start_step': start_step,
            'num_iterations': args.num_iterations,
            'warmup_iters': args.warmup_iters,
            'warmdown_iters': args.warmdown_iters,
            'weight_decay': args.weight_decay,
            'optimizer_lr': args.optimizer_lr,
            'optimizer_mom': args.optimizer_mom,
            'embed_lr': args.embed_lr,
            'backend': args.backend,
            'backend_steps': args.backend_steps,
            'vocab_size': args.vocab_size,
            'n_layer': args.n_layer,
            'n_head': args.n_head,
            'n_embd': args.n_embd,
            'grad_clip': args.grad_clip,
        }
        config_path = os.path.join(logdir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logfile = os.path.join(logdir, 'logs.txt')
        metrics_csv = os.path.join(logdir, 'metrics.csv')
        # initialize CSV with header
        with open(metrics_csv, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            # include json columns for per-parameter norms
            writer.writerow([
                'step', 'phase', 'train_loss', 'val_loss', 'total_ns_error',
                'max_singular', 'min_singular', 'training_time_ms',
                'step_avg_time_ms', 'learning_rate',
                'grad_norms_json', 'optimizer_norms_json', 'update_norms_json', 'weight_norms_json', 'correlations_json'
            ])
        # create the log file
        with open(logfile, "w") as f:
            # begin the log by printing this file (the Python code)
            f.write('='*100 + '\n')
            f.write(code)
            f.write('='*100 + '\n')
            # log information about the hardware/software environment this is running on
            # and print the full `nvidia-smi` to file
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'{result.stdout}\n')
            f.write('='*100 + '\n')

        # Initialize wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else str(run_id)
            wandb.init(
                mode="offline",
                project=args.wandb_project,
                name=wandb_run_name,
                config=vars(args)  # log all arguments
            )
            print(f"Initialized wandb project: {args.wandb_project}, run: {wandb_run_name}")
        elif args.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available")

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    # begin training
    train_loader.reset(start_step=start_step)
    x, y = train_loader.next_batch()
    # keep track of latest train stats so validation has something to record
    last_train_stats = None
    for step in tqdm(range(start_step, args.num_iterations + 1), desc="Training Steps"):
        last_step = (step == args.num_iterations)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        # once in a while evaluate the validation dataset
        if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            if master_process:
                step_avg = training_time_ms/(timed_steps-1) if timed_steps > 1 else 0
                ns_err = optimizers[1].total_error if args.optimizer == "muon" else 0.0
                print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} ns_error:{ns_err:.4f} train_time:{training_time_ms:.0f}ms step_avg:{step_avg:.2f}ms')
                with open(logfile, "a") as f:
                    f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} ns_error:{ns_err:.4f} train_time:{training_time_ms:.0f}ms step_avg:{step_avg:.2f}ms\n')
                    # append to CSV metrics
                    try:
                        val_loss_val = float(val_loss.item()) if hasattr(val_loss, 'item') else float(val_loss)
                    except Exception:
                        val_loss_val = float(val_loss)
                    tot_err_val = float(optimizers[1].total_error) if args.optimizer == "muon" else 0.0
                    max_s_val = float(optimizers[1].max_singular) if args.optimizer == "muon" else 0.0
                    min_s_val = float(optimizers[1].min_singular) if args.optimizer == "muon" else 0.0
                    avg_s_val = float(optimizers[1].avg_singular) if args.optimizer == "muon" else 0.0
                    current_lr = get_lr(step) if step < args.num_iterations else 0
                    with open(metrics_csv, 'a', newline='') as csvf:
                        writer = csv.writer(csvf)
                        if last_train_stats is not None:
                            t_loss, t_grad, t_optimizer, t_update, t_weight, t_correlations = last_train_stats
                            writer.writerow([step, 'val', t_loss, val_loss_val, tot_err_val, max_s_val, min_s_val, avg_s_val, training_time_ms, step_avg, current_lr, json.dumps(t_grad), json.dumps(t_optimizer), json.dumps(t_update), json.dumps(t_weight), json.dumps(t_correlations)])
                        else:
                            writer.writerow([step, 'val', '', val_loss_val, tot_err_val, max_s_val, min_s_val, avg_s_val, training_time_ms, step_avg, current_lr, '', '', '', '', ''])

                # Log to wandb
                if args.use_wandb and WANDB_AVAILABLE:
                    current_lr = get_lr(step) if step < args.num_iterations else 0
                    wandb.log({
                        "val_loss": val_loss.item(),
                        "step": step,
                        "training_time_ms": training_time_ms,
                        "step_avg_time_ms": step_avg,
                        "learning_rate": current_lr,
                        "total_ns_error": tot_err_val,
                        "max_singular": max_s_val,
                        "min_singular": min_s_val,
                        "avg_singular": avg_s_val
                    }, step=step)
                    
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            save_path = os.path.join(modeldir, 'state_step%06d.pt' % step)
            torch.save(log, save_path)
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        for i in range(1, train_accumulation_steps+1):
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            if i < train_accumulation_steps:
                with model.no_sync(): # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                loss.backward() # just sync on the last step
        # normalize gradients accumulated across steps
        for p in model.parameters():
            p.grad /= train_accumulation_steps

        # compute gradient norm *per parameter* for q, k, v, linear layers
        grad_norms_per = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            # only log the requested types
            if any(tag in name for tag in ['.c_q', '.c_k', '.c_v', '.c_fc', '.c_proj']):
                grad_norms_per[name] = p.grad.detach().norm().item()

        if args.grad_clip > 0:
            clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            clip_norm = None

        # Build parameter -> optimizer mapping for later norm computation
        param_id_to_opt_info = {}
        for opt_idx, opt in enumerate(optimizers):
            for param_group in opt.param_groups:
                for param in param_group['params']:
                    param_id_to_opt_info[id(param)] = (opt_idx, opt, param_group)

        # Get current learning rates before stepping
        current_lrs = [sched.get_last_lr()[0] for sched in schedulers]

        # capture parameter values before optimizer step for update norm
        old_params = {name: p.detach().clone() for name, p in model.named_parameters()}

        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        # compute update norm, optimizer norm, and weight norm per parameter
        update_norms_per = {}
        optimizer_norms_per = {}
        weight_norms_per = {}
        correlations_per = {}
        for name, p in model.named_parameters():
            prev = old_params.get(name)
            if prev is not None:
                update_norms_per[name] = (p.data - prev).norm().item()
                
                # Compute optimizer norm (u_t before learning rate and weight decay scaling)
                # From update equation: w_{t+1} = w_t - η(u_t + λw_t)
                # Therefore: u_t = -(w_{t+1} - w_t)/η - λw_t
                param_id = id(p)
                if param_id in param_id_to_opt_info:
                    opt_idx, opt, param_group = param_id_to_opt_info[param_id]
                    lr = param_group.get('lr', current_lrs[opt_idx])
                    weight_decay = param_group.get('weight_decay', 0)
                    
                    update = p.data - prev  # This is -η(u_t + λw_t)
                    if lr != 0:
                        # u_t = -update/η - λw_t
                        u_t = -update / lr - weight_decay * prev
                        optimizer_norms_per[name] = u_t.norm().item()
                        
                        # Compute correlation: inner_prod(u_t, w_t) / ||u_t||^2
                        u_t_norm_sq = u_t.norm() ** 2
                        if u_t_norm_sq > 0:
                            correlation = torch.dot(u_t.flatten(), prev.flatten()) / u_t_norm_sq
                            correlations_per[name] = correlation.item()
                        else:
                            correlations_per[name] = 0.0
            
            weight_norms_per[name] = p.data.norm().item()

        # compute simple summaries for printing
        # e.g. number of tracked params of each type
        counts = {'q':0,'k':0,'v':0,'linear':0}
        for name in grad_norms_per:
            if '.c_q' in name: counts['q'] += 1
            if '.c_k' in name: counts['k'] += 1
            if '.c_v' in name: counts['v'] += 1
            if '.c_fc' in name or '.c_proj' in name: counts['linear'] += 1
        # store latest train metrics so validation can record them
        last_train_stats = (
            float(train_loss.item()) if hasattr(train_loss, 'item') else float(train_loss),
            grad_norms_per,
            optimizer_norms_per,
            update_norms_per,
            weight_norms_per,
            correlations_per,
        )

        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            step_avg = approx_time/timed_steps if timed_steps > 0 else 0
            # print a brief summary rather than every parameter
            grad_info = (
                f"q={counts['q']} k={counts['k']} v={counts['v']} lin={counts['linear']}"
            )
            if clip_norm is not None:
                grad_info += f" clip_norm:{clip_norm:.4f}"
            ns_err = optimizers[1].total_error if args.optimizer == 'muon' else 0.0
            print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} {grad_info} ns_error:{ns_err:.4f} train_time:{approx_time:.0f}ms step_avg:{step_avg:.2f}ms")
            # with open(logfile, "a") as f:
            #     f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} ns_error:{optimizers[1].total_error if args.optimizer == "muon" else 0:.4f} train_time:{approx_time:.0f}ms step_avg:{step_avg:.2f}ms\n")
            
            #     # append to CSV metrics
            #     try:
            #         train_loss_val = float(train_loss.item()) if hasattr(train_loss, 'item') else float(train_loss)
            #     except Exception:
            #         train_loss_val = float(train_loss)
            #     tot_err_val = float(optimizers[1].total_error) if args.optimizer == "muon" else 0.0
            #     max_s_val = float(optimizers[1].max_singular) if args.optimizer == "muon" else 0.0
            #     min_s_val = float(optimizers[1].min_singular) if args.optimizer == "muon" else 0.0
            #     avg_s_val = float(optimizers[1].avg_singular) if args.optimizer == "muon" else 0.0
            #     current_lr = get_lr(step+1) if (step+1) < args.num_iterations else 0
            #     with open(metrics_csv, 'a', newline='') as csvf:
            #         writer = csv.writer(csvf)
            #         # include norms JSON
            #         writer.writerow([step+1, 'train', train_loss_val, '', tot_err_val, max_s_val, min_s_val, approx_time, step_avg, current_lr, json.dumps(grad_norms_per), json.dumps(update_norms_per), json.dumps(weight_norms_per)])
            # Log training loss to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                # Get total_error from Newton-Schulz optimizer (index 1 for ours, may not exist for others)
                # log per-parameter norms to wandb using hierarchical keys
                log_dict = {"train_loss": train_loss.item(), "step": step + 1}
                for name, val in grad_norms_per.items():
                    log_dict[f"grad_norm/{name}"] = val
                for name, val in optimizer_norms_per.items():
                    log_dict[f"optimizer_norm/{name}"] = val
                for name, val in update_norms_per.items():
                    log_dict[f"update_norm/{name}"] = val
                for name, val in weight_norms_per.items():
                    log_dict[f"weight_norm/{name}"] = val
                for name, val in correlations_per.items():
                    log_dict[f"correlation/{name}"] = val
                if clip_norm is not None:
                    log_dict["clip_norm"] = clip_norm
                wandb.log(log_dict, step=step + 1)
                if args.optimizer == "muon":
                    wandb.log({
                        "total_ns_error": optimizers[1].total_error,
                        "max_singular": optimizers[1].max_singular,
                        "min_singular": optimizers[1].min_singular,
                        "avg_singular": optimizers[1].avg_singular,
                    }, step=step + 1)

    if master_process:
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
        
        # Finish wandb run
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        # -----------------------------------------------------------------
        # Backup logs and wandb run folder to a centralized filesystem if requested
        # Ensure uniqueness by including hostname and timestamp so multi-node runs
        # don't overwrite each other's backups.
        # -----------------------------------------------------------------
        try:
            backup_root = args.backup_root
            if args.enable_backup:
                if os.path.exists(backup_root):
                    hostname = socket.gethostname()
                    ts = int(time.time())
                    # backup the local logdir (created earlier)
                    if logdir and os.path.exists(logdir):
                        dest = os.path.join(backup_root, f"{run_id}_{hostname}_{ts}")
                        try:
                            shutil.copytree(logdir, dest, dirs_exist_ok=True)
                            print(f'Backed up logs to {dest}')
                            if logfile:
                                with open(logfile, 'a') as f:
                                    f.write(f'Backed up logs to {dest}\n')
                        except Exception as e:
                            print(f'Warning: failed to backup logs to {dest}: {e}')
                            if logfile:
                                with open(logfile, 'a') as f:
                                    f.write(f'Warning: failed to backup logs to {dest}: {e}\n')

                    # backup wandb run directory if wandb was used and run dir is available
                    if args.use_wandb and WANDB_AVAILABLE:
                        try:
                            run_dir = wandb.run.dir if hasattr(wandb, 'run') and wandb.run is not None else None
                            if run_dir and os.path.exists(run_dir):
                                dest_w = os.path.join(backup_root, f'wandb_run_{run_id}_{hostname}_{ts}')
                                try:
                                    shutil.copytree(run_dir, dest_w, dirs_exist_ok=True)
                                    print(f'Backed up wandb run folder to {dest_w}')
                                    if logfile:
                                        with open(logfile, 'a') as f:
                                            f.write(f'Backed up wandb run folder to {dest_w}\n')
                                except Exception as e:
                                    print(f'Warning: failed to backup wandb run folder to {dest_w}: {e}')
                                    if logfile:
                                        with open(logfile, 'a') as f:
                                            f.write(f'Warning: failed to backup wandb run folder to {dest_w}: {e}\n')
                        except Exception as e:
                            print(f'Warning: error while locating wandb.run.dir: {e}')
                else:
                    print(f'Backup destination {backup_root} does not exist, skipping backup')
            else:
                print('Backup not enabled (use --enable_backup)')
        except Exception as e:
            print(f'Warning: backup step failed: {e}')
            if logfile:
                with open(logfile, 'a') as f:
                    f.write(f'Warning: backup step failed: {e}\n')
        try:
            output_file = args.write_file

            # Ensure file exists; if not, create an empty one
            if not os.path.exists(output_file):
                with open(output_file, "w") as f:
                    pass  # just create empty file

            # Append rather than overwrite
            with open(output_file, "a") as f:
                f.write("\n")  # separate from previous content
                f.write("==== New Training Run ====\n")
                f.write(f"optimizer_type: {args.optim_choice}\n")
                f.write(f"optimizer_lr: {args.optimizer_lr}\n")
                f.write(f"optimizer_mom: {args.optimizer_mom}\n")
                f.write(f"n_embd: {args.n_embd}\n")
                f.write(f"final_val_loss: {val_loss.item() if hasattr(val_loss, 'item') else float(val_loss)}\n")

            print(f"Appended final metrics to {output_file}")

        except Exception as e:
            print(f"Warning: failed to write data.txt: {e}")

        # Generate plots from metrics.csv (if present)
        try:
            steps_train, loss_train = [], []
            steps_val, loss_val = [], []
            steps_err, err_vals = [], []
            with open(metrics_csv, 'r') as csvf:
                reader = csv.DictReader(csvf)
                for row in reader:
                    phase = row.get('phase', '')
                    step = int(row.get('step', 0)) if row.get('step', '') != '' else None
                    if phase == 'train':
                        if row.get('train_loss', '') != '':
                            steps_train.append(step)
                            loss_train.append(float(row['train_loss']))
                    elif phase == 'val':
                        if row.get('val_loss', '') != '':
                            steps_val.append(step)
                            loss_val.append(float(row['val_loss']))
                    # ns error may appear in either row
                    if row.get('total_ns_error', '') not in (None, ''):
                        try:
                            err = float(row['total_ns_error'])
                            if step is not None:
                                steps_err.append(step)
                                err_vals.append(err)
                        except Exception:
                            pass

            # plot losses
            if loss_train or loss_val:
                plt.figure()
                if loss_train:
                    plt.plot(steps_train, loss_train, label='train_loss')
                if loss_val:
                    plt.plot(steps_val, loss_val, label='val_loss')
                plt.xlabel('step')
                plt.ylabel('loss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(logdir, 'losses.png'))
                plt.close()

            # # plot ns_error
            # if err_vals:
            #     plt.figure()
            #     plt.plot(steps_err, err_vals, label='total_ns_error')
            #     plt.xlabel('step')
            #     plt.ylabel('ns_error')
            #     plt.legend()
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(logdir, 'ns_error.png'))
            #     plt.close()

            # plot singular values (max and min)
            # try:
            #     steps_max, max_vals = [], []
            #     steps_min, min_vals = [], []
            #     with open(metrics_csv, 'r') as csvf:
            #         reader = csv.DictReader(csvf)
            #         for row in reader:
            #             step = int(row.get('step', 0)) if row.get('step', '') != '' else None
            #             if row.get('max_singular', '') not in (None, '') and step is not None:
            #                 try:
            #                     mv = float(row['max_singular'])
            #                     steps_max.append(step)
            #                     max_vals.append(mv)
            #                 except Exception:
            #                     pass
            #             if row.get('min_singular', '') not in (None, '') and step is not None:
            #                 try:
            #                     mv = float(row['min_singular'])
            #                     steps_min.append(step)
            #                     min_vals.append(mv)
            #                 except Exception:
            #                     pass

            #     if max_vals or min_vals:
            #         plt.figure()
            #         if max_vals:
            #             plt.plot(steps_max, max_vals, label='max_singular')
            #         if min_vals:
            #             plt.plot(steps_min, min_vals, label='min_singular')
            #         plt.xlabel('step')
            #         plt.ylabel('singular value')
            #         plt.legend()
            #         plt.grid(True)
            #         plt.tight_layout()
            #         plt.savefig(os.path.join(logdir, 'singular_values.png'))
            #         plt.close()
            # except Exception:
            #     # don't fail the whole plotting pipeline for singular value parse errors
            #     pass
        except Exception as e:
            print(f"Warning: failed to generate plots: {e}")

    # -------------------------------------------------------------------------
    # clean up nice
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

