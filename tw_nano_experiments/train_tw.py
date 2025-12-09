# NOTE: record from https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt
# ====================================================================================================
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Token Importance Weighting Configuration

@dataclass
class TokenWeightingConfig:
    """Configuration for token importance weighting ablations."""
    enabled: bool = False
    function: str = "linear"  # linear, sqrt, log, focal
    clamp_min: float = 0.5
    clamp_max: float = 2.0
    schedule: str = "constant"  # constant, warmup, anneal, cyclical, adaptive
    warmup_steps: int = 1000
    anneal_final_strength: float = 0.3  # for anneal schedule
    cyclical_period: int = 500  # for cyclical schedule
    focal_gamma: float = 2.0  # for focal loss style weighting
    percentile_clamp: bool = False  # use percentile-based clamping instead of fixed bounds
    percentile_low: float = 0.05  # lower percentile for clamping
    percentile_high: float = 0.95  # upper percentile for clamping
    log_weights: bool = True  # whether to log weight statistics
    log_every: int = 50  # log weight stats every N steps

def compute_schedule_strength(
    step: int,
    total_steps: int,
    config: TokenWeightingConfig,
    val_loss: Optional[float] = None
) -> float:
    """
    Compute the weighting strength multiplier based on the schedule.
    Returns a value in [0, 1] that scales the deviation from uniform weighting.
    """
    if config.schedule == "constant":
        return 1.0
    
    elif config.schedule == "warmup":
        # Linear warmup: start uniform, gradually introduce weighting
        return min(step / config.warmup_steps, 1.0)
    
    elif config.schedule == "anneal":
        # Annealing: start with strong weighting, decay toward uniform
        # decay_ratio goes from 1.0 to anneal_final_strength
        progress = step / total_steps
        return 1.0 - progress * (1.0 - config.anneal_final_strength)
    
    elif config.schedule == "cyclical":
        # Sine wave oscillation between weak and strong weighting
        return 0.5 + 0.5 * math.sin(step / config.cyclical_period * 2 * math.pi)
    
    elif config.schedule == "adaptive":
        # Validation-loss-based: strong when loss is high, weaker as loss decreases
        if val_loss is None:
            return 1.0  # Default to full strength if no val loss available
        # Normalize by approximate initial loss (~4.0 for GPT training)
        return min(val_loss / 4.0, 1.0)
    
    else:
        raise ValueError(f"Unknown schedule: {config.schedule}")

def compute_token_weights(
    per_token_loss: torch.Tensor,
    config: TokenWeightingConfig,
    step: int,
    total_steps: int,
    val_loss: Optional[float] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute importance weights for tokens based on their loss.
    
    Args:
        per_token_loss: Tensor of shape (batch_size * seq_len,) with per-token losses
        config: TokenWeightingConfig with weighting parameters
        step: Current training step
        total_steps: Total number of training steps
        val_loss: Current validation loss (for adaptive schedule)
    
    Returns:
        weights: Tensor of same shape as per_token_loss with importance weights
        stats: Dictionary with weight statistics for logging
    """
    if not config.enabled:
        return torch.ones_like(per_token_loss), {}
    
    # Detach to prevent gradient flow through weights
    loss_detached = per_token_loss.detach()
    
    # Compute raw weights based on weighting function
    if config.function == "linear":
        # Relative weighting: weight = loss / mean_loss
        mean_loss = loss_detached.mean()
        weights = loss_detached / (mean_loss + 1e-8)
    
    elif config.function == "sqrt":
        # Square root weighting: more moderate than linear
        mean_loss = loss_detached.mean()
        normalized_loss = loss_detached / (mean_loss + 1e-8)
        weights = torch.sqrt(normalized_loss)
    
    elif config.function == "log":
        # Logarithmic weighting: most conservative
        mean_loss = loss_detached.mean()
        normalized_loss = loss_detached / (mean_loss + 1e-8)
        # log1p for numerical stability (log(1 + x))
        weights = torch.log1p(normalized_loss)
    
    elif config.function == "focal":
        # Focal loss style: (loss / max_loss)^gamma
        max_loss = loss_detached.max()
        normalized_loss = loss_detached / (max_loss + 1e-8)
        weights = torch.pow(normalized_loss, config.focal_gamma)
    
    else:
        raise ValueError(f"Unknown weighting function: {config.function}")
    
    # Apply clamping
    if config.percentile_clamp:
        # Dynamic bounds based on percentiles
        p_low = torch.quantile(weights, config.percentile_low)
        p_high = torch.quantile(weights, config.percentile_high)
        weights = weights.clamp(p_low, p_high)
    else:
        # Fixed bounds
        weights = weights.clamp(config.clamp_min, config.clamp_max)
    
    # Apply schedule (interpolate between uniform and computed weights)
    strength = compute_schedule_strength(step, total_steps, config, val_loss)
    # weights = 1.0 + (weights - 1.0) * strength
    # This interpolates: at strength=0, weights=1 (uniform); at strength=1, weights=computed
    weights = 1.0 + (weights - 1.0) * strength
    
    # Normalize weights to have mean 1.0 (preserve expected gradient magnitude)
    weights = weights / (weights.mean() + 1e-8)
    
    # Compute statistics for logging
    stats = {}
    if config.log_weights:
        stats = {
            'weight_mean': weights.mean().item(),
            'weight_std': weights.std().item(),
            'weight_min': weights.min().item(),
            'weight_max': weights.max().item(),
            'weight_gt_1.5': (weights > 1.5).float().mean().item() * 100,  # percentage
            'weight_lt_0.5': (weights < 0.5).float().mean().item() * 100,  # percentage
            'schedule_strength': strength,
        }
    
    return weights, stats

# -----------------------------------------------------------------------------
# Random seed utilities for reproducibility

def set_seed(seed: int, rank: int = 0, deterministic: bool = True):
    """
    Set random seeds for reproducibility across distributed training.
    
    Args:
        seed: Base random seed
        rank: DDP rank (used to offset data sampling seeds)
        deterministic: If True, enable deterministic algorithms (may impact performance)
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy seed (offset by rank for different data sampling per GPU)
    np.random.seed(seed + rank)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Configure deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass  # Some operations may not have deterministic implementations
    else:
        # For maximum performance, allow non-deterministic algorithms
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1): # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True, return_per_token_loss=False):
        """
        Forward pass through the GPT model.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices of shape (batch_size, seq_len)
            return_logits: Whether to return logits
            return_per_token_loss: Whether to return per-token loss (for token weighting)
        
        Returns:
            logits: Output logits (if return_logits=True)
            loss: Scalar loss (mean) or per-token loss tensor (if return_per_token_loss=True)
        """
        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            
            if return_per_token_loss:
                # Return per-token loss for token importance weighting
                # Shape: (batch_size * seq_len,)
                per_token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1), 
                    ignore_index=-1,
                    reduction='none'
                )
                loss = per_token_loss  # Return unreduced loss
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

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
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, seed=None):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.seed = seed

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

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

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
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 5100 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    # reproducibility hyperparams
    seed : int = None # random seed for reproducibility (None = no seeding, uses system entropy)
    deterministic : bool = False # if True, use deterministic algorithms (slower but reproducible)
    # token weighting hyperparams (for ablations)
    tw_enabled : bool = False  # enable token importance weighting
    tw_function : str = "linear"  # weighting function: linear, sqrt, log, focal
    tw_clamp_min : float = 0.5  # minimum weight clamp
    tw_clamp_max : float = 2.0  # maximum weight clamp
    tw_schedule : str = "constant"  # schedule: constant, warmup, anneal, cyclical, adaptive
    tw_warmup_steps : int = 1000  # warmup steps for warmup schedule
    tw_anneal_final : float = 0.3  # final strength for anneal schedule
    tw_cyclical_period : int = 500  # period for cyclical schedule
    tw_focal_gamma : float = 2.0  # gamma for focal loss style weighting
    tw_percentile_clamp : bool = False  # use percentile-based clamping
    tw_percentile_low : float = 0.05  # lower percentile for clamping
    tw_percentile_high : float = 0.95  # upper percentile for clamping
    tw_log_weights : bool = True  # log weight statistics
    tw_log_every : int = 50  # log weight stats every N steps

def parse_args():
    """Parse command line arguments and return Hyperparameters and TokenWeightingConfig."""
    parser = argparse.ArgumentParser(description='NanoGPT Training with Token Importance Weighting')
    
    # Data hyperparams
    parser.add_argument('--input_bin', type=str, default='../fineweb10B/fineweb_train_*.bin')
    parser.add_argument('--input_val_bin', type=str, default='../fineweb10B/fineweb_val_*.bin')
    
    # Optimization hyperparams
    parser.add_argument('--batch_size', type=int, default=8*64)
    parser.add_argument('--device_batch_size', type=int, default=64)
    parser.add_argument('--sequence_length', type=int, default=1024)
    parser.add_argument('--num_iterations', type=int, default=5100)
    parser.add_argument('--learning_rate', type=float, default=0.0036)
    parser.add_argument('--warmup_iters', type=int, default=0)
    parser.add_argument('--warmdown_iters', type=int, default=1450)
    parser.add_argument('--weight_decay', type=float, default=0)
    
    # Evaluation and logging hyperparams
    parser.add_argument('--val_loss_every', type=int, default=125)
    parser.add_argument('--val_tokens', type=int, default=10485760)
    parser.add_argument('--save_every', type=int, default=0)
    
    # Reproducibility hyperparams
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    
    # Token weighting hyperparams (for ablations)
    parser.add_argument('--tw_enabled', action='store_true', help='Enable token importance weighting')
    parser.add_argument('--tw_function', type=str, default='linear', 
                        choices=['linear', 'sqrt', 'log', 'focal'],
                        help='Weighting function')
    parser.add_argument('--tw_clamp_min', type=float, default=0.5, help='Minimum weight clamp')
    parser.add_argument('--tw_clamp_max', type=float, default=2.0, help='Maximum weight clamp')
    parser.add_argument('--tw_schedule', type=str, default='constant',
                        choices=['constant', 'warmup', 'anneal', 'cyclical', 'adaptive'],
                        help='Weighting schedule')
    parser.add_argument('--tw_warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--tw_anneal_final', type=float, default=0.3, help='Final strength for anneal')
    parser.add_argument('--tw_cyclical_period', type=int, default=500, help='Period for cyclical')
    parser.add_argument('--tw_focal_gamma', type=float, default=2.0, help='Gamma for focal loss')
    parser.add_argument('--tw_percentile_clamp', action='store_true', help='Use percentile clamping')
    parser.add_argument('--tw_percentile_low', type=float, default=0.05, help='Lower percentile')
    parser.add_argument('--tw_percentile_high', type=float, default=0.95, help='Upper percentile')
    parser.add_argument('--tw_log_weights', action='store_true', help='Log weight statistics')
    parser.add_argument('--tw_log_every', type=int, default=50, help='Log weights every N steps')
    
    parsed_args = parser.parse_args()
    
    # Create Hyperparameters
    args = Hyperparameters(
        input_bin=parsed_args.input_bin,
        input_val_bin=parsed_args.input_val_bin,
        batch_size=parsed_args.batch_size,
        device_batch_size=parsed_args.device_batch_size,
        sequence_length=parsed_args.sequence_length,
        num_iterations=parsed_args.num_iterations,
        learning_rate=parsed_args.learning_rate,
        warmup_iters=parsed_args.warmup_iters,
        warmdown_iters=parsed_args.warmdown_iters,
        weight_decay=parsed_args.weight_decay,
        val_loss_every=parsed_args.val_loss_every,
        val_tokens=parsed_args.val_tokens,
        save_every=parsed_args.save_every,
        seed=parsed_args.seed,
        deterministic=parsed_args.deterministic,
        tw_enabled=parsed_args.tw_enabled,
        tw_function=parsed_args.tw_function,
        tw_clamp_min=parsed_args.tw_clamp_min,
        tw_clamp_max=parsed_args.tw_clamp_max,
        tw_schedule=parsed_args.tw_schedule,
        tw_warmup_steps=parsed_args.tw_warmup_steps,
        tw_anneal_final=parsed_args.tw_anneal_final,
        tw_cyclical_period=parsed_args.tw_cyclical_period,
        tw_focal_gamma=parsed_args.tw_focal_gamma,
        tw_percentile_clamp=parsed_args.tw_percentile_clamp,
        tw_percentile_low=parsed_args.tw_percentile_low,
        tw_percentile_high=parsed_args.tw_percentile_high,
        tw_log_weights=parsed_args.tw_log_weights,
        tw_log_every=parsed_args.tw_log_every,
    )
    
    # Create TokenWeightingConfig
    tw_config = TokenWeightingConfig(
        enabled=parsed_args.tw_enabled,
        function=parsed_args.tw_function,
        clamp_min=parsed_args.tw_clamp_min,
        clamp_max=parsed_args.tw_clamp_max,
        schedule=parsed_args.tw_schedule,
        warmup_steps=parsed_args.tw_warmup_steps,
        anneal_final_strength=parsed_args.tw_anneal_final,
        cyclical_period=parsed_args.tw_cyclical_period,
        focal_gamma=parsed_args.tw_focal_gamma,
        percentile_clamp=parsed_args.tw_percentile_clamp,
        percentile_low=parsed_args.tw_percentile_low,
        percentile_high=parsed_args.tw_percentile_high,
        log_weights=parsed_args.tw_log_weights,
        log_every=parsed_args.tw_log_every,
    )
    
    return args, tw_config

# Parse arguments
args, tw_config = parse_args()

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

# Set random seeds for reproducibility (must be done after DDP init but before model creation)
if args.seed is not None:
    set_seed(args.seed, rank=ddp_rank, deterministic=args.deterministic)
    if master_process:
        print(f"Random seed set to {args.seed} (deterministic={args.deterministic})")

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size, seed=args.seed)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size, seed=args.seed)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
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
        # log the random seed if set
        if args.seed is not None:
            f.write(f"Random seed: {args.seed} (deterministic={args.deterministic})\n")
            f.write('='*100 + '\n')
        # log token weighting configuration
        if tw_config.enabled:
            f.write(f"Token Weighting Configuration:\n")
            f.write(f"  function: {tw_config.function}\n")
            f.write(f"  clamp: [{tw_config.clamp_min}, {tw_config.clamp_max}]\n")
            f.write(f"  schedule: {tw_config.schedule}\n")
            if tw_config.schedule == 'warmup':
                f.write(f"  warmup_steps: {tw_config.warmup_steps}\n")
            elif tw_config.schedule == 'anneal':
                f.write(f"  anneal_final_strength: {tw_config.anneal_final_strength}\n")
            elif tw_config.schedule == 'cyclical':
                f.write(f"  cyclical_period: {tw_config.cyclical_period}\n")
            if tw_config.function == 'focal':
                f.write(f"  focal_gamma: {tw_config.focal_gamma}\n")
            if tw_config.percentile_clamp:
                f.write(f"  percentile_clamp: [{tw_config.percentile_low}, {tw_config.percentile_high}]\n")
            f.write('='*100 + '\n')
        else:
            f.write("Token Weighting: DISABLED (baseline run)\n")
            f.write('='*100 + '\n')
    
    # Also log token weighting config to console
    if tw_config.enabled:
        print(f"Token Weighting: ENABLED")
        print(f"  function={tw_config.function}, clamp=[{tw_config.clamp_min}, {tw_config.clamp_max}], schedule={tw_config.schedule}")
    else:
        print(f"Token Weighting: DISABLED (baseline run)")

# Track validation loss for adaptive schedule
current_val_loss = None

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
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
        current_val_loss = val_loss.item()  # Store for adaptive schedule
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock and keep timing accurate, but skip checkpoint writes to save disk
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        if master_process:
            print("Checkpoint saving disabled; skipping state serialization.")
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
    
    # Accumulate weight statistics across accumulation steps
    accumulated_weight_stats = {}
    
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            if tw_config.enabled:
                # Get per-token loss for token weighting
                _, per_token_loss = model(x, y, return_logits=False, return_per_token_loss=True)
                
                # Compute token importance weights
                weights, weight_stats = compute_token_weights(
                    per_token_loss,
                    tw_config,
                    step,
                    args.num_iterations,
                    current_val_loss
                )
                
                # Apply weights and compute mean loss
                weighted_loss = per_token_loss * weights
                loss = weighted_loss.mean()
                train_loss = per_token_loss.mean().detach()  # Log unweighted loss for fair comparison
                
                # Accumulate weight stats
                if weight_stats and (step % tw_config.log_every == 0):
                    for k, v in weight_stats.items():
                        if k not in accumulated_weight_stats:
                            accumulated_weight_stats[k] = []
                        accumulated_weight_stats[k].append(v)
            else:
                # Standard training without token weighting
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
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        
        # Build log message
        log_msg = f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
        
        # Add weight statistics if enabled and this is a logging step
        if tw_config.enabled and tw_config.log_weights and accumulated_weight_stats and (step % tw_config.log_every == 0):
            # Average the accumulated stats
            avg_weight_stats = {k: sum(v)/len(v) for k, v in accumulated_weight_stats.items()}
            weight_log = f" w_mean:{avg_weight_stats.get('weight_mean', 1.0):.3f} w_std:{avg_weight_stats.get('weight_std', 0.0):.3f} w_min:{avg_weight_stats.get('weight_min', 1.0):.3f} w_max:{avg_weight_stats.get('weight_max', 1.0):.3f}"
            log_msg += weight_log
        
        print(log_msg)
        with open(logfile, "a") as f:
            f.write(log_msg + "\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")