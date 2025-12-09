# Modified NanoGPT with Auxiliary Prediction Heads at Intermediate Layers
# 
# Hypothesis: Adding lightweight auxiliary prediction heads at layers 4 and 8 
# (with small loss weights like 0.1) provides additional gradient signal to 
# early/middle layers, accelerating their learning.
#
# Based on: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt
# ====================================================================================================
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import List, Optional, Dict
import random

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
    where S' is diagonal with S_{ii}' sim Uniform(0.5, 1.5), which turns out not to hurt model
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
# ADD: Auxiliary Head for Deep Supervision
# -----------------------------------------------------------------------------

class AuxiliaryHead(nn.Module):
    """
    Lightweight auxiliary prediction head for deep supervision.
    
    Design choices:
    - Single linear layer (no bias) to minimize overhead
    - Zero initialization to start with no aux contribution
    - Tuneable feat: Can share weights with main head (controlled by config)
    """
    def __init__(self, n_embd, vocab_size, zero_init=True):
        super().__init__()
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if zero_init:
            # Zero init so auxiliary heads start with minimal contribution
            # This lets the model gradually learn to use them
            self.head.weight.data.zero_()
    
    def forward(self, x):
        # RMS norm before projection (same as main head)
        x = F.rms_norm(x, (x.size(-1),))
        return self.head(x)

# -----------------------------------------------------------------------------
# The main GPT-2 model with Auxiliary Heads
# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # head dim 128 suggested by @Grad62304977
    n_embd: int = 768
    # Auxiliary head configuration
    aux_head_layers: List[int] = field(default_factory=lambda: [])  # Layers to add aux heads (e.g., [4, 8])
    aux_loss_weight: float = 0.1  # Weight for auxiliary losses
    aux_head_zero_init: bool = True  # Whether to zero-init aux heads
    aux_loss_schedule: str = 'constant'  # 'constant', 'linear_decay', 'cosine_decay'

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
        
        # Create auxiliary heads at specified layers
        self.aux_heads = nn.ModuleDict()
        for layer_idx in config.aux_head_layers:
            if 0 <= layer_idx < config.n_layer:
                self.aux_heads[str(layer_idx)] = AuxiliaryHead(
                    config.n_embd, 
                    config.vocab_size,
                    zero_init=config.aux_head_zero_init
                )

    def forward(self, idx, targets=None, return_logits=True, current_step=None, total_steps=None):
        """
        Forward pass with optional auxiliary losses.
        
        Args:
            idx: Input token indices
            targets: Target token indices for loss computation
            return_logits: Whether to return logits
            current_step: Current training step (for aux loss scheduling)
            total_steps: Total training steps (for aux loss scheduling)
        
        Returns:
            logits: Main prediction logits (or None if return_logits=False)
            loss: Total loss (main + weighted auxiliary losses)
            aux_info: Dict containing auxiliary losses and other info (for logging)
        """
        aux_info = {}
        
        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        # Store intermediate hidden states for auxiliary heads
        intermediate_states = {}
        
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x)
            # Store state if we have an aux head at this layer
            if str(layer_idx) in self.aux_heads:
                intermediate_states[layer_idx] = x
        
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # Main loss computation
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Compute auxiliary losses
            aux_losses = {}
            total_aux_loss = 0.0
            
            # Compute aux loss weight based on schedule
            aux_weight = self._get_aux_weight(current_step, total_steps)
            
            for layer_idx, hidden_state in intermediate_states.items():
                aux_head = self.aux_heads[str(layer_idx)]
                aux_logits = aux_head(hidden_state)
                aux_logits = aux_logits.float()
                aux_loss = F.cross_entropy(aux_logits.view(-1, aux_logits.size(-1)), targets.view(-1), ignore_index=-1)
                aux_losses[f'aux_loss_layer_{layer_idx}'] = aux_loss.detach().item()
                total_aux_loss = total_aux_loss + aux_loss
            
            # Combine losses
            if len(aux_losses) > 0:
                loss = main_loss + aux_weight * total_aux_loss
            else:
                loss = main_loss
            
            # Store info for logging
            aux_info['main_loss'] = main_loss.detach().item()
            aux_info['aux_losses'] = aux_losses
            aux_info['aux_weight'] = aux_weight
            aux_info['total_aux_loss'] = total_aux_loss.detach().item() if isinstance(total_aux_loss, torch.Tensor) else 0.0
            
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss, aux_info
    
    def _get_aux_weight(self, current_step, total_steps):
        """
        Compute auxiliary loss weight based on schedule.
        
        Schedules:
        - 'constant': Always use config.aux_loss_weight
        - 'linear_decay': Linearly decay from aux_loss_weight to 0
        - 'cosine_decay': Cosine decay from aux_loss_weight to 0
        - 'warmup_decay': Warmup then decay
        """
        base_weight = self.config.aux_loss_weight
        
        if current_step is None or total_steps is None:
            return base_weight
        
        progress = current_step / total_steps
        
        if self.config.aux_loss_schedule == 'constant':
            return base_weight
        elif self.config.aux_loss_schedule == 'linear_decay':
            return base_weight * (1.0 - progress)
        elif self.config.aux_loss_schedule == 'cosine_decay':
            return base_weight * 0.5 * (1.0 + np.cos(np.pi * progress))
        elif self.config.aux_loss_schedule == 'warmup_decay':
            # Warmup for first 10%, then decay
            if progress < 0.1:
                return base_weight * (progress / 0.1)
            else:
                decay_progress = (progress - 0.1) / 0.9
                return base_weight * (1.0 - decay_progress)
        else:
            return base_weight

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
        print("---> HINT: Dataset encoding changed recent, re-run data prepro or refer again to README")
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
    # use parent dir so that we can leave cached_finweb10B in main dir
    input_bin : str = '../fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = '../fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
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
    # Auxiliary head hyperparams
    aux_head_layers : str = ''  # Comma-separated list of layers, e.g., '4,8'
    aux_loss_weight : float = 0.1  # Weight for auxiliary losses
    aux_loss_schedule : str = 'constant'  # 'constant', 'linear_decay', 'cosine_decay', 'warmup_decay'

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT with optional auxiliary heads')
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
    # Auxiliary head hyperparams
    parser.add_argument('--aux_head_layers', type=str, default='',
                        help='Comma-separated list of layers for aux heads, e.g., "4,8"')
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                        help='Weight for auxiliary losses')
    parser.add_argument('--aux_loss_schedule', type=str, default='constant',
                        choices=['constant', 'linear_decay', 'cosine_decay', 'warmup_decay'],
                        help='Schedule for auxiliary loss weight')
    parser.add_argument('--aux_head_zero_init', action='store_true', default=True,
                        help='Zero-initialize auxiliary heads')
    parser.add_argument('--no_aux_head_zero_init', action='store_false', dest='aux_head_zero_init',
                        help='Do not zero-initialize auxiliary heads')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed to use (overrides SEED env var); defaults to 42 if unset')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse auxiliary head layers
    aux_head_layers = []
    if args.aux_head_layers:
        aux_head_layers = [int(x.strip()) for x in args.aux_head_layers.split(',')]
    
    # ============ ADD SEED SETTING ============
    # Prefer explicit CLI seed, then SEED env var, then fallback default
    env_seed = os.environ.get('SEED')
    seed = args.seed if args.seed is not None else int(env_seed) if env_seed is not None else 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ==========================================
    
    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0)
    
    # ============ ADD GIT COMMIT HASH ============
    git_commit = "unknown"
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True, 
                                timeout=5)
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except:
        pass
    # =============================================
    
    if master_process:
        print(f"Auxiliary head configuration:")
        print(f"  Layers: {aux_head_layers if aux_head_layers else 'None (baseline)'}")
        print(f"  Loss weight: {args.aux_loss_weight}")
        print(f"  Loss schedule: {args.aux_loss_schedule}")
        # ============ ADD SEED + GIT LOGGING ============
        print(f"  Random seed: {seed}")
        print(f"  Git commit: {git_commit}")
        # ================================================

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
    x, y = train_loader.next_batch()

    # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
    # this originates from Karpathy's experiments.
    num_vocab = 50304
    
    # Create model config with auxiliary heads
    model_config = GPTConfig(
        vocab_size=num_vocab, 
        n_layer=12, 
        n_head=6, 
        n_embd=768,
        aux_head_layers=aux_head_layers,
        aux_loss_weight=args.aux_loss_weight,
        aux_head_zero_init=args.aux_head_zero_init,
        aux_loss_schedule=args.aux_loss_schedule,
    )
    
    model = GPT(model_config)
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init the optimizer(s)
    # Main head parameters (including tied embedding)
    main_head_params = list(raw_model.lm_head.parameters())
    
    # Auxiliary head parameters
    aux_head_params = []
    for aux_head in raw_model.aux_heads.values():
        aux_head_params.extend(list(aux_head.parameters()))
    
    # Combine main and aux head params for AdamW
    head_params = main_head_params + aux_head_params
    
    optimizer1 = torch.optim.AdamW(head_params, lr=args.learning_rate, betas=(0.9, 0.95),
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
        
        # ============ ENHANCED LOGGING HEADER ============
        # create the log file
        with open(logfile, "w") as f:
            # begin the log by printing this file (the Python code)
            f.write('='*100 + '\n')
            f.write(code)
            f.write('='*100 + '\n')
            
            # ============ ADD EXPERIMENT METADATA ============
            f.write('\nEXPERIMENT METADATA\n')
            f.write('='*100 + '\n')
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Git commit: {git_commit}\n")
            f.write(f"Random seed: {seed}\n")
            f.write('\n')
            
            # ============ ADD HYPERPARAMETERS ============
            f.write('HYPERPARAMETERS\n')
            f.write('-'*100 + '\n')
            f.write(f"Batch size (global): {args.batch_size}\n")
            f.write(f"Batch size (per device): {args.device_batch_size}\n")
            f.write(f"Sequence length: {args.sequence_length}\n")
            f.write(f"Number of iterations: {args.num_iterations}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Warmup iterations: {args.warmup_iters}\n")
            f.write(f"Warmdown iterations: {args.warmdown_iters}\n")
            f.write(f"Weight decay: {args.weight_decay}\n")
            f.write(f"Validation every: {args.val_loss_every}\n")
            f.write(f"Validation tokens: {args.val_tokens}\n")
            f.write('\n')
            
            # ============ ADD GPU/SYSTEM INFO ============
            f.write('SYSTEM INFORMATION\n')
            f.write('-'*100 + '\n')
            f.write(f"PyTorch version: {torch.version.__version__}\n")
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"Number of GPUs: {ddp_world_size}\n")
            f.write(f"GPU type: {torch.cuda.get_device_name(0)}\n")
            # ============ ADD RUNPOD INSTANCE INFO ============
            runpod_id = os.environ.get('RUNPOD_POD_ID', 'N/A')
            runpod_type = os.environ.get('RUNPOD_GPU_TYPE', 'N/A')
            f.write(f"RunPod instance ID: {runpod_id}\n")
            f.write(f"RunPod GPU type: {runpod_type}\n")
            f.write('\n')
            # ==================================================
            
            # log information about the hardware/software environment this is running on
            # and print the full `nvidia-smi` to file
            f.write('NVIDIA-SMI OUTPUT\n')
            f.write('-'*100 + '\n')
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'{result.stdout}\n')
            f.write('='*100 + '\n')
            
            # Log auxiliary head configuration
            f.write(f"\nAUXILIARY HEAD CONFIGURATION\n")
            f.write('-'*100 + '\n')
            f.write(f"Layers: {aux_head_layers if aux_head_layers else 'None (baseline)'}\n")
            f.write(f"Loss weight: {args.aux_loss_weight}\n")
            f.write(f"Loss schedule: {args.aux_loss_schedule}\n")
            f.write(f"Zero init: {args.aux_head_zero_init}\n")
            f.write('='*100 + '\n')
        # =================================================
        
        # Create separate log for auxiliary losses
        aux_logfile = 'logs/%s_aux.txt' % run_id
        with open(aux_logfile, "w") as f:
            f.write("step,main_loss,total_aux_loss,aux_weight")
            for layer in aux_head_layers:
                f.write(f",aux_loss_layer_{layer}")
            f.write("\n")

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
                    _, loss, _ = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            if master_process:
                print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                with open(logfile, "a") as f:
                    f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
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
                _, loss, aux_info = model(x, y, return_logits=False, 
                                          current_step=step, 
                                          total_steps=args.num_iterations)
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
            
            # Log main training info
            main_loss = aux_info.get('main_loss', train_loss.item())
            aux_loss_str = ""
            if aux_head_layers:
                aux_loss_str = f" main_loss:{main_loss:.4f} aux_weight:{aux_info.get('aux_weight', 0):.4f}"
                for layer in aux_head_layers:
                    aux_loss_str += f" aux_{layer}:{aux_info.get('aux_losses', {}).get(f'aux_loss_layer_{layer}', 0):.4f}"
            
            print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f}{aux_loss_str} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
            with open(logfile, "a") as f:
                f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f}{aux_loss_str} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
            
            # Log detailed auxiliary info to separate file
            if aux_head_layers:
                with open(aux_logfile, "a") as f:
                    f.write(f"{step+1},{main_loss:.6f},{aux_info.get('total_aux_loss', 0):.6f},{aux_info.get('aux_weight', 0):.6f}")
                    for layer in aux_head_layers:
                        f.write(f",{aux_info.get('aux_losses', {}).get(f'aux_loss_layer_{layer}', 0):.6f}")
                    f.write("\n")

    if master_process:
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
        
        # Add final summary to log
        with open(logfile, "a") as f:
            f.write("\n" + "="*100 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("="*100 + "\n")
            f.write(f"Total training time: {training_time_ms/1000:.1f}s\n")
            f.write(f"Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB\n")
            f.write(f"Completed iterations: {args.num_iterations}\n")
            f.write(f"Final validation loss: (see last val_loss entry above)\n")
            f.write("="*100 + "\n")
    # ==================================================