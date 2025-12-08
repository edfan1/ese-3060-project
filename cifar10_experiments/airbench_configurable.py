# Modified airbench94.py with configurable whitening layer parameters
# Based on https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py
#
# Modified to allow all hyperparameters to be passed via command line or imported as a module
# Purpose is to run our whitening experiments experiments

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
import json
import argparse
from math import ceil
from datetime import datetime
import subprocess

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

def get_git_info():
    """Get current git commit hash and status."""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        
        # Check for uncommitted changes
        try:
            subprocess.check_output(
                ['git', 'diff-index', '--quiet', 'HEAD', '--'],
                stderr=subprocess.DEVNULL
            )
            dirty = False
        except subprocess.CalledProcessError:
            dirty = True
        
        return {
            'commit_hash': commit_hash,
            'dirty': dirty,  # True if uncommitted changes exist
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            'commit_hash': 'unknown',
            'dirty': False,
        }

# Default hyperparameters (can be overridden)
def get_default_hyp():
    return {
        'opt': {
            'train_epochs': 9.9,
            'batch_size': 1024,
            'lr': 11.5,
            'momentum': 0.85,
            'weight_decay': 0.0153,
            'bias_scaler': 64.0,
            'label_smoothing': 0.2,
            'whiten_bias_epochs': 3,
        },
        'aug': {
            'flip': True,
            'translate': 2,
        },
        'net': {
            'widths': {
                'block1': 64,
                'block2': 256,
                'block3': 256,
            },
            'batchnorm_momentum': 0.6,
            'scaling_factor': 1/9,
            'tta_level': 2,
        },
        'alpha_schedule': {
            'coef' : 0.95,
            'coef_exp' : 5,
            'step_exp' : 3,
        },
        # Whitening layer hyperparameters
        'whiten': {
            'kernel_size': 2,           # Q1: spatial extent of whitening patches
            'width_multiplier': 2,      # Q2: 1=positive only, 2=+/- eigenvectors
            'eps': 5e-4,                # Q3: regularization for small eigenvalues
            'trainable_after_epoch': None,  # Q4: epoch to unfreeze weights (None=never)
            'trainable_lr_mult': 0.01,  # Q4: LR multiplier when trainable
        },
    }

# Global hyp dict - will be set by run_experiment()
hyp = get_default_hyp()

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, gpu=0):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device(gpu))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

def make_net():
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    
    # MODIFIED: Use configurable whitening parameters
    whiten_kernel_size = hyp['whiten']['kernel_size']
    whiten_width_mult = hyp['whiten']['width_multiplier']
    whiten_width = whiten_width_mult * 3 * whiten_kernel_size**2
    
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    """Initialize whitening conv with configurable width multiplier."""
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    
    width_mult = hyp['whiten']['width_multiplier']
    if width_mult == 1:
        # Only positive eigenvectors
        layer.weight.data[:] = eigenvectors_scaled
    elif width_mult == 2:
        # Positive and negative eigenvectors (default)
        layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
    elif width_mult == 3:
        # Positive, negative, and additional learned channels
        extra = torch.randn_like(eigenvectors_scaled) * 0.01
        layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled, extra))
    else:
        raise ValueError(f"Unsupported width_multiplier: {width_mult}")

############################################
#                Lookahead                 #
############################################

class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']

def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def train_one_run(run, seed=42, verbose=True):
    """Train a single run and return detailed results."""
    
    # Set and log random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'])
    
    if run == 'warmup':
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    
    total_train_steps = ceil(len(train_loader) * epochs)

    model = make_net()
    current_steps = 0

    # Setup optimizer with potential for whitening layer fine-tuning
    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.23)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (step - warmup_steps) / warmdown_steps
            return 1.0 * (1 - frac) + 0.07 * frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    alpha_schedule = hyp['alpha_schedule']['coef']**hyp['alpha_schedule']['coef_exp'] * (torch.arange(total_train_steps+1) / total_train_steps)**hyp['alpha_schedule']['step_exp']
    lookahead_state = LookaheadState(model)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize whitening layer
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images, eps=hyp['whiten']['eps'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    # Track per-epoch results
    epoch_results = []
    
    for epoch in range(ceil(epochs)):
        # Handle whitening bias training
        model[0].bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])
        
        # MODIFIED: Handle whitening weight trainability (Q4)
        trainable_after = hyp['whiten']['trainable_after_epoch']
        if trainable_after is not None and epoch >= trainable_after:
            if not model[0].weight.requires_grad:
                # First time unfreezing - need to add to optimizer
                model[0].weight.requires_grad = True
                whiten_lr = lr * hyp['whiten']['trainable_lr_mult']
                optimizer.add_param_group({
                    'params': [model[0].weight],
                    'lr': whiten_lr,
                    'weight_decay': wd / whiten_lr
                })

        ####################
        #     Training     #
        ####################

        starter.record()
        model.train()
        
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_steps += 1

            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=alpha_schedule[current_steps].item())

            if current_steps >= total_train_steps:
                if lookahead_state is not None:
                    lookahead_state.update(model, decay=1.0)
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        ####################
        #    Evaluation    #
        ####################

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size
        val_acc = evaluate(model, test_loader, tta_level=0)
        
        epoch_results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'time_so_far': total_time_seconds,
        })
        
        if verbose:
            print_training_details(locals(), is_final_entry=False)
            run = None

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    if verbose:
        epoch = 'eval'
        print_training_details(locals(), is_final_entry=True)

    return {
        'seed': seed,  # Add this
        'tta_val_acc': tta_val_acc,
        'total_time_seconds': total_time_seconds,
        'epoch_results': epoch_results,
    }


def run_experiment(config=None, num_runs=25, experiment_name=None, verbose=True, save_results=True, seeds=[]):
    """
    Run a complete experiment with the given configuration.
    
    Args:
        config: dict of hyperparameter overrides (will be merged with defaults)
        num_runs: number of training runs
        experiment_name: name for this experiment (for logging)
        verbose: whether to print training progress
        save_results: whether to save results to disk
    
    Returns:
        dict with experiment results and statistics
    """
    global hyp
    
    # Start with default hyperparameters
    hyp = get_default_hyp()
    
    # Apply config overrides
    if config:
        for key, value in config.items():
            if isinstance(value, dict) and key in hyp:
                hyp[key].update(value)
            else:
                hyp[key] = value
    
    # Generate experiment ID
    experiment_id = str(uuid.uuid4())[:8]
    if experiment_name:
        experiment_id = f"{experiment_name}_{experiment_id}"
    
    timestamp = datetime.now().isoformat()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_id}")
        print(f"Timestamp: {timestamp}")
        print(f"Config overrides: {config}")
        print(f"Number of runs: {num_runs}")
        print(f"{'='*60}\n")
        print_columns(logging_columns_list, is_head=True)
    
    # Run experiments
    all_results = []
    accuracies = []
    times = []
    if len(seeds) < num_runs:
        print("Not enough seeds provided, generating own list")
        seeds = [i for i in range(num_runs)]

    for run_idx in range(num_runs):
        result = train_one_run(run_idx, seed=seeds[run_idx], verbose=verbose)
        all_results.append(result)
        accuracies.append(result['tta_val_acc'])
        times.append(result['total_time_seconds'])
    
    # Compute statistics
    acc_tensor = torch.tensor(accuracies)
    time_tensor = torch.tensor(times)
    
    stats = {
        'accuracy': {
            'mean': acc_tensor.mean().item(),
            'std': acc_tensor.std().item(),
            'min': acc_tensor.min().item(),
            'max': acc_tensor.max().item(),
            'median': acc_tensor.median().item(),
        },
        'time': {
            'mean': time_tensor.mean().item(),
            'std': time_tensor.std().item(),
            'min': time_tensor.min().item(),
            'max': time_tensor.max().item(),
            'median': time_tensor.median().item(),
        }
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS: {experiment_id}")
        print(f"Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
        print(f"Time:     {stats['time']['mean']:.4f} ± {stats['time']['std']:.4f} seconds")
        print(f"{'='*60}\n")
    
    # Get git info and environment metadata
    git_info = get_git_info()
    import socket
    hostname = socket.gethostname()
    
    # Compile full results with ALL metadata
    experiment_results = {
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'config': config,
        'full_hyp': hyp,
        'num_runs': num_runs,
        'stats': stats,
        'all_accuracies': accuracies,
        'all_times': times,
        'all_results': all_results,
        'hardware': {
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
            'gpu_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
            'hostname': hostname,
            'runpod_instance': os.environ.get('RUNPOD_POD_ID', 'N/A'),  # Add this
        },
        'git': git_info,  # Add this
        'seeds': [r['seed'] for r in all_results],  # Add this
    }

    # Save results
    if save_results:
        log_dir = os.path.join('experiment_logs', experiment_id)
        os.makedirs(log_dir, exist_ok=True)
        
        # Save as JSON (human-readable)
        json_path = os.path.join(log_dir, 'results.json')
        with open(json_path, 'w') as f:
            # Convert non-serializable items
            json_safe = experiment_results.copy()
            json_safe['all_results'] = 'see results.pt'
            json.dump(json_safe, f, indent=2, default=str)
        
        # Save as PyTorch (full data)
        pt_path = os.path.join(log_dir, 'results.pt')
        torch.save(experiment_results, pt_path)
        
        if verbose:
            print(f"Results saved to: {log_dir}")
    
    return experiment_results


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CIFAR-10 training experiments')
    parser.add_argument('--num_runs', type=int, default=25, help='Number of training runs')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name for this experiment')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    # Whitening layer parameters (Q1-Q4)
    parser.add_argument('--whiten_kernel_size', type=int, default=None, help='Whitening kernel size (Q1)')
    parser.add_argument('--whiten_width_mult', type=int, default=None, help='Whitening width multiplier (Q2)')
    parser.add_argument('--whiten_eps', type=float, default=None, help='Whitening epsilon (Q3)')
    parser.add_argument('--whiten_trainable_after', type=int, default=None, help='Epoch to unfreeze whitening (Q4)')
    parser.add_argument('--whiten_trainable_lr_mult', type=float, default=None, help='LR multiplier for trainable whitening (Q4)')
    
    args = parser.parse_args()
    
    # Build config from command-line arguments
    config = {}
    whiten_config = {}
    
    if args.whiten_kernel_size is not None:
        whiten_config['kernel_size'] = args.whiten_kernel_size
    if args.whiten_width_mult is not None:
        whiten_config['width_multiplier'] = args.whiten_width_mult
    if args.whiten_eps is not None:
        whiten_config['eps'] = args.whiten_eps
    if args.whiten_trainable_after is not None:
        whiten_config['trainable_after_epoch'] = args.whiten_trainable_after
    if args.whiten_trainable_lr_mult is not None:
        whiten_config['trainable_lr_mult'] = args.whiten_trainable_lr_mult
    
    if whiten_config:
        config['whiten'] = whiten_config
    
    # Run experiment
    run_experiment(
        config=config if config else None,
        num_runs=args.num_runs,
        experiment_name=args.experiment_name,
        verbose=not args.quiet,
    )