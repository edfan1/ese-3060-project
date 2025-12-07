#!/usr/bin/env python3
"""
This script defines and runs all experiments for testing our whitening layer modification hypothesis. 

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --phase baseline   # Run only baseline
    python run_experiments.py --phase Q1         # Run only kernel size ablation
"""

import os
import sys
import json
import argparse
from datetime import datetime
from itertools import product

# Import our configurable airbench
from airbench_configurable import run_experiment, get_default_hyp

#############################################
#         Experiment Definitions            #
#############################################

# Number of runs per experiment setting
# Adjust based on your compute budget
RUNS_BASELINE = 100      # More runs for baseline to establish reliable reference
RUNS_ABLATION = 50       # Fewer runs for ablations during exploration
RUNS_FINAL = 100         # More runs for final best configuration

def get_all_experiments():
    """
    Define all experiments to run.
    
    Returns a dict mapping experiment names to their configurations.
    """
    experiments = {}
    
    # Phase 0: baseline
    experiments['baseline'] = {
        'config': None,  # No changes from default
        'num_runs': RUNS_BASELINE,
        'description': 'Baseline: unmodified airbench94',
        'phase': 'baseline',
    }
    
    # Phase 1: kernel size ablation (Q1)
    # Default is 2x2, test 1x1, 3x3, 4x4
    
    experiments['Q1_kernel_1x1'] = {
        'config': {'whiten': {'kernel_size': 1}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q1: Whitening kernel size 1x1 (per-pixel)',
        'phase': 'Q1',
    }
    
    experiments['Q1_kernel_3x3'] = {
        'config': {'whiten': {'kernel_size': 3}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q1: Whitening kernel size 3x3',
        'phase': 'Q1',
    }
    
    experiments['Q1_kernel_4x4'] = {
        'config': {'whiten': {'kernel_size': 4}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q1: Whitening kernel size 4x4',
        'phase': 'Q1',
    }
    
    # Phase 2: width multiplier ablation (Q2)
    # Default is 2 (+-eigenvectors), test 1 and 3
    
    experiments['Q2_width_1'] = {
        'config': {'whiten': {'width_multiplier': 1}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q2: Width multiplier 1 (positive eigenvectors only)',
        'phase': 'Q2',
    }
    
    experiments['Q2_width_3'] = {
        'config': {'whiten': {'width_multiplier': 3}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q2: Width multiplier 3 (add learned channels)',
        'phase': 'Q2',
    }
    
    # Phase 3: epsilon ablation
    # Default is 5e-4, test range from 5e-5 to 5e-3
    
    experiments['Q3_eps_5e-5'] = {
        'config': {'whiten': {'eps': 5e-5}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q3: Whitening epsilon 5e-5 (less regularization)',
        'phase': 'Q3',
    }
    
    experiments['Q3_eps_1e-4'] = {
        'config': {'whiten': {'eps': 1e-4}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q3: Whitening epsilon 1e-4',
        'phase': 'Q3',
    }
    
    experiments['Q3_eps_1e-3'] = {
        'config': {'whiten': {'eps': 1e-3}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q3: Whitening epsilon 1e-3',
        'phase': 'Q3',
    }
    
    experiments['Q3_eps_5e-3'] = {
        'config': {'whiten': {'eps': 5e-3}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q3: Whitening epsilon 5e-3 (more regularization)',
        'phase': 'Q3',
    }
    
    # Phase 4: trainability ablation (Q4)
    # Default is never trainable, test unfreezing at different epochs
    
    experiments['Q4_trainable_epoch0'] = {
        'config': {'whiten': {'trainable_after_epoch': 0, 'trainable_lr_mult': 0.01}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q4: Whitening trainable from epoch 0 (0.01x LR)',
        'phase': 'Q4',
    }
    
    experiments['Q4_trainable_epoch3'] = {
        'config': {'whiten': {'trainable_after_epoch': 3, 'trainable_lr_mult': 0.01}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q4: Whitening trainable from epoch 3 (0.01x LR)',
        'phase': 'Q4',
    }
    
    experiments['Q4_trainable_epoch5'] = {
        'config': {'whiten': {'trainable_after_epoch': 5, 'trainable_lr_mult': 0.01}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q4: Whitening trainable from epoch 5 (0.01x LR)',
        'phase': 'Q4',
    }
    
    experiments['Q4_trainable_epoch3_lr0.1'] = {
        'config': {'whiten': {'trainable_after_epoch': 3, 'trainable_lr_mult': 0.1}},
        'num_runs': RUNS_ABLATION,
        'description': 'Q4: Whitening trainable from epoch 3 (0.1x LR)',
        'phase': 'Q4',
    }
    
    return experiments


def get_combination_experiments():
    """Define combination experiments to test promising combinations, call after phases"""
    experiments = {}
    
    # Example combinations (modify based on your Phase 1-4 results!)
    # These are placeholders - you should update based on what works
    
    experiments['combo_1'] = {
        'config': {
            'whiten': {
                'kernel_size': 3,      # If Q1 shows 3x3 is better
                'eps': 1e-3,           # If Q3 shows higher eps is better
            }
        },
        'num_runs': RUNS_ABLATION,
        'description': 'Combination: 3x3 kernel + eps=1e-3',
        'phase': 'combo',
    }
    
    experiments['combo_2'] = {
        'config': {
            'whiten': {
                'kernel_size': 3,
                'width_multiplier': 1,  # If Q2 shows width=1 is sufficient
            }
        },
        'num_runs': RUNS_ABLATION,
        'description': 'Combination: 3x3 kernel + width=1',
        'phase': 'combo',
    }
    
    return experiments


#############################################
#           Experiment Execution
#############################################

def run_single_experiment(name, exp_config, verbose=True):
    """Run a single experiment and return results."""
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {name}")
    print(f"# {exp_config['description']}")
    print(f"# Runs: {exp_config['num_runs']}")
    print(f"{'#'*70}\n")
    
    results = run_experiment(
        config=exp_config['config'],
        num_runs=exp_config['num_runs'],
        experiment_name=name,
        verbose=verbose,
    )
    
    return results


def run_experiments_by_phase(phase, verbose=True):
    """Run all experiments in a specific phase."""
    all_experiments = get_all_experiments()
    
    experiments_to_run = {
        name: config for name, config in all_experiments.items()
        if config['phase'] == phase
    }
    
    if not experiments_to_run:
        print(f"No experiments found for phase: {phase}")
        return {}
    
    print(f"\n{'='*70}")
    print(f"RUNNING PHASE: {phase}")
    print(f"Experiments: {list(experiments_to_run.keys())}")
    print(f"{'='*70}\n")
    
    results = {}
    for name, exp_config in experiments_to_run.items():
        results[name] = run_single_experiment(name, exp_config, verbose)
    
    return results


def run_all_experiments(verbose=True):
    """Run all experiments in sequence."""
    all_experiments = get_all_experiments()
    
    print(f"\n{'='*70}")
    print(f"RUNNING ALL EXPERIMENTS")
    print(f"Total experiments: {len(all_experiments)}")
    total_runs = sum(e['num_runs'] for e in all_experiments.values())
    print(f"Total runs: {total_runs}")
    print(f"Estimated time: ~{total_runs * 4 / 60:.1f} minutes (at 4s/run)")
    print(f"{'='*70}\n")
    
    results = {}
    for name, exp_config in all_experiments.items():
        results[name] = run_single_experiment(name, exp_config, verbose)
    
    return results

#############################################
#              Main Entry Point
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run whitening layer experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--phase', type=str, default=None,
                        choices=['baseline', 'Q1', 'Q2', 'Q3', 'Q4', 'combo'],
                        help='Run only experiments in this phase')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Run a specific experiment by name')
    parser.add_argument('--quiet', action='store_true',
                        help='No verbose')
    
    args = parser.parse_args()

    if args.experiment:
        all_experiments = get_all_experiments()
        if args.experiment not in all_experiments:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(all_experiments.keys())}")
            sys.exit(1)
        run_single_experiment(args.experiment, all_experiments[args.experiment], 
                             verbose=not args.quiet)
    elif args.phase:
        run_experiments_by_phase(args.phase, verbose=not args.quiet)
    else:
        run_all_experiments(verbose=not args.quiet)