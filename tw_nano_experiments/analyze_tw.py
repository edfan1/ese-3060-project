#!/usr/bin/env python3
"""
Token Importance Weighting Ablation Analysis

This script analyzes results from the ablation experiments, computing
statistics, performing hypothesis tests, and generating visualizations.

Usage:
    # Analyze all results
    python analyze_tw.py --results-dir ablation_results

    # Analyze specific phase
    python analyze_tw.py --results-dir ablation_results --phase 2

    # Generate comparison report
    python analyze_tw.py --results-dir ablation_results --compare-to-baseline

    # Export results to CSV
    python analyze_tw.py --results-dir ablation_results --export-csv
"""

import os
import re
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainingRun:
    """Parsed data from a single training run."""
    name: str
    config: Dict[str, Any]
    phase: int
    stage: str
    seed: int
    
    # Training metrics (per step)
    steps: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_steps: List[int] = field(default_factory=list)
    train_times_ms: List[float] = field(default_factory=list)
    step_avgs_ms: List[float] = field(default_factory=list)
    
    # Weight statistics (if token weighting enabled)
    weight_means: List[float] = field(default_factory=list)
    weight_stds: List[float] = field(default_factory=list)
    weight_mins: List[float] = field(default_factory=list)
    weight_maxs: List[float] = field(default_factory=list)
    weight_steps: List[int] = field(default_factory=list)
    
    # Final metrics
    final_val_loss: Optional[float] = None
    final_train_time_ms: Optional[float] = None
    peak_memory_mib: Optional[int] = None
    
    # Derived metrics
    time_to_targets: Dict[float, Optional[float]] = field(default_factory=dict)


@dataclass 
class ExperimentGroup:
    """Group of runs with the same configuration but different seeds."""
    name: str
    config: Dict[str, Any]
    phase: int
    stage: str
    runs: List[TrainingRun] = field(default_factory=list)
    
    @property
    def seeds(self) -> List[int]:
        return [r.seed for r in self.runs]
    
    @property
    def n_runs(self) -> int:
        return len(self.runs)
    
    def get_final_val_losses(self) -> np.ndarray:
        return np.array([r.final_val_loss for r in self.runs if r.final_val_loss is not None])
    
    def get_final_train_times(self) -> np.ndarray:
        return np.array([r.final_train_time_ms for r in self.runs if r.final_train_time_ms is not None])
    
    def get_step_avg_times(self) -> np.ndarray:
        """Get the final step average time from each run."""
        avgs = []
        for r in self.runs:
            if r.step_avgs_ms:
                # Get the last valid (non-nan) step average
                valid_avgs = [x for x in r.step_avgs_ms if not np.isnan(x)]
                if valid_avgs:
                    avgs.append(valid_avgs[-1])
        return np.array(avgs)
    
    def get_peak_memory(self) -> np.ndarray:
        """Get peak memory usage from each run."""
        return np.array([r.peak_memory_mib for r in self.runs if r.peak_memory_mib is not None])
    
    def get_val_losses_at_step(self, step: int) -> np.ndarray:
        """Get validation losses at a specific step (for truncated comparison)."""
        losses = []
        for r in self.runs:
            loss = get_val_loss_at_step(r, step)
            if loss is not None:
                losses.append(loss)
        return np.array(losses)
    
    def get_train_times_at_step(self, step: int) -> np.ndarray:
        """Get training times at a specific step (for truncated comparison)."""
        times = []
        for r in self.runs:
            t = get_train_time_at_step(r, step)
            if t is not None:
                times.append(t)
        return np.array(times)


def compare_at_step(
    baseline: ExperimentGroup,
    treatment: ExperimentGroup,
    step: int,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compare treatment group to baseline at a specific step.
    Useful for comparing screening runs to truncated full runs.
    """
    baseline_vals = baseline.get_val_losses_at_step(step)
    treatment_vals = treatment.get_val_losses_at_step(step)
    
    baseline_times = baseline.get_train_times_at_step(step)
    treatment_times = treatment.get_train_times_at_step(step)
    
    result = {
        'step': step,
        'baseline_n': len(baseline_vals),
        'treatment_n': len(treatment_vals),
    }
    
    # Validation loss comparison
    if len(baseline_vals) >= 1 and len(treatment_vals) >= 1:
        baseline_mean = np.mean(baseline_vals)
        treatment_mean = np.mean(treatment_vals)
        mean_diff = treatment_mean - baseline_mean
        percent_change = (mean_diff / baseline_mean) * 100 if baseline_mean != 0 else np.nan
        
        result.update({
            'baseline_mean': baseline_mean,
            'treatment_mean': treatment_mean,
            'mean_diff': mean_diff,
            'percent_change': percent_change,
            'baseline_std': np.std(baseline_vals, ddof=1) if len(baseline_vals) > 1 else 0,
            'treatment_std': np.std(treatment_vals, ddof=1) if len(treatment_vals) > 1 else 0,
        })
        
        # Statistical test if enough samples
        if len(baseline_vals) >= 2 and len(treatment_vals) >= 2:
            t_stat, p_value = stats.ttest_ind(treatment_vals, baseline_vals, equal_var=False)
            result.update({
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'cohens_d': cohens_d(treatment_vals, baseline_vals),
            })
        else:
            result.update({
                't_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'cohens_d': np.nan,
            })
    else:
        result.update({
            'baseline_mean': np.nan,
            'treatment_mean': np.nan,
            'mean_diff': np.nan,
            'percent_change': np.nan,
        })
    
    # Time comparison
    if len(baseline_times) >= 1 and len(treatment_times) >= 1:
        result.update({
            'time_baseline_mean_ms': np.mean(baseline_times),
            'time_treatment_mean_ms': np.mean(treatment_times),
            'time_diff_ms': np.mean(treatment_times) - np.mean(baseline_times),
        })
    
    return result


def generate_truncated_comparison_report(
    groups: Dict[str, ExperimentGroup],
    baseline_name: str,
    comparison_step: int,
    output_dir: Path
):
    """
    Generate comparison report at a specific step.
    Useful for comparing screening runs against truncated baseline.
    """
    if baseline_name not in groups:
        print(f"Baseline '{baseline_name}' not found")
        return
    
    baseline = groups[baseline_name]
    
    print(f"\n{'='*60}")
    print(f"TRUNCATED COMPARISON AT STEP {comparison_step}")
    print(f"{'='*60}")
    
    # Get baseline val loss at this step
    baseline_losses = baseline.get_val_losses_at_step(comparison_step)
    if len(baseline_losses) == 0:
        print(f"No baseline data at step {comparison_step}")
        return
    
    print(f"\nBaseline at step {comparison_step}:")
    print(f"  Val Loss: {np.mean(baseline_losses):.4f} ± {np.std(baseline_losses, ddof=1) if len(baseline_losses) > 1 else 0:.4f} (n={len(baseline_losses)})")
    
    # Compare each treatment
    results = []
    print(f"\nComparisons:")
    print("-" * 60)
    
    for group_name, group in sorted(groups.items()):
        if group_name == baseline_name:
            continue
        
        comparison = compare_at_step(baseline, group, comparison_step)
        
        if np.isnan(comparison.get('treatment_mean', np.nan)):
            continue
        
        results.append({
            'name': group_name,
            **comparison
        })
        
        sig_marker = "✓" if comparison.get('significant', False) else ""
        print(f"\n{group_name}:")
        print(f"  Val Loss: {comparison['treatment_mean']:.4f} ± {comparison.get('treatment_std', 0):.4f}")
        print(f"  vs Baseline: {comparison['mean_diff']:.4f} ({comparison['percent_change']:+.2f}%) {sig_marker}")
        if not np.isnan(comparison.get('p_value', np.nan)):
            print(f"  p-value: {comparison['p_value']:.4f}, Cohen's d: {comparison.get('cohens_d', np.nan):.3f}")
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('treatment_mean')
        csv_path = output_dir / f"comparison_at_step_{comparison_step}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        
        # Print ranking
        print(f"\n{'='*60}")
        print(f"RANKING AT STEP {comparison_step} (by val loss)")
        print(f"{'='*60}")
        for i, row in df.iterrows():
            sig = "✓" if row.get('significant', False) else " "
            print(f"  {row['treatment_mean']:.4f} ({row['percent_change']:+.2f}%) [{sig}] {row['name']}")
    
    return results


# =============================================================================
# Log Parsing
# =============================================================================

def parse_log_file(log_path: Path) -> Dict[str, Any]:
    """Parse a training log file and extract metrics."""
    metrics = {
        'steps': [],
        'train_losses': [],
        'val_losses': [],
        'val_steps': [],
        'train_times_ms': [],
        'step_avgs_ms': [],
        'weight_means': [],
        'weight_stds': [],
        'weight_mins': [],
        'weight_maxs': [],
        'weight_steps': [],
        'final_val_loss': None,
        'final_train_time_ms': None,
        'peak_memory_mib': None,
    }
    
    if not log_path.exists():
        return metrics
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern for training steps
    # step:1/5100 train_loss:10.8273 train_time:1234ms step_avg:123.45ms
    train_pattern = r'step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms'
    
    # Pattern for training steps with weight stats
    # ... w_mean:1.000 w_std:0.123 w_min:0.500 w_max:2.000
    train_weight_pattern = r'step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms w_mean:([\d.]+) w_std:([\d.]+) w_min:([\d.]+) w_max:([\d.]+)'
    
    # Pattern for validation steps
    # step:0/5100 val_loss:10.9562 train_time:0ms step_avg:nanms
    val_pattern = r'step:(\d+)/\d+ val_loss:([\d.]+) train_time:(\d+)ms'
    
    # Pattern for peak memory
    peak_mem_pattern = r'peak memory consumption: (\d+) MiB'
    
    # Parse training steps with weight stats first
    for match in re.finditer(train_weight_pattern, content):
        step = int(match.group(1))
        metrics['steps'].append(step)
        metrics['train_losses'].append(float(match.group(2)))
        metrics['train_times_ms'].append(float(match.group(3)))
        metrics['step_avgs_ms'].append(float(match.group(4)))
        metrics['weight_steps'].append(step)
        metrics['weight_means'].append(float(match.group(5)))
        metrics['weight_stds'].append(float(match.group(6)))
        metrics['weight_mins'].append(float(match.group(7)))
        metrics['weight_maxs'].append(float(match.group(8)))
    
    # Parse training steps without weight stats (if not already parsed)
    if not metrics['steps']:
        for match in re.finditer(train_pattern, content):
            step = int(match.group(1))
            metrics['steps'].append(step)
            metrics['train_losses'].append(float(match.group(2)))
            metrics['train_times_ms'].append(float(match.group(3)))
            metrics['step_avgs_ms'].append(float(match.group(4)))
    
    # Parse validation steps
    for match in re.finditer(val_pattern, content):
        step = int(match.group(1))
        val_loss = float(match.group(2))
        metrics['val_steps'].append(step)
        metrics['val_losses'].append(val_loss)
        metrics['final_val_loss'] = val_loss  # Last one is final
        metrics['final_train_time_ms'] = float(match.group(3))
    
    # Parse peak memory
    mem_match = re.search(peak_mem_pattern, content)
    if mem_match:
        metrics['peak_memory_mib'] = int(mem_match.group(1))
    
    return metrics


def find_log_file(exp_dir: Path) -> Optional[Path]:
    """Find the training log file in an experiment directory."""
    # Check for run.log first (from our runner)
    run_log = exp_dir / "run.log"
    if run_log.exists():
        return run_log
    
    # Check logs subdirectory for UUID-named logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        # Find most recent log file
        log_files = list(logs_dir.glob("*.txt"))
        if log_files:
            return max(log_files, key=lambda p: p.stat().st_mtime)
    
    return None


def load_experiment(exp_dir: Path) -> Optional[TrainingRun]:
    """Load a single experiment from its directory."""
    config_file = exp_dir / "config.json"
    if not config_file.exists():
        return None
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Find and parse log file
    log_file = find_log_file(exp_dir)
    if log_file:
        metrics = parse_log_file(log_file)
    else:
        metrics = {}
    
    # Create TrainingRun
    run = TrainingRun(
        name=config.get('name', exp_dir.name),
        config=config,
        phase=config.get('phase', 0),
        stage=config.get('stage', 'unknown'),
        seed=config.get('seed', 0),
        steps=metrics.get('steps', []),
        train_losses=metrics.get('train_losses', []),
        val_losses=metrics.get('val_losses', []),
        val_steps=metrics.get('val_steps', []),
        train_times_ms=metrics.get('train_times_ms', []),
        step_avgs_ms=metrics.get('step_avgs_ms', []),
        weight_means=metrics.get('weight_means', []),
        weight_stds=metrics.get('weight_stds', []),
        weight_mins=metrics.get('weight_mins', []),
        weight_maxs=metrics.get('weight_maxs', []),
        weight_steps=metrics.get('weight_steps', []),
        final_val_loss=metrics.get('final_val_loss'),
        final_train_time_ms=metrics.get('final_train_time_ms'),
        peak_memory_mib=metrics.get('peak_memory_mib'),
    )
    
    # Compute time-to-target metrics
    targets = [3.4, 3.35, 3.30, 3.28]
    for target in targets:
        run.time_to_targets[target] = compute_time_to_target(run, target)
    
    return run


def get_val_loss_at_step(run: TrainingRun, target_step: int) -> Optional[float]:
    """Get validation loss at or nearest to target step."""
    if not run.val_steps or not run.val_losses:
        return None
    
    # Find the closest step <= target_step
    best_idx = None
    best_step = -1
    for i, step in enumerate(run.val_steps):
        if step <= target_step and step > best_step:
            best_step = step
            best_idx = i
    
    if best_idx is not None:
        return run.val_losses[best_idx]
    return None


def get_train_time_at_step(run: TrainingRun, target_step: int) -> Optional[float]:
    """Get training time at or nearest to target step."""
    if not run.steps or not run.train_times_ms:
        return None
    
    # Find the closest step <= target_step
    best_idx = None
    best_step = -1
    for i, step in enumerate(run.steps):
        if step <= target_step and step > best_step:
            best_step = step
            best_idx = i
    
    if best_idx is not None:
        return run.train_times_ms[best_idx]
    return None


def compute_time_to_target(run: TrainingRun, target_loss: float) -> Optional[float]:
    """Compute time (in ms) to reach target validation loss."""
    for i, (step, loss) in enumerate(zip(run.val_steps, run.val_losses)):
        if loss <= target_loss:
            # Interpolate time if we have step timing
            if run.train_times_ms and step > 0:
                # Find the training time at this step
                for j, (train_step, train_time) in enumerate(zip(run.steps, run.train_times_ms)):
                    if train_step >= step:
                        return train_time
            return None  # Can't determine time
    return None  # Target not reached


def load_all_experiments(results_dir: Path) -> List[TrainingRun]:
    """Load all experiments from the results directory."""
    runs = []
    
    # Walk through phase directories
    for phase_dir in sorted(results_dir.glob("phase*")):
        if not phase_dir.is_dir():
            continue
        
        for stage_dir in sorted(phase_dir.glob("*")):
            if not stage_dir.is_dir():
                continue
            
            for exp_dir in sorted(stage_dir.glob("*")):
                if not exp_dir.is_dir():
                    continue
                
                run = load_experiment(exp_dir)
                if run:
                    runs.append(run)
    
    return runs


def group_experiments(runs: List[TrainingRun]) -> Dict[str, ExperimentGroup]:
    """Group runs by configuration (ignoring seed)."""
    groups = {}
    
    for run in runs:
        # Create group key from config without seed
        config_key = {k: v for k, v in run.config.items() if k != 'seed'}
        
        # Determine group name (remove seed suffix)
        group_name = re.sub(r'_seed\d+$', '', run.name)
        
        if group_name not in groups:
            groups[group_name] = ExperimentGroup(
                name=group_name,
                config=config_key,
                phase=run.phase,
                stage=run.stage,
            )
        
        groups[group_name].runs.append(run)
    
    return groups


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for an array of values."""
    if len(values) == 0:
        return {}
    
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'n': len(values),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compare_to_baseline(
    baseline: ExperimentGroup,
    treatment: ExperimentGroup,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compare treatment group to baseline using statistical tests.
    
    Returns dict with:
    - mean_diff: Difference in means (treatment - baseline)
    - percent_change: Percentage change from baseline
    - t_statistic: T-test statistic
    - p_value: Two-tailed p-value
    - significant: Whether difference is significant at alpha level
    - cohens_d: Effect size
    - ci_low, ci_high: 95% confidence interval for difference
    - Runtime comparisons (time_*)
    """
    baseline_vals = baseline.get_final_val_losses()
    treatment_vals = treatment.get_final_val_losses()
    
    # Also get runtime data
    baseline_times = baseline.get_final_train_times()
    treatment_times = treatment.get_final_train_times()
    
    # Get per-step timing
    baseline_step_avgs = baseline.get_step_avg_times()
    treatment_step_avgs = treatment.get_step_avg_times()
    
    result = {}
    
    # === Validation Loss Comparison ===
    if len(baseline_vals) < 2 or len(treatment_vals) < 2:
        result.update({
            'mean_diff': np.nan,
            'percent_change': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'cohens_d': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'baseline_mean': np.mean(baseline_vals) if len(baseline_vals) > 0 else np.nan,
            'treatment_mean': np.mean(treatment_vals) if len(treatment_vals) > 0 else np.nan,
            'baseline_std': np.std(baseline_vals, ddof=1) if len(baseline_vals) > 1 else np.nan,
            'treatment_std': np.std(treatment_vals, ddof=1) if len(treatment_vals) > 1 else np.nan,
            'baseline_n': len(baseline_vals),
            'treatment_n': len(treatment_vals),
        })
    else:
        # Two-sample t-test (Welch's t-test for unequal variances)
        t_stat, p_value = stats.ttest_ind(treatment_vals, baseline_vals, equal_var=False)
        
        # Effect size
        d = cohens_d(treatment_vals, baseline_vals)
        
        # Mean difference and percent change
        baseline_mean = np.mean(baseline_vals)
        treatment_mean = np.mean(treatment_vals)
        mean_diff = treatment_mean - baseline_mean
        percent_change = (mean_diff / baseline_mean) * 100 if baseline_mean != 0 else np.nan
        
        # Confidence interval for difference (using pooled SE approximation)
        se_diff = np.sqrt(np.var(baseline_vals, ddof=1)/len(baseline_vals) + 
                          np.var(treatment_vals, ddof=1)/len(treatment_vals))
        df = len(baseline_vals) + len(treatment_vals) - 2
        t_crit = stats.t.ppf(1 - alpha/2, df)
        ci_low = mean_diff - t_crit * se_diff
        ci_high = mean_diff + t_crit * se_diff
        
        result.update({
            'mean_diff': mean_diff,
            'percent_change': percent_change,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': d,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'baseline_mean': baseline_mean,
            'treatment_mean': treatment_mean,
            'baseline_std': np.std(baseline_vals, ddof=1),
            'treatment_std': np.std(treatment_vals, ddof=1),
            'baseline_n': len(baseline_vals),
            'treatment_n': len(treatment_vals),
        })
    
    # === Runtime Comparison ===
    if len(baseline_times) >= 1 and len(treatment_times) >= 1:
        baseline_time_mean = np.mean(baseline_times) / 1000  # Convert to seconds
        treatment_time_mean = np.mean(treatment_times) / 1000
        time_diff = treatment_time_mean - baseline_time_mean
        time_percent_change = (time_diff / baseline_time_mean) * 100 if baseline_time_mean != 0 else np.nan
        
        result.update({
            'time_baseline_mean_s': baseline_time_mean,
            'time_treatment_mean_s': treatment_time_mean,
            'time_diff_s': time_diff,
            'time_percent_change': time_percent_change,
            'time_baseline_std_s': np.std(baseline_times, ddof=1) / 1000 if len(baseline_times) > 1 else np.nan,
            'time_treatment_std_s': np.std(treatment_times, ddof=1) / 1000 if len(treatment_times) > 1 else np.nan,
        })
        
        # Statistical test for runtime if we have enough samples
        if len(baseline_times) >= 2 and len(treatment_times) >= 2:
            time_t_stat, time_p_value = stats.ttest_ind(treatment_times, baseline_times, equal_var=False)
            result.update({
                'time_t_statistic': time_t_stat,
                'time_p_value': time_p_value,
                'time_significant': time_p_value < alpha,
                'time_cohens_d': cohens_d(treatment_times, baseline_times),
            })
        else:
            result.update({
                'time_t_statistic': np.nan,
                'time_p_value': np.nan,
                'time_significant': False,
                'time_cohens_d': np.nan,
            })
    else:
        result.update({
            'time_baseline_mean_s': np.nan,
            'time_treatment_mean_s': np.nan,
            'time_diff_s': np.nan,
            'time_percent_change': np.nan,
            'time_baseline_std_s': np.nan,
            'time_treatment_std_s': np.nan,
            'time_t_statistic': np.nan,
            'time_p_value': np.nan,
            'time_significant': False,
            'time_cohens_d': np.nan,
        })
    
    # === Per-Step Timing (Computational Overhead) ===
    if len(baseline_step_avgs) >= 1 and len(treatment_step_avgs) >= 1:
        baseline_step_mean = np.mean(baseline_step_avgs)
        treatment_step_mean = np.mean(treatment_step_avgs)
        step_overhead = treatment_step_mean - baseline_step_mean
        step_overhead_percent = (step_overhead / baseline_step_mean) * 100 if baseline_step_mean != 0 else np.nan
        
        result.update({
            'step_avg_baseline_ms': baseline_step_mean,
            'step_avg_treatment_ms': treatment_step_mean,
            'step_overhead_ms': step_overhead,
            'step_overhead_percent': step_overhead_percent,
        })
        
        if len(baseline_step_avgs) >= 2 and len(treatment_step_avgs) >= 2:
            step_t_stat, step_p_value = stats.ttest_ind(treatment_step_avgs, baseline_step_avgs, equal_var=False)
            result.update({
                'step_t_statistic': step_t_stat,
                'step_p_value': step_p_value,
                'step_significant': step_p_value < alpha,
            })
    else:
        result.update({
            'step_avg_baseline_ms': np.nan,
            'step_avg_treatment_ms': np.nan,
            'step_overhead_ms': np.nan,
            'step_overhead_percent': np.nan,
            'step_t_statistic': np.nan,
            'step_p_value': np.nan,
            'step_significant': False,
        })
    
    return result


def apply_fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values and track original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    
    # Apply BH procedure
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(sorted_pairs, 1):
        threshold = (rank / n) * alpha
        if p <= threshold:
            significant[orig_idx] = True
        else:
            break
    
    return significant


# =============================================================================
# Visualization
# =============================================================================

def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })


def plot_training_curves(
    groups: Dict[str, ExperimentGroup],
    output_path: Path,
    title: str = "Training Curves",
    metric: str = "val_loss"
):
    """Plot training curves for multiple experiment groups."""
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    for (group_name, group), color in zip(groups.items(), colors):
        if metric == "val_loss":
            # Collect validation losses across runs
            all_steps = set()
            for run in group.runs:
                all_steps.update(run.val_steps)
            all_steps = sorted(all_steps)
            
            if not all_steps:
                continue
            
            # Interpolate to common steps
            losses_matrix = []
            for run in group.runs:
                if run.val_steps and run.val_losses:
                    losses = np.interp(all_steps, run.val_steps, run.val_losses)
                    losses_matrix.append(losses)
            
            if not losses_matrix:
                continue
            
            losses_matrix = np.array(losses_matrix)
            mean_losses = np.mean(losses_matrix, axis=0)
            std_losses = np.std(losses_matrix, axis=0)
            
            # Plot mean with confidence band
            ax.plot(all_steps, mean_losses, label=group_name, color=color, linewidth=2)
            ax.fill_between(all_steps, mean_losses - std_losses, mean_losses + std_losses,
                           color=color, alpha=0.2)
        
        elif metric == "train_loss":
            # Similar for training loss
            all_steps = set()
            for run in group.runs:
                all_steps.update(run.steps)
            all_steps = sorted(all_steps)
            
            if not all_steps:
                continue
            
            losses_matrix = []
            for run in group.runs:
                if run.steps and run.train_losses:
                    losses = np.interp(all_steps, run.steps, run.train_losses)
                    losses_matrix.append(losses)
            
            if not losses_matrix:
                continue
            
            losses_matrix = np.array(losses_matrix)
            mean_losses = np.mean(losses_matrix, axis=0)
            std_losses = np.std(losses_matrix, axis=0)
            
            ax.plot(all_steps, mean_losses, label=group_name, color=color, linewidth=2)
            ax.fill_between(all_steps, mean_losses - std_losses, mean_losses + std_losses,
                           color=color, alpha=0.2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_loss_comparison(
    groups: Dict[str, ExperimentGroup],
    output_path: Path,
    baseline_name: Optional[str] = None
):
    """Create box plot comparing final validation losses."""
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    data = []
    labels = []
    colors = []
    
    for group_name, group in groups.items():
        losses = group.get_final_val_losses()
        if len(losses) > 0:
            data.append(losses)
            labels.append(f"{group_name}\n(n={len(losses)})")
            colors.append('lightblue' if group_name == baseline_name else 'lightgreen')
    
    if not data:
        print("No data to plot for final loss comparison")
        return
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, s=50, zorder=3)
    
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Final Validation Loss Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # Add baseline reference line if available
    if baseline_name and baseline_name in groups:
        baseline_mean = np.mean(groups[baseline_name].get_final_val_losses())
        ax.axhline(y=baseline_mean, color='red', linestyle='--', alpha=0.7, 
                   label=f'Baseline mean: {baseline_mean:.4f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_weight_statistics(
    runs: List[TrainingRun],
    output_path: Path
):
    """Plot weight statistics over training."""
    setup_plotting_style()
    
    # Filter runs with weight statistics
    runs_with_weights = [r for r in runs if r.weight_means]
    
    if not runs_with_weights:
        print("No weight statistics to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_with_weights)))
    
    for run, color in zip(runs_with_weights, colors):
        label = f"{run.name}"
        
        # Mean weights
        axes[0, 0].plot(run.weight_steps, run.weight_means, label=label, color=color, alpha=0.7)
        axes[0, 0].set_ylabel('Weight Mean')
        axes[0, 0].set_title('Mean Token Weights Over Training')
        
        # Std weights
        axes[0, 1].plot(run.weight_steps, run.weight_stds, label=label, color=color, alpha=0.7)
        axes[0, 1].set_ylabel('Weight Std Dev')
        axes[0, 1].set_title('Weight Standard Deviation Over Training')
        
        # Min weights
        axes[1, 0].plot(run.weight_steps, run.weight_mins, label=label, color=color, alpha=0.7)
        axes[1, 0].set_ylabel('Weight Min')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_title('Minimum Token Weights Over Training')
        
        # Max weights
        axes[1, 1].plot(run.weight_steps, run.weight_maxs, label=label, color=color, alpha=0.7)
        axes[1, 1].set_ylabel('Weight Max')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_title('Maximum Token Weights Over Training')
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='upper right', fontsize=8)
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_phase_comparison_heatmap(
    comparison_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    metric: str = 'percent_change'
):
    """Create heatmap of comparison results."""
    setup_plotting_style()
    
    if not comparison_results:
        print("No comparison results to plot")
        return
    
    # Extract data for heatmap
    names = list(comparison_results.keys())
    values = [comparison_results[n].get(metric, np.nan) for n in names]
    p_values = [comparison_results[n].get('p_value', np.nan) for n in names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.5)))
    
    # Create horizontal bar chart instead of heatmap for single row
    colors = ['green' if v < 0 else 'red' for v in values]
    bars = ax.barh(names, values, color=colors, alpha=0.7)
    
    # Add significance markers
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        if not np.isnan(p) and p < 0.05:
            ax.annotate('*', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                       fontsize=14, fontweight='bold', va='center')
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel(f'{metric.replace("_", " ").title()} (%)')
    ax.set_title(f'Comparison to Baseline ({metric.replace("_", " ").title()})\n* indicates p < 0.05')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_time_to_target(
    groups: Dict[str, ExperimentGroup],
    output_path: Path,
    target: float = 3.28
):
    """Plot time to reach target validation loss."""
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = []
    times = []
    errors = []
    
    for group_name, group in groups.items():
        group_times = []
        for run in group.runs:
            t = run.time_to_targets.get(target)
            if t is not None:
                group_times.append(t / 1000)  # Convert to seconds
        
        if group_times:
            names.append(group_name)
            times.append(np.mean(group_times))
            errors.append(np.std(group_times) if len(group_times) > 1 else 0)
    
    if not names:
        print(f"No runs reached target loss {target}")
        return
    
    x = np.arange(len(names))
    bars = ax.bar(x, times, yerr=errors, capsize=5, color='steelblue', alpha=0.7)
    
    ax.set_ylabel('Time to Target (seconds)')
    ax.set_title(f'Time to Reach Validation Loss {target}')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_runtime_comparison(
    groups: Dict[str, ExperimentGroup],
    output_path: Path,
    baseline_name: Optional[str] = None
):
    """Plot runtime comparison across experiment groups."""
    setup_plotting_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    names = []
    total_times = []
    total_errors = []
    step_times = []
    step_errors = []
    memory_usage = []
    memory_errors = []
    
    for group_name, group in groups.items():
        names.append(group_name)
        
        # Total training time
        times = group.get_final_train_times() / 1000  # Convert to seconds
        total_times.append(np.mean(times) if len(times) > 0 else 0)
        total_errors.append(np.std(times) if len(times) > 1 else 0)
        
        # Per-step time
        step_avgs = group.get_step_avg_times()
        step_times.append(np.mean(step_avgs) if len(step_avgs) > 0 else 0)
        step_errors.append(np.std(step_avgs) if len(step_avgs) > 1 else 0)
        
        # Memory usage
        memory = group.get_peak_memory()
        memory_usage.append(np.mean(memory) if len(memory) > 0 else 0)
        memory_errors.append(np.std(memory) if len(memory) > 1 else 0)
    
    x = np.arange(len(names))
    
    # Determine colors (highlight baseline)
    colors = ['steelblue' if n != baseline_name else 'coral' for n in names]
    
    # Plot 1: Total Training Time
    axes[0].bar(x, total_times, yerr=total_errors, capsize=3, color=colors, alpha=0.7)
    axes[0].set_ylabel('Total Time (seconds)')
    axes[0].set_title('Total Training Time')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    
    # Add baseline reference line
    if baseline_name and baseline_name in groups:
        baseline_idx = names.index(baseline_name)
        axes[0].axhline(y=total_times[baseline_idx], color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Per-Step Time
    axes[1].bar(x, step_times, yerr=step_errors, capsize=3, color=colors, alpha=0.7)
    axes[1].set_ylabel('Time per Step (ms)')
    axes[1].set_title('Per-Step Training Time')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    
    if baseline_name and baseline_name in groups:
        baseline_idx = names.index(baseline_name)
        axes[1].axhline(y=step_times[baseline_idx], color='red', linestyle='--', alpha=0.5)
    
    # Plot 3: Memory Usage
    axes[2].bar(x, memory_usage, yerr=memory_errors, capsize=3, color=colors, alpha=0.7)
    axes[2].set_ylabel('Peak Memory (MiB)')
    axes[2].set_title('Peak Memory Usage')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    
    if baseline_name and baseline_name in groups:
        baseline_idx = names.index(baseline_name)
        if memory_usage[baseline_idx] > 0:
            axes[2].axhline(y=memory_usage[baseline_idx], color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_overhead_analysis(
    comparisons: Dict[str, Dict[str, Any]],
    output_path: Path
):
    """Plot computational overhead analysis."""
    setup_plotting_style()
    
    if not comparisons:
        print("No comparison data for overhead analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(comparisons.keys())
    
    # Per-step overhead
    step_overheads = [comparisons[n].get('step_overhead_percent', np.nan) for n in names]
    step_colors = ['green' if x < 0 else 'red' if x > 0 else 'gray' for x in step_overheads]
    
    valid_names = [n for n, v in zip(names, step_overheads) if not np.isnan(v)]
    valid_overheads = [v for v in step_overheads if not np.isnan(v)]
    valid_colors = [c for c, v in zip(step_colors, step_overheads) if not np.isnan(v)]
    
    if valid_names:
        axes[0].barh(valid_names, valid_overheads, color=valid_colors, alpha=0.7)
        axes[0].axvline(x=0, color='black', linewidth=1)
        axes[0].set_xlabel('Per-Step Overhead (%)')
        axes[0].set_title('Computational Overhead per Step\n(negative = faster)')
        
        # Add significance markers
        for i, name in enumerate(valid_names):
            if comparisons[name].get('step_significant', False):
                axes[0].annotate('*', xy=(valid_overheads[i], i), fontsize=14, fontweight='bold')
    
    # Total time overhead
    time_overheads = [comparisons[n].get('time_percent_change', np.nan) for n in names]
    time_colors = ['green' if x < 0 else 'red' if x > 0 else 'gray' for x in time_overheads]
    
    valid_names = [n for n, v in zip(names, time_overheads) if not np.isnan(v)]
    valid_overheads = [v for v in time_overheads if not np.isnan(v)]
    valid_colors = [c for c, v in zip(time_colors, time_overheads) if not np.isnan(v)]
    
    if valid_names:
        axes[1].barh(valid_names, valid_overheads, color=valid_colors, alpha=0.7)
        axes[1].axvline(x=0, color='black', linewidth=1)
        axes[1].set_xlabel('Total Time Change (%)')
        axes[1].set_title('Total Training Time Change\n(negative = faster)')
        
        for i, name in enumerate(valid_names):
            if comparisons[name].get('time_significant', False):
                axes[1].annotate('*', xy=(valid_overheads[i], i), fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================

def generate_summary_table(
    groups: Dict[str, ExperimentGroup],
    baseline_name: Optional[str] = None
) -> pd.DataFrame:
    """Generate summary statistics table."""
    rows = []
    
    baseline_group = groups.get(baseline_name) if baseline_name else None
    
    for group_name, group in groups.items():
        val_losses = group.get_final_val_losses()
        train_times = group.get_final_train_times()
        step_avgs = group.get_step_avg_times()
        peak_mem = group.get_peak_memory()
        
        row = {
            'Name': group_name,
            'Phase': group.phase,
            'Stage': group.stage,
            'N Runs': group.n_runs,
            'Seeds': ', '.join(map(str, group.seeds)),
            'Val Loss (mean)': np.mean(val_losses) if len(val_losses) > 0 else np.nan,
            'Val Loss (std)': np.std(val_losses, ddof=1) if len(val_losses) > 1 else np.nan,
            'Train Time (s)': np.mean(train_times) / 1000 if len(train_times) > 0 else np.nan,
            'Step Avg (ms)': np.mean(step_avgs) if len(step_avgs) > 0 else np.nan,
            'Memory (MiB)': np.mean(peak_mem) if len(peak_mem) > 0 else np.nan,
        }
        
        # Add comparison to baseline if available
        if baseline_group and group_name != baseline_name:
            comparison = compare_to_baseline(baseline_group, group)
            row['Loss vs Base (%)'] = comparison['percent_change']
            row['Loss p-value'] = comparison['p_value']
            row["Cohen's d"] = comparison['cohens_d']
            row['Loss Sig'] = '✓' if comparison['significant'] else ''
            # Runtime comparison
            row['Time vs Base (%)'] = comparison.get('time_percent_change', np.nan)
            row['Time p-value'] = comparison.get('time_p_value', np.nan)
            row['Time Sig'] = '✓' if comparison.get('time_significant', False) else ''
            row['Step Overhead (%)'] = comparison.get('step_overhead_percent', np.nan)
        
        # Add config details
        config = group.config
        row['TW Enabled'] = config.get('tw_enabled', False)
        if config.get('tw_enabled'):
            row['Function'] = config.get('tw_function', '')
            row['Clamp'] = f"[{config.get('tw_clamp_min', '')}, {config.get('tw_clamp_max', '')}]"
            row['Schedule'] = config.get('tw_schedule', '')
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by phase, then by validation loss
    df = df.sort_values(['Phase', 'Val Loss (mean)'])
    
    return df


def generate_report(
    runs: List[TrainingRun],
    groups: Dict[str, ExperimentGroup],
    output_dir: Path,
    baseline_name: Optional[str] = None
):
    """Generate comprehensive analysis report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("ABLATION ANALYSIS REPORT")
    print("="*60)
    
    # Summary statistics
    print(f"\nTotal experiments loaded: {len(runs)}")
    print(f"Experiment groups: {len(groups)}")
    
    # Generate summary table
    summary_df = generate_summary_table(groups, baseline_name)
    
    print("\n" + "-"*60)
    print("SUMMARY TABLE")
    print("-"*60)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    csv_path = output_dir / "summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    # Statistical comparisons
    if baseline_name and baseline_name in groups:
        print("\n" + "-"*60)
        print(f"STATISTICAL COMPARISONS (vs {baseline_name})")
        print("-"*60)
        
        baseline_group = groups[baseline_name]
        comparisons = {}
        
        for group_name, group in groups.items():
            if group_name == baseline_name:
                continue
            
            comparison = compare_to_baseline(baseline_group, group)
            comparisons[group_name] = comparison
            
            print(f"\n{group_name}:")
            print(f"  === Validation Loss ===")
            print(f"  Baseline: {comparison['baseline_mean']:.4f} ± {comparison['baseline_std']:.4f} (n={comparison['baseline_n']})")
            print(f"  Treatment: {comparison['treatment_mean']:.4f} ± {comparison['treatment_std']:.4f} (n={comparison['treatment_n']})")
            print(f"  Difference: {comparison['mean_diff']:.4f} ({comparison['percent_change']:.2f}%)")
            print(f"  95% CI: [{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}]")
            print(f"  t = {comparison['t_statistic']:.3f}, p = {comparison['p_value']:.4f}")
            print(f"  Cohen's d = {comparison['cohens_d']:.3f}")
            print(f"  Significant: {'Yes' if comparison['significant'] else 'No'}")
            
            # Runtime comparison
            if not np.isnan(comparison.get('time_baseline_mean_s', np.nan)):
                print(f"\n  === Runtime ===")
                print(f"  Baseline: {comparison['time_baseline_mean_s']:.1f} ± {comparison.get('time_baseline_std_s', 0):.1f}s")
                print(f"  Treatment: {comparison['time_treatment_mean_s']:.1f} ± {comparison.get('time_treatment_std_s', 0):.1f}s")
                print(f"  Difference: {comparison['time_diff_s']:.1f}s ({comparison['time_percent_change']:.2f}%)")
                if not np.isnan(comparison.get('time_p_value', np.nan)):
                    print(f"  t = {comparison['time_t_statistic']:.3f}, p = {comparison['time_p_value']:.4f}")
                    print(f"  Significant: {'Yes' if comparison['time_significant'] else 'No'}")
            
            # Per-step overhead
            if not np.isnan(comparison.get('step_overhead_ms', np.nan)):
                print(f"\n  === Per-Step Overhead ===")
                print(f"  Baseline: {comparison['step_avg_baseline_ms']:.2f}ms/step")
                print(f"  Treatment: {comparison['step_avg_treatment_ms']:.2f}ms/step")
                print(f"  Overhead: {comparison['step_overhead_ms']:.2f}ms ({comparison['step_overhead_percent']:.2f}%)")
        
        # Save comparisons
        comparisons_df = pd.DataFrame(comparisons).T
        comparisons_path = output_dir / "statistical_comparisons.csv"
        comparisons_df.to_csv(comparisons_path)
        print(f"\nSaved: {comparisons_path}")
        
        # Apply FDR correction
        p_values = [comparisons[n]['p_value'] for n in comparisons if not np.isnan(comparisons[n]['p_value'])]
        if p_values:
            fdr_significant = apply_fdr_correction(p_values)
            print(f"\nFDR-corrected significant results: {sum(fdr_significant)}/{len(fdr_significant)}")
    
    # Generate plots
    print("\n" + "-"*60)
    print("GENERATING PLOTS")
    print("-"*60)
    
    # Training curves
    plot_training_curves(
        groups, 
        output_dir / "training_curves_val.png",
        title="Validation Loss Training Curves",
        metric="val_loss"
    )
    
    plot_training_curves(
        groups,
        output_dir / "training_curves_train.png", 
        title="Training Loss Curves",
        metric="train_loss"
    )
    
    # Final loss comparison
    plot_final_loss_comparison(
        groups,
        output_dir / "final_loss_comparison.png",
        baseline_name=baseline_name
    )
    
    # Weight statistics
    plot_weight_statistics(
        runs,
        output_dir / "weight_statistics.png"
    )
    
    # Time to target
    for target in [3.4, 3.35, 3.30, 3.28]:
        plot_time_to_target(
            groups,
            output_dir / f"time_to_target_{target}.png",
            target=target
        )
    
    # Runtime comparison
    plot_runtime_comparison(
        groups,
        output_dir / "runtime_comparison.png",
        baseline_name=baseline_name
    )
    
    # Comparison heatmap
    if baseline_name and baseline_name in groups:
        plot_phase_comparison_heatmap(
            comparisons,
            output_dir / "comparison_heatmap.png"
        )
        
        # Overhead analysis
        plot_overhead_analysis(
            comparisons,
            output_dir / "overhead_analysis.png"
        )
    
    print("\n" + "="*60)
    print("REPORT COMPLETE")
    print(f"Output directory: {output_dir}")
    print("="*60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze token importance weighting ablation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--results-dir", type=str, default="ablation_results",
        help="Directory containing ablation results"
    )
    parser.add_argument(
        "--output-dir", type=str, default="analysis_output",
        help="Directory for analysis output"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4, 5],
        help="Analyze only a specific phase"
    )
    parser.add_argument(
        "--baseline-name", type=str, default="baseline",
        help="Name of baseline experiment group for comparisons"
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export all data to CSV files"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--compare-at-step", type=int, default=None,
        help="Compare all experiments at a specific step (for truncated baseline comparison)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return
    
    # Load experiments
    print(f"Loading experiments from: {results_dir}")
    runs = load_all_experiments(results_dir)
    
    if not runs:
        print("No experiments found!")
        return
    
    print(f"Loaded {len(runs)} experiment runs")
    
    # Filter by phase if specified
    if args.phase:
        runs = [r for r in runs if r.phase == args.phase]
        print(f"Filtered to {len(runs)} runs from phase {args.phase}")
    
    # Group experiments
    groups = group_experiments(runs)
    print(f"Grouped into {len(groups)} experiment configurations")
    
    # Generate report
    generate_report(
        runs=runs,
        groups=groups,
        output_dir=output_dir,
        baseline_name=args.baseline_name
    )
    
    # Truncated comparison if requested
    if args.compare_at_step:
        generate_truncated_comparison_report(
            groups=groups,
            baseline_name=args.baseline_name,
            comparison_step=args.compare_at_step,
            output_dir=output_dir
        )
    
    # Export raw data if requested
    if args.export_csv:
        export_dir = output_dir / "raw_data"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export all runs
        all_runs_data = []
        for run in runs:
            run_data = {
                'name': run.name,
                'phase': run.phase,
                'stage': run.stage,
                'seed': run.seed,
                'final_val_loss': run.final_val_loss,
                'final_train_time_ms': run.final_train_time_ms,
                'peak_memory_mib': run.peak_memory_mib,
                **{f'time_to_{k}': v for k, v in run.time_to_targets.items()},
                **{f'config_{k}': v for k, v in run.config.items()},
            }
            all_runs_data.append(run_data)
        
        runs_df = pd.DataFrame(all_runs_data)
        runs_df.to_csv(export_dir / "all_runs.csv", index=False)
        print(f"Exported: {export_dir / 'all_runs.csv'}")
        
        # Export training curves
        for run in runs:
            if run.val_steps and run.val_losses:
                curve_df = pd.DataFrame({
                    'step': run.val_steps,
                    'val_loss': run.val_losses,
                })
                curve_df.to_csv(export_dir / f"{run.name}_val_curve.csv", index=False)
            
            if run.steps and run.train_losses:
                curve_df = pd.DataFrame({
                    'step': run.steps,
                    'train_loss': run.train_losses,
                })
                curve_df.to_csv(export_dir / f"{run.name}_train_curve.csv", index=False)


if __name__ == "__main__":
    main()