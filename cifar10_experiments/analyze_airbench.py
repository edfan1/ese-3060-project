#!/usr/bin/env python3
"""
Analysis Script for Whitening Layer Experiments

This script analyzes experiment results, computes statistics, 
performs significance tests, and generates visualizations.

Usage:
    python analyze_results.py                           # Analyze all results
    python analyze_results.py --compare baseline Q1_kernel_3x3  # Compare specific experiments
"""

import os
import json
import argparse
from glob import glob
from datetime import datetime

import torch
import numpy as np
from scipy import stats

# Optional: for plotting (install with: pip install matplotlib)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting disabled.")


#############################################
#           Load Experiment Results         #
#############################################

def load_all_results(log_dir='experiment_logs'):
    """Load all experiment results from the log directory."""
    results = {}
    
    # Find all result files
    pattern = os.path.join(log_dir, '*', 'results.pt')
    result_files = glob(pattern)
    
    for filepath in result_files:
        try:
            data = torch.load(filepath)
            exp_name = data.get('experiment_name', os.path.basename(os.path.dirname(filepath)))
            results[exp_name] = data
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return results


def load_specific_results(experiment_names, log_dir='experiment_logs'):
    """Load results for specific experiments."""
    all_results = load_all_results(log_dir)
    return {name: all_results[name] for name in experiment_names if name in all_results}


#############################################
#           Statistical Analysis            #
#############################################

def compute_statistics(accuracies, times):
    """Compute comprehensive statistics for a set of runs."""
    acc_arr = np.array(accuracies)
    time_arr = np.array(times)
    n = len(acc_arr)
    
    return {
        'accuracy': {
            'n': n,
            'mean': np.mean(acc_arr),
            'std': np.std(acc_arr, ddof=1),
            'sem': np.std(acc_arr, ddof=1) / np.sqrt(n),  # Standard error of mean
            'ci_95': 1.96 * np.std(acc_arr, ddof=1) / np.sqrt(n),
            'min': np.min(acc_arr),
            'max': np.max(acc_arr),
            'median': np.median(acc_arr),
            'q25': np.percentile(acc_arr, 25),
            'q75': np.percentile(acc_arr, 75),
        },
        'time': {
            'n': n,
            'mean': np.mean(time_arr),
            'std': np.std(time_arr, ddof=1),
            'sem': np.std(time_arr, ddof=1) / np.sqrt(n),
            'ci_95': 1.96 * np.std(time_arr, ddof=1) / np.sqrt(n),
            'min': np.min(time_arr),
            'max': np.max(time_arr),
            'median': np.median(time_arr),
        }
    }


def welch_ttest(data1, data2):
    """
    Perform Welch's t-test (t-test with unequal variances).
    
    Returns t-statistic, p-value, and effect size (Cohen's d).
    """
    arr1 = np.array(data1)
    arr2 = np.array(data2)
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.var(arr1, ddof=1) + np.var(arr2, ddof=1)) / 2)
    cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
    }


def compare_experiments(baseline_result, test_result):
    """
    Compare a test experiment against a baseline.
    
    Returns detailed comparison statistics.
    """
    baseline_acc = baseline_result['all_accuracies']
    baseline_time = baseline_result['all_times']
    test_acc = test_result['all_accuracies']
    test_time = test_result['all_times']
    
    baseline_stats = compute_statistics(baseline_acc, baseline_time)
    test_stats = compute_statistics(test_acc, test_time)
    
    # Statistical tests
    acc_test = welch_ttest(test_acc, baseline_acc)
    time_test = welch_ttest(test_time, baseline_time)
    
    # Compute deltas
    delta_acc = test_stats['accuracy']['mean'] - baseline_stats['accuracy']['mean']
    delta_time = test_stats['time']['mean'] - baseline_stats['time']['mean']
    
    return {
        'baseline': baseline_stats,
        'test': test_stats,
        'delta_accuracy': delta_acc,
        'delta_time': delta_time,
        'accuracy_test': acc_test,
        'time_test': time_test,
    }


#############################################
#              Visualizations               #
#############################################

def plot_accuracy_comparison(results, output_path=None):
    """Create a box plot comparing accuracy across experiments."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    data = [results[name]['all_accuracies'] for name in names]
    
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    
    # Color the baseline differently
    for i, name in enumerate(names):
        if 'baseline' in name.lower():
            bp['boxes'][i].set_facecolor('lightblue')
        else:
            bp['boxes'][i].set_facecolor('lightgreen')
    
    ax.set_ylabel('TTA Accuracy')
    ax.set_title('Accuracy Distribution by Experiment')
    ax.tick_params(axis='x', rotation=45)
    
    # Add horizontal line at baseline mean
    if 'baseline' in results:
        baseline_mean = np.mean(results['baseline']['all_accuracies'])
        ax.axhline(y=baseline_mean, color='blue', linestyle='--', alpha=0.5, label=f'Baseline mean: {baseline_mean:.4f}')
        ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_time_comparison(results, output_path=None):
    """Create a box plot comparing training time across experiments."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    data = [results[name]['all_times'] for name in names]
    
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    
    for i, name in enumerate(names):
        if 'baseline' in name.lower():
            bp['boxes'][i].set_facecolor('lightblue')
        else:
            bp['boxes'][i].set_facecolor('lightyellow')
    
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Distribution by Experiment')
    ax.tick_params(axis='x', rotation=45)
    
    if 'baseline' in results:
        baseline_mean = np.mean(results['baseline']['all_times'])
        ax.axhline(y=baseline_mean, color='blue', linestyle='--', alpha=0.5, label=f'Baseline mean: {baseline_mean:.3f}s')
        ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_accuracy_vs_time(results, output_path=None):
    """Create a scatter plot of accuracy vs time with error bars."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, result in results.items():
        acc_mean = np.mean(result['all_accuracies'])
        acc_std = np.std(result['all_accuracies'])
        time_mean = np.mean(result['all_times'])
        time_std = np.std(result['all_times'])
        
        color = 'blue' if 'baseline' in name.lower() else 'green'
        marker = 's' if 'baseline' in name.lower() else 'o'
        
        ax.errorbar(time_mean, acc_mean, 
                   xerr=time_std, yerr=acc_std,
                   fmt=marker, markersize=10, capsize=5,
                   color=color, label=name)
    
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('TTA Accuracy')
    ax.set_title('Accuracy vs Training Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(results, output_path=None):
    """Plot training curves (val accuracy over epochs) for all experiments."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, result in results.items():
        # Average across runs
        all_curves = []
        for run_result in result['all_results']:
            val_accs = [e['val_acc'] for e in run_result['epoch_results']]
            all_curves.append(val_accs)
        
        # Compute mean and std
        curves_arr = np.array(all_curves)
        mean_curve = np.mean(curves_arr, axis=0)
        std_curve = np.std(curves_arr, axis=0)
        epochs = np.arange(len(mean_curve))
        
        line, = ax.plot(epochs, mean_curve, label=name, linewidth=2)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                       alpha=0.2, color=line.get_color())
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Training Curves (mean +- std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


#############################################
#              Report Generation            #
#############################################

def generate_comparison_table(results, baseline_name='baseline'):
    """Generate a formatted comparison table."""
    baseline = results.get(baseline_name)
    
    print(f"\n{'='*130}")
    print("EXPERIMENT COMPARISON TABLE")
    print(f"{'='*130}")
    print(f"{'Experiment':<30} {'Accuracy':<20} {'Time (s)':<18} {'Delta Acc':<10} {'Acc p':<10} {'Delta Time':<10} {'Time p':<10} {'Sig?':<6}")
    print(f"{'-'*130}")
    
    for name, result in sorted(results.items()):
        acc_mean = np.mean(result['all_accuracies'])
        acc_std = np.std(result['all_accuracies'])
        time_mean = np.mean(result['all_times'])
        time_std = np.std(result['all_times'])
        
        acc_str = f"{acc_mean:.4f} +- {acc_std:.4f}"
        time_str = f"{time_mean:.3f} +- {time_std:.3f}"
        
        if baseline and name != baseline_name:
            comparison = compare_experiments(baseline, result)
            delta_acc = comparison['delta_accuracy']
            delta_time = comparison['delta_time']
            acc_p = comparison['accuracy_test']['p_value']
            time_p = comparison['time_test']['p_value']
            
            # Significance markers (combine accuracy and time)
            acc_sig = "A" if acc_p < 0.05 else ""
            time_sig = "T" if time_p < 0.05 else ""
            sig = acc_sig + time_sig
            if acc_p < 0.01: sig = sig.replace("A", "A*")
            if time_p < 0.01: sig = sig.replace("T", "T*")
            
            delta_acc_str = f"{delta_acc:+.4f}"
            delta_time_str = f"{delta_time:+.3f}"
            acc_p_str = f"{acc_p:.4f}"
            time_p_str = f"{time_p:.4f}"
        else:
            delta_acc_str = "---"
            delta_time_str = "---"
            acc_p_str = "---"
            time_p_str = "---"
            sig = ""
        
        print(f"{name:<30} {acc_str:<20} {time_str:<18} {delta_acc_str:<10} {acc_p_str:<10} {delta_time_str:<10} {time_p_str:<10} {sig:<6}")
    
    print(f"{'='*130}")
    print("Significance: A=accuracy p<0.05, T=time p<0.05, *=p<0.01")
    print()


def generate_detailed_report(results, baseline_name='baseline', output_path=None):
    """Generate a detailed analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("DETAILED EXPERIMENT ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 80)
    lines.append("")
    
    baseline = results.get(baseline_name)
    
    for name, result in sorted(results.items()):
        lines.append(f"\n{'='*60}")
        lines.append(f"EXPERIMENT: {name}")
        lines.append(f"{'='*60}")
        
        # Config
        lines.append(f"\nConfiguration:")
        config = result.get('config', {})
        if config:
            for key, value in config.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  (baseline - no modifications)")
        
        # Statistics
        stats_data = compute_statistics(result['all_accuracies'], result['all_times'])
        
        lines.append(f"\nAccuracy Statistics (n={stats_data['accuracy']['n']}):")
        lines.append(f"  Mean:   {stats_data['accuracy']['mean']:.4f}")
        lines.append(f"  Std:    {stats_data['accuracy']['std']:.4f}")
        lines.append(f"  95% CI: +- {stats_data['accuracy']['ci_95']:.4f}")
        lines.append(f"  Range:  [{stats_data['accuracy']['min']:.4f}, {stats_data['accuracy']['max']:.4f}]")
        lines.append(f"  Median: {stats_data['accuracy']['median']:.4f}")
        
        lines.append(f"\nTime Statistics:")
        lines.append(f"  Mean:   {stats_data['time']['mean']:.3f} s")
        lines.append(f"  Std:    {stats_data['time']['std']:.3f} s")
        lines.append(f"  Range:  [{stats_data['time']['min']:.3f}, {stats_data['time']['max']:.3f}] s")
        
        # Comparison with baseline
        if baseline and name != baseline_name:
            comparison = compare_experiments(baseline, result)
            lines.append(f"\nComparison with Baseline:")
            lines.append(f"  Delta Accuracy:  {comparison['delta_accuracy']:+.4f}")
            lines.append(f"  Delta Time:      {comparison['delta_time']:+.3f} s")
            lines.append(f"\n  Accuracy t-test:")
            lines.append(f"    t-statistic: {comparison['accuracy_test']['t_statistic']:.4f}")
            lines.append(f"    p-value:     {comparison['accuracy_test']['p_value']:.6f}")
            lines.append(f"    Cohen's d:   {comparison['accuracy_test']['cohens_d']:.4f}")
            lines.append(f"    Significant (p<0.05): {comparison['accuracy_test']['significant_0.05']}")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    else:
        print(report)
    
    return report


#############################################
#              Main Entry Point             #
#############################################

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--compare', nargs='+', help='Compare specific experiments')
    parser.add_argument('--baseline', default='baseline', help='Name of baseline experiment')
    parser.add_argument('--output_dir', default='analysis_output', help='Directory for output files')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Load results
    if args.compare:
        results = load_specific_results(args.compare)
    else:
        results = load_all_results()
    
    if not results:
        print("No results found. Run experiments first with run_experiments.py")
        return
    
    print(f"Loaded {len(results)} experiments: {list(results.keys())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate comparison table
    generate_comparison_table(results, baseline_name=args.baseline)
    
    # Generate detailed report
    report_path = os.path.join(args.output_dir, 'detailed_report.txt')
    generate_detailed_report(results, baseline_name=args.baseline, output_path=report_path)
    
    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        
        plot_accuracy_comparison(
            results, 
            output_path=os.path.join(args.output_dir, 'accuracy_comparison.png')
        )
        
        plot_time_comparison(
            results,
            output_path=os.path.join(args.output_dir, 'time_comparison.png')
        )
        
        plot_accuracy_vs_time(
            results,
            output_path=os.path.join(args.output_dir, 'accuracy_vs_time.png')
        )
        
        plot_training_curves(
            results,
            output_path=os.path.join(args.output_dir, 'training_curves.png')
        )
    
    print(f"\nAnalysis complete. Output saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()