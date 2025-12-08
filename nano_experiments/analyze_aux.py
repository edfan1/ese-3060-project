#!/usr/bin/env python3
"""
Analysis Script for Auxiliary Head Experiments

This script analyzes experiment results, computes statistics, 
performs significance tests, and generates visualizations.

Usage:
    python analyze_aux.py --exp_dir experiments/aux_heads_XXXXXX
    python analyze_aux.py --exp_dir experiments/aux_heads_XXXXXX --compare baseline aux_4_8
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics

import numpy as np
from scipy import stats

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting disabled.")
    print("Install with: pip install matplotlib")


# =============================================================================
# Data Loading
# =============================================================================

def load_experiment_results(exp_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all experiment results from a directory.
    
    Returns dict mapping experiment names to list of run results.
    """
    results = defaultdict(list)
    
    # Look for result.json files in experiment subdirectories
    for exp_path in exp_dir.iterdir():
        if not exp_path.is_dir():
            continue
        
        exp_name = exp_path.name
        
        # Each experiment can have multiple runs
        for run_path in exp_path.iterdir():
            if not run_path.is_dir():
                continue
            
            result_file = run_path / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
                    results[exp_name].append(result)
    
    return dict(results)


def load_config(exp_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load experiment configurations."""
    config_file = exp_dir / "experiment_configs.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics for a list of values."""
    if not values:
        return {}
    
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return {}
    
    n = len(values)
    mean = np.mean(values)
    
    if n > 1:
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(n)
        ci_95 = 1.96 * sem
    else:
        std = 0.0
        sem = 0.0
        ci_95 = 0.0
    
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_95": ci_95,
        "min": min(values),
        "max": max(values),
        "median": np.median(values),
    }


def welch_ttest(data1: List[float], data2: List[float]) -> Dict[str, float]:
    """
    Perform Welch's t-test (unequal variances).
    
    Returns t-statistic, p-value, and effect size (Cohen's d).
    """
    arr1 = np.array([x for x in data1 if not np.isnan(x)])
    arr2 = np.array([x for x in data2 if not np.isnan(x)])
    
    if len(arr1) < 2 or len(arr2) < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": 1.0,
            "cohens_d": 0.0,
            "significant_0.05": False,
            "significant_0.01": False,
        }
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.var(arr1, ddof=1) + np.var(arr2, ddof=1)) / 2)
    cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant_0.05": p_value < 0.05,
        "significant_0.01": p_value < 0.01,
    }


def analyze_experiment(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze results from a single experiment (possibly multiple runs)."""
    val_losses = [r["final_val_loss"] for r in results if r.get("success", False)]
    train_times = [r["training_time_ms"] for r in results if r.get("success", False)]
    step_avgs = [r["step_avg_ms"] for r in results if r.get("success", False)]
    
    return {
        "num_runs": len(results),
        "num_successful": len(val_losses),
        "val_loss": compute_statistics(val_losses),
        "training_time_ms": compute_statistics(train_times),
        "step_avg_ms": compute_statistics(step_avgs),
        "raw_val_losses": val_losses,
        "raw_training_times": train_times,
    }


def compare_to_baseline(
    baseline_results: List[Dict[str, Any]],
    test_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare an experiment to baseline."""
    baseline_analysis = analyze_experiment(baseline_results)
    test_analysis = analyze_experiment(test_results)
    
    # Statistical tests
    val_loss_test = welch_ttest(
        baseline_analysis["raw_val_losses"],
        test_analysis["raw_val_losses"],
    )
    
    time_test = welch_ttest(
        baseline_analysis["raw_training_times"],
        test_analysis["raw_training_times"],
    )
    
    # Compute deltas
    baseline_val = baseline_analysis["val_loss"].get("mean", float("nan"))
    test_val = test_analysis["val_loss"].get("mean", float("nan"))
    delta_val = test_val - baseline_val
    
    baseline_time = baseline_analysis["training_time_ms"].get("mean", float("nan"))
    test_time = test_analysis["training_time_ms"].get("mean", float("nan"))
    delta_time = test_time - baseline_time
    
    return {
        "baseline": baseline_analysis,
        "test": test_analysis,
        "delta_val_loss": delta_val,
        "delta_time_ms": delta_time,
        "val_loss_test": val_loss_test,
        "time_test": time_test,
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_comparison_table(
    results: Dict[str, List[Dict[str, Any]]],
    configs: Dict[str, Dict[str, Any]],
    baseline_name: str = "baseline",
) -> str:
    """Generate a formatted comparison table."""
    lines = []
    lines.append("=" * 120)
    lines.append("EXPERIMENT COMPARISON TABLE")
    lines.append("=" * 120)
    
    header = (
        f"{'Experiment':<25} "
        f"{'Aux Layers':<12} "
        f"{'Val Loss':<18} "
        f"{'Time (s)':<15} "
        f"{'Δ Val Loss':<12} "
        f"{'Δ Time':<10} "
        f"{'Sig?':<6}"
    )
    lines.append(header)
    lines.append("-" * 120)
    
    baseline_results = results.get(baseline_name, [])
    baseline_analysis = analyze_experiment(baseline_results) if baseline_results else None
    
    for exp_name, exp_results in sorted(results.items()):
        analysis = analyze_experiment(exp_results)
        config = configs.get(exp_name, {})
        
        aux_layers = config.get("aux_head_layers", "none") or "none"
        
        val_mean = analysis["val_loss"].get("mean", float("nan"))
        val_std = analysis["val_loss"].get("std", 0)
        val_str = f"{val_mean:.4f} ± {val_std:.4f}"
        
        time_mean = analysis["training_time_ms"].get("mean", float("nan")) / 1000
        time_std = analysis["training_time_ms"].get("std", 0) / 1000
        time_str = f"{time_mean:.1f} ± {time_std:.1f}"
        
        if baseline_analysis and exp_name != baseline_name:
            comparison = compare_to_baseline(baseline_results, exp_results)
            delta_val = comparison["delta_val_loss"]
            delta_time = comparison["delta_time_ms"] / 1000
            
            delta_val_str = f"{delta_val:+.4f}"
            delta_time_str = f"{delta_time:+.1f}s"
            
            # Significance markers
            sig_marks = []
            if comparison["val_loss_test"]["significant_0.05"]:
                sig_marks.append("V" if comparison["val_loss_test"]["significant_0.01"] else "v")
            if comparison["time_test"]["significant_0.05"]:
                sig_marks.append("T" if comparison["time_test"]["significant_0.01"] else "t")
            sig_str = "".join(sig_marks) or "-"
        else:
            delta_val_str = "---"
            delta_time_str = "---"
            sig_str = "BASE"
        
        line = (
            f"{exp_name:<25} "
            f"{aux_layers:<12} "
            f"{val_str:<18} "
            f"{time_str:<15} "
            f"{delta_val_str:<12} "
            f"{delta_time_str:<10} "
            f"{sig_str:<6}"
        )
        lines.append(line)
    
    lines.append("=" * 120)
    lines.append("Significance: V/v=val_loss (p<0.01/0.05), T/t=time (p<0.01/0.05)")
    lines.append("")
    
    return "\n".join(lines)


def generate_phase_analysis(
    results: Dict[str, List[Dict[str, Any]]],
    configs: Dict[str, Dict[str, Any]],
) -> str:
    """Generate analysis grouped by experimental phase."""
    lines = []
    
    # Group experiments by phase
    phases = defaultdict(list)
    for exp_name, exp_results in results.items():
        config = configs.get(exp_name, {})
        phase = config.get("phase", "unknown")
        phases[phase].append((exp_name, exp_results, config))
    
    for phase, experiments in sorted(phases.items()):
        lines.append(f"\n{'='*60}")
        lines.append(f"PHASE: {phase.upper()}")
        lines.append(f"{'='*60}")
        
        # Analyze each experiment in this phase
        for exp_name, exp_results, config in experiments:
            analysis = analyze_experiment(exp_results)
            
            lines.append(f"\n{exp_name}:")
            lines.append(f"  Description: {config.get('description', 'N/A')}")
            lines.append(f"  Aux layers: {config.get('aux_head_layers', 'none') or 'none'}")
            lines.append(f"  Aux weight: {config.get('aux_loss_weight', 0)}")
            lines.append(f"  Runs: {analysis['num_successful']}/{analysis['num_runs']}")
            
            if analysis["val_loss"]:
                lines.append(f"  Val loss: {analysis['val_loss']['mean']:.4f} ± {analysis['val_loss']['std']:.4f}")
                lines.append(f"    Range: [{analysis['val_loss']['min']:.4f}, {analysis['val_loss']['max']:.4f}]")
            
            if analysis["training_time_ms"]:
                time_s = analysis["training_time_ms"]["mean"] / 1000
                time_std_s = analysis["training_time_ms"]["std"] / 1000
                lines.append(f"  Time: {time_s:.1f} ± {time_std_s:.1f} seconds")
    
    return "\n".join(lines)


def generate_detailed_report(
    results: Dict[str, List[Dict[str, Any]]],
    configs: Dict[str, Dict[str, Any]],
    baseline_name: str = "baseline",
) -> str:
    """Generate a comprehensive analysis report."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("AUXILIARY PREDICTION HEAD EXPERIMENT ANALYSIS")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 80)
    
    # Overall summary
    lines.append("\n## SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total experiments: {len(results)}")
    total_runs = sum(len(r) for r in results.values())
    lines.append(f"Total runs: {total_runs}")
    
    # Add comparison table
    lines.append("\n")
    lines.append(generate_comparison_table(results, configs, baseline_name))
    
    # Add phase analysis
    lines.append(generate_phase_analysis(results, configs))
    
    # Detailed baseline comparison for each experiment
    baseline_results = results.get(baseline_name, [])
    if baseline_results:
        lines.append(f"\n{'='*60}")
        lines.append("DETAILED COMPARISONS TO BASELINE")
        lines.append(f"{'='*60}")
        
        for exp_name, exp_results in sorted(results.items()):
            if exp_name == baseline_name:
                continue
            
            comparison = compare_to_baseline(baseline_results, exp_results)
            
            lines.append(f"\n## {exp_name} vs baseline")
            lines.append("-" * 40)
            
            # Val loss comparison
            lines.append(f"Val Loss Delta: {comparison['delta_val_loss']:+.4f}")
            lines.append(f"  t-statistic: {comparison['val_loss_test']['t_statistic']:.4f}")
            lines.append(f"  p-value: {comparison['val_loss_test']['p_value']:.6f}")
            lines.append(f"  Cohen's d: {comparison['val_loss_test']['cohens_d']:.4f}")
            lines.append(f"  Significant (p<0.05): {comparison['val_loss_test']['significant_0.05']}")
            
            # Time comparison
            lines.append(f"\nTime Delta: {comparison['delta_time_ms']/1000:+.1f}s")
            lines.append(f"  t-statistic: {comparison['time_test']['t_statistic']:.4f}")
            lines.append(f"  p-value: {comparison['time_test']['p_value']:.6f}")
            
            # Interpretation
            if comparison["delta_val_loss"] < 0 and comparison["val_loss_test"]["significant_0.05"]:
                lines.append("\n  → IMPROVEMENT: Lower val loss (statistically significant)")
            elif comparison["delta_val_loss"] > 0 and comparison["val_loss_test"]["significant_0.05"]:
                lines.append("\n  → REGRESSION: Higher val loss (statistically significant)")
            else:
                lines.append("\n  → NO SIGNIFICANT DIFFERENCE in val loss")
    
    return "\n".join(lines)


# =============================================================================
# Visualization
# =============================================================================

def plot_val_loss_comparison(
    results: Dict[str, List[Dict[str, Any]]],
    configs: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
):
    """Create a bar chart comparing final validation losses."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    # Prepare data
    experiments = []
    means = []
    stds = []
    colors = []
    
    for exp_name in sorted(results.keys()):
        analysis = analyze_experiment(results[exp_name])
        if not analysis["val_loss"]:
            continue
        
        experiments.append(exp_name)
        means.append(analysis["val_loss"]["mean"])
        stds.append(analysis["val_loss"]["std"])
        
        # Color based on phase
        phase = configs.get(exp_name, {}).get("phase", "unknown")
        color_map = {
            "baseline": "steelblue",
            "layer_position": "forestgreen",
            "num_heads": "darkorange",
            "loss_weight": "mediumpurple",
            "loss_schedule": "crimson",
            "best": "gold",
            "quick": "gray",
        }
        colors.append(color_map.get(phase, "gray"))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(experiments))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
    
    ax.set_ylabel("Final Validation Loss")
    ax.set_xlabel("Experiment")
    ax.set_title("Validation Loss Comparison Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    
    # Add baseline reference line
    if "baseline" in results:
        baseline_mean = analyze_experiment(results["baseline"])["val_loss"]["mean"]
        ax.axhline(y=baseline_mean, color="steelblue", linestyle="--", alpha=0.7, 
                   label=f"Baseline: {baseline_mean:.4f}")
        ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    results: Dict[str, List[Dict[str, Any]]],
    experiments_to_plot: List[str],
    output_path: Optional[Path] = None,
):
    """Plot validation loss curves over training."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_to_plot)))
    
    for exp_name, color in zip(experiments_to_plot, colors):
        if exp_name not in results:
            continue
        
        # Average across runs
        all_curves = []
        for result in results[exp_name]:
            if result.get("val_loss_history"):
                all_curves.append(result["val_loss_history"])
        
        if not all_curves:
            continue
        
        # Align curves to same length (use minimum)
        min_len = min(len(c) for c in all_curves)
        aligned = [c[:min_len] for c in all_curves]
        
        mean_curve = np.mean(aligned, axis=0)
        std_curve = np.std(aligned, axis=0)
        steps = np.arange(len(mean_curve)) * 125  # Assuming val_loss_every=125
        
        ax.plot(steps, mean_curve, label=exp_name, color=color, linewidth=2)
        ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color=color)
    
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Training Curves (mean ± std)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_layer_ablation(
    results: Dict[str, List[Dict[str, Any]]],
    configs: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
):
    """Plot results of layer position ablation."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    # Extract single-layer experiments
    layer_exps = {}
    for exp_name, exp_results in results.items():
        config = configs.get(exp_name, {})
        layers_str = config.get("aux_head_layers", "")
        
        if layers_str and "," not in layers_str:  # Single layer only
            try:
                layer = int(layers_str)
                analysis = analyze_experiment(exp_results)
                if analysis["val_loss"]:
                    layer_exps[layer] = analysis
            except ValueError:
                pass
    
    if not layer_exps:
        print("No single-layer experiments found")
        return
    
    # Get baseline
    baseline_analysis = None
    if "baseline" in results:
        baseline_analysis = analyze_experiment(results["baseline"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = sorted(layer_exps.keys())
    means = [layer_exps[l]["val_loss"]["mean"] for l in layers]
    stds = [layer_exps[l]["val_loss"]["std"] for l in layers]
    
    ax.errorbar(layers, means, yerr=stds, fmt="o-", capsize=5, markersize=8,
                color="forestgreen", linewidth=2, label="Aux head at layer")
    
    if baseline_analysis and baseline_analysis["val_loss"]:
        baseline_mean = baseline_analysis["val_loss"]["mean"]
        baseline_std = baseline_analysis["val_loss"]["std"]
        ax.axhline(y=baseline_mean, color="steelblue", linestyle="--",
                   label=f"Baseline: {baseline_mean:.4f}")
        ax.fill_between(ax.get_xlim(), baseline_mean - baseline_std, 
                        baseline_mean + baseline_std, alpha=0.2, color="steelblue")
    
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("Effect of Auxiliary Head Position")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze auxiliary head experiment results")
    
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline",
        help="Name of baseline experiment",
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Specific experiments to compare",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output files (default: same as exp_dir)",
    )
    
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip generating plots",
    )
    
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Error: Directory not found: {exp_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from: {exp_dir}")
    results = load_experiment_results(exp_dir)
    configs = load_config(exp_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} experiments")
    
    # Filter if specific experiments requested
    if args.compare:
        results = {k: v for k, v in results.items() if k in args.compare}
        if not results:
            print(f"None of the requested experiments found: {args.compare}")
            return
    
    # Generate comparison table
    print("\n")
    table = generate_comparison_table(results, configs, args.baseline)
    print(table)
    
    # Generate detailed report
    report = generate_detailed_report(results, configs, args.baseline)
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        
        plot_val_loss_comparison(
            results, configs,
            output_path=output_dir / "val_loss_comparison.png"
        )
        
        plot_layer_ablation(
            results, configs,
            output_path=output_dir / "layer_ablation.png"
        )
        
        # Plot training curves for key experiments
        key_experiments = ["baseline", "aux_4", "aux_8", "aux_4_8", "aux_3_6_9"]
        key_experiments = [e for e in key_experiments if e in results]
        if key_experiments:
            plot_training_curves(
                results, key_experiments,
                output_path=output_dir / "training_curves.png"
            )
    
    print(f"\nAnalysis complete! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()