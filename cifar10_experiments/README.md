# Whitening Layer Experiment Framework

This framework allows you to systematically test modifications to the whitening layer
in the CIFAR-10 airbench94 speedrun code.

## Hypothesis

**Q: The whitening layer's architecture (kernel size, width, and trainability) can be 
modified to achieve faster convergence to 94% accuracy.**

### Sub-hypotheses:
- **Q1**: Kernel size affects convergence (test 1×1, 2×2, 3×3, 4×4)
- **Q2**: Width multiplier affects efficiency (test 1, 2, 3)
- **Q3**: Epsilon regularization affects training stability (test 5e-5 to 5e-3)
- **Q4**: Making whitening weights trainable may help adaptation (test unfreezing at various epochs)

## Files

| File | Description |
|------|-------------|
| `airbench94_configurable.py` | Modified airbench94 with configurable whitening parameters |
| `run_experiments.py` | Defines and runs all experiments |
| `analyze_results.py` | Analyzes results, computes statistics, generates plots |

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib scipy numpy
```

### 2. List all experiments

```bash
python run_experiments.py --list
```

### 3. Run experiments

```bash
# Run all experiments (takes ~2 hours on A100)
python run_experiments.py

# Or run by phase:
python run_experiments.py --phase baseline   # ~7 min
python run_experiments.py --phase Q1         # ~7 min
python run_experiments.py --phase Q2         # ~3 min
python run_experiments.py --phase Q3         # ~7 min
python run_experiments.py --phase Q4         # ~7 min

# Or run a specific experiment:
python run_experiments.py --experiment Q1_kernel_3x3
```

### 4. Analyze results

```bash
python analyze_results.py
```

This generates:
- Comparison table with statistical significance
- Detailed report (`analysis_output/detailed_report.txt`)
- Plots (`analysis_output/*.png`)

## Running Individual Experiments via Command Line

You can also run the configurable airbench directly:

```bash
# Baseline (default settings)
python airbench94_configurable.py --num_runs 50 --experiment_name baseline

# Test 3×3 kernel
python airbench94_configurable.py --num_runs 50 --whiten_kernel_size 3 --experiment_name Q1_3x3

# Test width multiplier = 1
python airbench94_configurable.py --num_runs 50 --whiten_width_mult 1 --experiment_name Q2_width1

# Test epsilon = 1e-3
python airbench94_configurable.py --num_runs 50 --whiten_eps 1e-3 --experiment_name Q3_eps1e-3

# Test trainable whitening from epoch 3
python airbench94_configurable.py --num_runs 50 --whiten_trainable_after 3 --experiment_name Q4_train3
```

## Experiment Design

### Recommended Workflow

1. **Run baseline** (100 runs) - Establish reference accuracy and time
2. **Run Q1-Q4 ablations** (50 runs each) - Test each dimension independently
3. **Analyze results** - Identify promising modifications
4. **Run combinations** (50 runs) - Combine best settings from each dimension
5. **Final validation** (100 runs) - Confirm results with more runs

### Statistical Analysis

The framework computes:
- Mean ± standard deviation
- 95% confidence intervals
- Welch's t-test (p-values)
- Cohen's d (effect size)

### Output Structure

```
experiment_logs/
├── baseline_abc12345/
│   ├── results.json    # Human-readable summary
│   └── results.pt      # Full PyTorch data
├── Q1_kernel_3x3_def67890/
│   ├── results.json
│   └── results.pt
└── summary_20241208_120000.json  # Cross-experiment summary

analysis_output/
├── detailed_report.txt
├── accuracy_comparison.png
├── time_comparison.png
├── accuracy_vs_time.png
└── training_curves.png
```

## Modifying Experiments

### Adding new experiments

Edit `run_experiments.py` and add to `get_all_experiments()`:

```python
experiments['Q1_kernel_5x5'] = {
    'config': {'whiten': {'kernel_size': 5}},
    'num_runs': 50,
    'description': 'Q1: Whitening kernel size 5x5',
    'phase': 'Q1',
}
```

### Adjusting number of runs

Edit the constants at the top of `run_experiments.py`:

```python
RUNS_BASELINE = 100      # Baseline runs
RUNS_ABLATION = 50       # Ablation runs
RUNS_FINAL = 100         # Final validation runs
```

## Tips for Your Report

1. **Always compare to baseline** - Report Δ accuracy and Δ time
2. **Report p-values** - Use Welch's t-test for significance
3. **Show confidence intervals** - 95% CI = mean ± 1.96 × SEM
4. **Include training curves** - Show how modifications affect learning dynamics
5. **Discuss trade-offs** - Some changes may improve accuracy but slow training

## Example Results Table

| Experiment | Accuracy | Time (s) | Δ Acc | p-value |
|------------|----------|----------|-------|---------|
| baseline | 0.9401 ± 0.0021 | 3.83 ± 0.05 | --- | --- |
| Q1_kernel_3x3 | 0.9412 ± 0.0019 | 3.91 ± 0.06 | +0.0011 | 0.023* |
| Q2_width_1 | 0.9389 ± 0.0024 | 3.71 ± 0.04 | -0.0012 | 0.041* |
| Q3_eps_1e-3 | 0.9405 ± 0.0020 | 3.84 ± 0.05 | +0.0004 | 0.312 |

*Significance: * p<0.05, ** p<0.01, *** p<0.001*