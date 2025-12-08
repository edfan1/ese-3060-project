# Staged Ablation Strategy for Auxiliary Heads

## Overview

This directory contains a compute-efficient staged ablation framework for testing the **Auxiliary Prediction Heads** hypothesis on NanoGPT.

**Hypothesis**: Adding lightweight auxiliary prediction heads at intermediate layers (e.g., layers 4 and 8) provides additional gradient signal to early/middle layers, accelerating learning.

**Compute Budget**: ~7 hours on 8xH100 GPUs  
**Full Run Time**: ~15 minutes per experiment  
**Maximum Experiments**: ~28 full-length runs

## Strategy

We use a 4-stage approach with automatic decision points to maximize statistical power within the compute budget:

### Stage 1: Quick Validation (1 hour, ~4 runs)

**Goal**: Verify hypothesis has merit before committing full compute

- Runs at 1000 iterations (~3 min each) instead of full 5100 iterations
- Tests: baseline, aux_6, aux_4_8, aux_3_6_9
- **Decision criteria**: If ANY aux config shows >2% improvement → proceed to Stage 2

### Stage 2: Layer Position Screening (1.5 hours, ~6 runs)

**Goal**: Identify which layer positions benefit most from auxiliary heads

- Full-length runs (5100 iterations)
- Tests: baseline, aux_4, aux_6, aux_8, aux_4_8, aux_6_repeat
- **Decision**: Select best performing layer configuration

### Stage 3: Targeted Deep Dive (2.5 hours, ~10 runs)

**Goal**: Optimize loss weight and schedule for best layer config from Stage 2

- Tests loss weights: 0.05, 0.1, 0.2
- Tests schedules: constant, linear_decay, cosine_decay
- Includes 4 validation runs for variance estimation
- **Decision**: Finalize best configuration

### Stage 4: Final Validation (2 hours, ~8 runs)

**Goal**: Achieve statistical significance for claimed improvements

- 8 runs each of best config and baseline
- Statistical tests (t-test, confidence intervals)
- **Target**: p < 0.05 for improvements ≥10% speedup or ≥0.02 val loss improvement

## Quick Start

### Automated Staged Ablation (Recommended)

```bash
# Run complete staged ablation with automatic decision points
python run_aux.py --staged --max_budget_hours 7
```

This will:
1. Run Stage 1 screening
2. Automatically decide whether to proceed
3. If promising, run Stage 2 to find best layers
4. Run Stage 3 to optimize hyperparameters
5. Run Stage 4 for final validation with statistics

### Manual Stage-by-Stage

If you want more control:

```bash
# Stage 1: Quick validation
python run_aux.py --stage 1

# Check results
python staged_analysis_helper.py --stage 1 --exp_dir experiments/aux_heads_XXXXXX

# If promising, continue...
python run_aux.py --stage 2

# Analyze
python staged_analysis_helper.py --stage 2 --exp_dir experiments/aux_heads_XXXXXX

# Stage 3 & 4
python run_aux.py --stage 3
python run_aux.py --stage 4
```

### Legacy Mode (for custom experiments)

```bash
# List all available experiments
python run_aux.py --list

# Run specific experiment
python run_aux.py --experiment aux_4_8 --runs 5

# Run all experiments in a phase
python run_aux.py --phase screening --runs 1
```

## Expected Outputs

### During Execution

Each stage provides real-time decision support:

```
📊 STAGE 1 RESULTS:
  Baseline loss: 3.2845
  Best aux loss: 3.2612 (screen_aux_4_8)
  Improvement: +0.71%

❌ DECISION: STOP
  Insufficient improvement (0.71% < 2.0%)
  Hypothesis may not be valid for this architecture
```

Or if promising:

```
📊 STAGE 1 RESULTS:
  Baseline loss: 3.2845
  Best aux loss: 3.2156 (screen_aux_4_8)
  Improvement: +2.10%

✅ DECISION: PROCEED to Stage 2
  Aux heads show promise (2.10% improvement)
```

### Final Results

After Stage 4:

```
📊 STAGE 4 FINAL RESULTS:

  Best Configuration:
    Layers: 4,8
    Weight: 0.1
    Schedule: linear_decay

  Validation Loss:
    Baseline: 3.2845 ± 0.0124
    Best:     3.2612 ± 0.0098
    Improvement: +0.71%
    p-value: 0.002341
    Significant (p<0.05): True

  Training Time:
    Baseline: 234.2s ± 3.1s
    Best:     236.8s ± 2.9s
    Delta: +2.6s
    p-value: 0.084512

✅ CONCLUSION: Auxiliary heads provide SIGNIFICANT improvement
   Val loss improved by 0.71% (p=0.0023)
```

## Statistical Considerations

### Sample Size Requirements

For different effect sizes with p < 0.05:

| Effect Size | Required Runs Each | Total Runtime |
|-------------|-------------------|---------------|
| 10% speedup | 5-8 runs | ~2.5-4 hours |
| 5% speedup | 15-20 runs | ~7.5-10 hours |
| 2% speedup | 40-50 runs | ~20-25 hours |

**Our target**: ≥10% improvement (feasible within budget) or ≥0.02 val loss improvement

### Variance Control

- Stage 2 includes a repeat experiment (aux_6_repeat) to estimate run-to-run variance
- Stage 3 includes 4 validation runs to understand variance in best config
- Stage 4 uses multiple runs for robust statistical testing

## Decision Trees

### After Stage 1

```
Improvement > 2%?
  ├─ YES → Continue to Stage 2
  │        (Hypothesis shows promise)
  │
  └─ NO  → STOP and reconsider
           Options:
           - Try different layer positions
           - Try different loss weights
           - Consider alternative approaches
```

### After Stage 2

```
Found improved configuration?
  ├─ YES → Continue to Stage 3 with best layers
  │        (Optimize hyperparameters)
  │
  └─ NO  → Document null result
           (Well-reasoned negative result is valuable!)
```

## Files Generated

```
experiments/aux_heads_YYYYMMDD_HHMMSS/
├── staged_ablation_results.json          # All stage results
├── experiment_configs.json                # Experiment definitions
├── screen_baseline/                       # Stage 1 experiments
│   └── run_0/
│       ├── config.json
│       ├── result.json
│       ├── output.log
│       └── train_log.txt
├── screen_aux_6/
├── screen_aux_4_8/
├── screen_aux_3_6_9/
├── baseline/                              # Stage 2 experiments
├── aux_4/
├── aux_6/
├── aux_8/
├── aux_4_8/
├── dive_w0p05/                           # Stage 3 experiments
├── dive_w0p1/
├── dive_w0p2/
├── dive_constant/
├── dive_linear_decay/
├── dive_cosine_decay/
├── validation_0/
├── validation_1/
├── validation_2/
├── validation_3/
├── final_best/                           # Stage 4 experiments
│   ├── run_0/
│   ├── run_1/
│   └── ...
└── final_baseline/
    ├── run_0/
    ├── run_1/
    └── ...
```

## Analysis

### Quick Between-Stage Analysis

```bash
# After each stage
python run_aux.py --stage 1 --exp_dir experiments/aux_heads_XXXXXX
python run_aux.py --stage 2 --exp_dir experiments/aux_heads_XXXXXX
python run_aux.py --stage 3 --exp_dir experiments/aux_heads_XXXXXX
```

### Comprehensive Final Analysis

```bash
# After all stages complete
python analyze_aux.py --exp_dir experiments/aux_heads_XXXXXX
```

This generates:
- `analysis_report.txt`: Detailed statistical analysis
- `val_loss_comparison.png`: Bar chart of all experiments
- `layer_ablation.png`: Effect of layer position
- `training_curves.png`: Val loss over time

### Unexpected Results

If Stage 1 shows no improvement:
1. Check training logs for errors
2. Verify baseline achieves expected loss
3. Consider if hypothesis needs refinement

### Statistical Power Issues

If Stage 4 shows improvement but not significant (p > 0.05):
- Effect size may be smaller than expected
- More runs needed (but budget limited)
- Document as "promising but needs more validation"

## Theoretical Background

From the NanoGPT speedrun records, we see repeated efforts to improve gradient flow to early/middle layers:

- **Record #9**: Skip connections from embedding to every block
- **Record #11**: U-net pattern skip connections  
- **Record #40**: "Backout" - lets model reduce contributions from first 8 layers
- **Record #45**: "Refine Skip Arch"

The speedrun has been optimizing *implicit* gradient flow via skip connections. Auxiliary heads provide *explicit* deep supervision - a fundamentally different approach to the same problem.

### Why This Might Work

1. **Deep supervision works in vision**: U-Net, Inception, DenseNet all use auxiliary losses
2. **Gradient flow is a known bottleneck**: The speedrun's focus on skip connections confirms this
3. **Novel to transformers**: This technique hasn't been systematically tested on transformers

### Why This Might Not Work

1. **Skip connections may be sufficient**: Current architecture already has extensive gradient shortcuts
2. **Compute overhead**: Auxiliary heads add forward/backward cost
3. **Optimization challenges**: Multiple loss terms can destabilize training

## References

- NanoGPT Speedrun: https://github.com/KellerJordan/modded-nanogpt
- Deep Supervision in U-Net: Ronneberger et al. (2015)
- Auxiliary Classifiers in Inception: Szegedy et al. (2015)
- Project Description: `ESE_3060_Final_Project_1.pdf`