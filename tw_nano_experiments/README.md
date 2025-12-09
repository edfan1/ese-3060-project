# List Phase 1 experiments without running
python run_ablations.py --phase 1 --dry-run

# Run all Phase 1 baselines (full runs, 5 seeds)
python run_ablations.py --phase 1

# Run Phase 2 screening (500 steps, quick elimination)
python run_ablations.py --phase 2 --stage screening

# After analyzing Phase 2 results, run Phase 3 with best function
python run_ablations.py --phase 3 --stage screening --best-function sqrt

# Run Phase 4 with best function + clamps from Phase 3
python run_ablations.py --phase 4 --stage screening \
    --best-function sqrt --best-clamp-min 0.7 --best-clamp-max 1.5

# Final Phase 5 validation
python run_ablations.py --phase 5 \
    --best-function sqrt --best-clamp-min 0.7 --best-clamp-max 1.5 \
    --best-schedule warmup

# Run single experiment by name
python run_ablations.py --experiment p2_linear_seed42

# Use fewer GPUs
python run_ablations.py --phase 1 --gpus 4
```

**Output Structure:**
```
ablation_results/
├── experiment_status.json    # Tracks all experiment status
├── phase1/
│   └── full/
│       ├── baseline_seed42/
│       │   ├── config.json
│       │   └── run.log
│       └── ...
├── phase2/
│   ├── screening/
│   └── full/
└── ...