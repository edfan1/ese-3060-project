#!/usr/bin/env python3
"""
Token Importance Weighting Ablation Runner

This script manages the execution of ablation experiments for testing
token importance weighting in NanoGPT training.

Usage:
    # Run all Phase 1 baseline experiments
    python run_tw.py --phase 1

    # Run Phase 2 screening (short runs)
    python run_tw.py --phase 2 --stage screening

    # Run a specific experiment by name
    python run_tw.py --experiment baseline_seed42

    # List all experiments without running
    python run_tw.py --phase 2 --dry-run

    # Run with custom GPU count
    python run_tw.py --phase 1 --gpus 4
"""

import os
import sys
import json
import argparse
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import hashlib


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    description: str
    phase: int
    stage: str  # 'screening', 'medium', 'full'
    
    # Training parameters
    seed: int = 42
    num_iterations: int = 5100
    
    # Token weighting parameters
    tw_enabled: bool = False
    tw_function: str = "linear"
    tw_clamp_min: float = 0.5
    tw_clamp_max: float = 2.0
    tw_schedule: str = "constant"
    tw_warmup_steps: int = 1000
    tw_anneal_final: float = 0.3
    tw_cyclical_period: int = 500
    tw_focal_gamma: float = 2.0
    tw_percentile_clamp: bool = False
    tw_percentile_low: float = 0.05
    tw_percentile_high: float = 0.95
    
    # Logging
    tw_log_weights: bool = True
    tw_log_every: int = 50
    val_loss_every: int = 125
    
    # Additional notes
    notes: str = ""
    
    def get_run_id(self) -> str:
        """Generate a unique but deterministic run ID based on config."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        hash_short = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{self.name}_{hash_short}"
    
    def to_cmd_args(self) -> List[str]:
        """Convert config to command line arguments for train_tw.py."""
        args = [
            f"--seed={self.seed}",
            f"--num_iterations={self.num_iterations}",
            f"--val_loss_every={self.val_loss_every}",
        ]
        
        if self.tw_enabled:
            args.extend([
                "--tw_enabled",
                f"--tw_function={self.tw_function}",
                f"--tw_clamp_min={self.tw_clamp_min}",
                f"--tw_clamp_max={self.tw_clamp_max}",
                f"--tw_schedule={self.tw_schedule}",
                f"--tw_warmup_steps={self.tw_warmup_steps}",
                f"--tw_anneal_final={self.tw_anneal_final}",
                f"--tw_cyclical_period={self.tw_cyclical_period}",
                f"--tw_focal_gamma={self.tw_focal_gamma}",
                f"--tw_log_every={self.tw_log_every}",
            ])
            
            if self.tw_log_weights:
                args.append("--tw_log_weights")
            
            if self.tw_percentile_clamp:
                args.extend([
                    "--tw_percentile_clamp",
                    f"--tw_percentile_low={self.tw_percentile_low}",
                    f"--tw_percentile_high={self.tw_percentile_high}",
                ])
        
        return args


# =============================================================================
# Experiment Definitions
# =============================================================================

# Standard seeds for reproducibility
SEEDS = [42, 123, 456, 789]

# Iteration counts for different stages
ITERATIONS = {
    'screening': 500,
    'medium': 1500,
    'full': 3000,
}


def generate_phase1_experiments() -> List[ExperimentConfig]:
    """Phase 1: Baseline establishment (3-4 seeds, full runs)."""
    experiments = []
    for seed in SEEDS[:4]:  # Use all 4 seeds for baseline
        experiments.append(ExperimentConfig(
            name=f"baseline_seed{seed}",
            description=f"Baseline run without token weighting (seed={seed})",
            phase=1,
            stage='full',
            seed=seed,
            num_iterations=ITERATIONS['full'],
            tw_enabled=False,
        ))
    return experiments


def generate_phase2_experiments(stage: str = 'screening') -> List[ExperimentConfig]:
    """
    Phase 2: Core hypothesis test (weighting functions).
    
    Ablations:
    - 2.1: Linear weighting (clamp 0.5-2.0)
    - 2.2: Square root weighting (clamp 0.7-1.5)
    - 2.3: Logarithmic weighting (clamp 0.8-1.3)
    - 2.4: Focal loss style (gamma=2.0, clamp 0.5-2.0)
    """
    experiments = []
    num_iterations = ITERATIONS[stage]
    seeds = SEEDS[:3] if stage == 'screening' else SEEDS[:3]  # 3 seeds for Phase 2
    
    # Ablation 2.1: Linear Weighting
    for seed in seeds:
        experiments.append(ExperimentConfig(
            name=f"p2_linear_seed{seed}",
            description="Linear weighting: weight = loss / mean_loss",
            phase=2,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function="linear",
            tw_clamp_min=0.5,
            tw_clamp_max=2.0,
            tw_schedule="constant",
            notes="Ablation 2.1: Simplest, most direct approach",
        ))
    
    # Ablation 2.2: Square Root Weighting
    for seed in seeds:
        experiments.append(ExperimentConfig(
            name=f"p2_sqrt_seed{seed}",
            description="Square root weighting: moderate non-linearity",
            phase=2,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function="sqrt",
            tw_clamp_min=0.7,
            tw_clamp_max=1.5,
            tw_schedule="constant",
            notes="Ablation 2.2: Less aggressive than linear, may be more stable",
        ))
    
    # Ablation 2.3: Logarithmic Weighting
    for seed in seeds:
        experiments.append(ExperimentConfig(
            name=f"p2_log_seed{seed}",
            description="Logarithmic weighting: conservative non-linearity",
            phase=2,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function="log",
            tw_clamp_min=0.8,
            tw_clamp_max=1.3,
            tw_schedule="constant",
            notes="Ablation 2.3: Most conservative, minimal disruption",
        ))
    
    # Ablation 2.4: Focal Loss Style
    for seed in seeds:
        experiments.append(ExperimentConfig(
            name=f"p2_focal_seed{seed}",
            description="Focal loss style: (loss/max_loss)^gamma",
            phase=2,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function="focal",
            tw_focal_gamma=2.0,
            tw_clamp_min=0.5,
            tw_clamp_max=2.0,
            tw_schedule="constant",
            notes="Ablation 2.4: Proven approach in focal loss literature",
        ))
    
    return experiments


def generate_phase3_experiments(
    best_function: str = "linear",
    stage: str = 'screening'
) -> List[ExperimentConfig]:
    """
    Phase 3: Hyperparameter tuning on best function from Phase 2.
    
    Ablations:
    - 3.1: Clamp bounds sweep
    - 3.2: Percentile-based clamping
    """
    experiments = []
    num_iterations = ITERATIONS[stage]
    seeds = SEEDS[:2]  # 2 seeds for hyperparameter sweep
    
    # Ablation 3.1: Clamp Bounds Sweep
    clamp_configs = [
        ("conservative", 0.8, 1.2),
        ("moderate", 0.6, 1.5),
        ("aggressive", 0.4, 2.0),
        ("very_aggressive", 0.3, 3.0),
    ]
    
    for clamp_name, clamp_min, clamp_max in clamp_configs:
        for seed in seeds:
            experiments.append(ExperimentConfig(
                name=f"p3_clamp_{clamp_name}_seed{seed}",
                description=f"Clamp bounds sweep: [{clamp_min}, {clamp_max}]",
                phase=3,
                stage=stage,
                seed=seed,
                num_iterations=num_iterations,
                tw_enabled=True,
                tw_function=best_function,
                tw_clamp_min=clamp_min,
                tw_clamp_max=clamp_max,
                tw_schedule="constant",
                notes=f"Ablation 3.1: {clamp_name} clamp bounds",
            ))
    
    # Ablation 3.2: Percentile-Based Clamping
    for seed in seeds:
        experiments.append(ExperimentConfig(
            name=f"p3_percentile_seed{seed}",
            description="Percentile-based clamping (5th-95th percentile)",
            phase=3,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function=best_function,
            tw_percentile_clamp=True,
            tw_percentile_low=0.05,
            tw_percentile_high=0.95,
            tw_schedule="constant",
            notes="Ablation 3.2: Adaptive to batch composition",
        ))
    
    return experiments


def generate_phase4_experiments(
    best_function: str = "linear",
    best_clamp_min: float = 0.5,
    best_clamp_max: float = 2.0,
    stage: str = 'screening'
) -> List[ExperimentConfig]:
    """
    Phase 4: Scheduling strategies.
    
    Ablations:
    - 4.1: Warmup schedule
    - 4.2: Annealing schedule
    - 4.3: Cyclical schedule
    - 4.4: Adaptive (validation-loss-based)
    """
    experiments = []
    num_iterations = ITERATIONS[stage]
    
    # Ablation 4.1: Warmup Schedule (3 seeds)
    for seed in SEEDS[:3]:
        experiments.append(ExperimentConfig(
            name=f"p4_warmup_seed{seed}",
            description="Warmup schedule: linear warmup over 1000 steps",
            phase=4,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function=best_function,
            tw_clamp_min=best_clamp_min,
            tw_clamp_max=best_clamp_max,
            tw_schedule="warmup",
            tw_warmup_steps=1000,
            notes="Ablation 4.1: Avoid instability when model is random",
        ))
    
    # Ablation 4.2: Annealing Schedule (3 seeds)
    for seed in SEEDS[:3]:
        experiments.append(ExperimentConfig(
            name=f"p4_anneal_seed{seed}",
            description="Annealing schedule: decay to 30% strength",
            phase=4,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function=best_function,
            tw_clamp_min=best_clamp_min,
            tw_clamp_max=best_clamp_max,
            tw_schedule="anneal",
            tw_anneal_final=0.3,
            notes="Ablation 4.2: Focus on hard tokens early, uniform refinement later",
        ))
    
    # Ablation 4.3: Cyclical Schedule (2 seeds)
    for seed in SEEDS[:2]:
        experiments.append(ExperimentConfig(
            name=f"p4_cyclical_seed{seed}",
            description="Cyclical schedule: sine wave with period 500",
            phase=4,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function=best_function,
            tw_clamp_min=best_clamp_min,
            tw_clamp_max=best_clamp_max,
            tw_schedule="cyclical",
            tw_cyclical_period=500,
            notes="Ablation 4.3: Prevent overfitting to hard token distribution",
        ))
    
    # Ablation 4.4: Adaptive Schedule (2 seeds)
    for seed in SEEDS[:2]:
        experiments.append(ExperimentConfig(
            name=f"p4_adaptive_seed{seed}",
            description="Adaptive schedule: validation-loss-based",
            phase=4,
            stage=stage,
            seed=seed,
            num_iterations=num_iterations,
            tw_enabled=True,
            tw_function=best_function,
            tw_clamp_min=best_clamp_min,
            tw_clamp_max=best_clamp_max,
            tw_schedule="adaptive",
            notes="Ablation 4.4: Automatic curriculum based on model capability",
        ))
    
    return experiments


def generate_phase5_experiments(
    best_function: str = "linear",
    best_clamp_min: float = 0.5,
    best_clamp_max: float = 2.0,
    best_schedule: str = "constant",
    best_schedule_params: Optional[Dict[str, Any]] = None
) -> List[ExperimentConfig]:
    """
    Phase 5: Final validation with best configuration.
    
    Run 4 seeds for strong statistical confidence.
    """
    experiments = []
    best_schedule_params = best_schedule_params or {}
    
    for seed in SEEDS[:4]:
        config = ExperimentConfig(
            name=f"p5_best_seed{seed}",
            description=f"Best configuration: {best_function} + {best_schedule}",
            phase=5,
            stage='full',
            seed=seed,
            num_iterations=ITERATIONS['full'],
            tw_enabled=True,
            tw_function=best_function,
            tw_clamp_min=best_clamp_min,
            tw_clamp_max=best_clamp_max,
            tw_schedule=best_schedule,
            notes="Phase 5: Final validation with best configuration",
        )
        
        # Apply any schedule-specific parameters
        for key, value in best_schedule_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        experiments.append(config)
    
    return experiments


# =============================================================================
# Experiment Runner
# =============================================================================

class AblationRunner:
    """Manages execution and tracking of ablation experiments."""
    
    def __init__(
        self,
        train_script: str = "train_tw.py",
        results_dir: str = "ablation_results",
        gpus: int = 8,
        dry_run: bool = False,
    ):
        self.train_script = train_script
        self.results_dir = Path(results_dir)
        self.gpus = gpus
        self.dry_run = dry_run
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Track experiment status
        self.status_file = self.results_dir / "experiment_status.json"
        self.status = self._load_status()
    
    def _load_status(self) -> Dict[str, Any]:
        """Load experiment status from file."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {"experiments": {}, "started_at": datetime.now().isoformat()}
    
    def _save_status(self):
        """Save experiment status to file."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def _get_experiment_dir(self, config: ExperimentConfig) -> Path:
        """Get the directory for storing experiment results."""
        phase_dir = self.results_dir / f"phase{config.phase}" / config.stage
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir / config.name
    
    def run_experiment(self, config: ExperimentConfig) -> bool:
        """Run a single experiment."""
        run_id = config.get_run_id()
        exp_dir = self._get_experiment_dir(config)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already completed
        if run_id in self.status["experiments"]:
            exp_status = self.status["experiments"][run_id]
            if exp_status.get("status") == "completed":
                print(f"  [SKIP] {config.name} already completed")
                return True
        
        # Save experiment config
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Build command
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.gpus}",
            self.train_script,
        ] + config.to_cmd_args()
        
        print(f"\n{'='*60}")
        print(f"Experiment: {config.name}")
        print(f"Description: {config.description}")
        print(f"Phase: {config.phase}, Stage: {config.stage}")
        print(f"Iterations: {config.num_iterations}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        if self.dry_run:
            print("  [DRY RUN] Would execute above command")
            return True
        
        # Update status to running
        self.status["experiments"][run_id] = {
            "name": config.name,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "config": asdict(config),
        }
        self._save_status()
        
        # Run experiment
        log_file = exp_dir / "run.log"
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as log_f:
                # Write command to log
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write(f"Started: {datetime.now().isoformat()}\n")
                log_f.write("="*60 + "\n\n")
                log_f.flush()
                
                # Run the training
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                
                # Stream output to both console and log file
                for line in process.stdout:
                    print(line, end='')
                    log_f.write(line)
                    log_f.flush()
                
                process.wait()
                
                elapsed_time = time.time() - start_time
                
                if process.returncode == 0:
                    self.status["experiments"][run_id].update({
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(),
                        "elapsed_seconds": elapsed_time,
                    })
                    print(f"\n  [SUCCESS] Completed in {elapsed_time:.1f}s")
                else:
                    self.status["experiments"][run_id].update({
                        "status": "failed",
                        "failed_at": datetime.now().isoformat(),
                        "return_code": process.returncode,
                        "elapsed_seconds": elapsed_time,
                    })
                    print(f"\n  [FAILED] Return code: {process.returncode}")
                    return False
                
        except Exception as e:
            self.status["experiments"][run_id].update({
                "status": "error",
                "error": str(e),
                "failed_at": datetime.now().isoformat(),
            })
            print(f"\n  [ERROR] {e}")
            return False
        
        finally:
            self._save_status()
        
        return True
    
    def run_experiments(self, experiments: List[ExperimentConfig]) -> Dict[str, bool]:
        """Run a list of experiments."""
        results = {}
        total = len(experiments)
        
        print(f"\n{'#'*60}")
        print(f"# Running {total} experiments")
        print(f"# Results directory: {self.results_dir}")
        print(f"# GPUs: {self.gpus}")
        print(f"{'#'*60}")
        
        for i, config in enumerate(experiments, 1):
            print(f"\n[{i}/{total}] Starting {config.name}...")
            success = self.run_experiment(config)
            results[config.name] = success
            
            if not success and not self.dry_run:
                print(f"\n  [WARNING] Experiment {config.name} failed, continuing...")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        completed = sum(1 for v in results.values() if v)
        print(f"Completed: {completed}/{total}")
        
        if any(not v for v in results.values()):
            print("\nFailed experiments:")
            for name, success in results.items():
                if not success:
                    print(f"  - {name}")
        
        return results
    
    def list_experiments(self, experiments: List[ExperimentConfig]):
        """Print a list of experiments without running them."""
        print(f"\n{'='*60}")
        print(f"Experiments to run: {len(experiments)}")
        print(f"{'='*60}")
        
        # Group by phase and stage
        by_phase = {}
        for exp in experiments:
            key = (exp.phase, exp.stage)
            if key not in by_phase:
                by_phase[key] = []
            by_phase[key].append(exp)
        
        for (phase, stage), exps in sorted(by_phase.items()):
            print(f"\nPhase {phase} ({stage}): {len(exps)} experiments")
            for exp in exps:
                status = "?"
                run_id = exp.get_run_id()
                if run_id in self.status.get("experiments", {}):
                    status = self.status["experiments"][run_id].get("status", "?")
                print(f"  [{status:^9}] {exp.name}: {exp.description[:50]}...")
        
        # Estimate total compute
        total_iterations = sum(exp.num_iterations for exp in experiments)
        full_run_equiv = total_iterations / ITERATIONS['full']
        print(f"\n{'='*60}")
        print(f"Total iterations: {total_iterations:,}")
        print(f"Full run equivalents: {full_run_equiv:.1f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run token importance weighting ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all Phase 1 baseline experiments
  python run_ablations.py --phase 1

  # Run Phase 2 screening (short runs, 500 steps)
  python run_ablations.py --phase 2 --stage screening

  # Run Phase 2 full runs
  python run_ablations.py --phase 2 --stage full

  # Run a specific experiment by name
  python run_ablations.py --experiment baseline_seed42

  # List all experiments without running
  python run_ablations.py --phase 2 --dry-run

  # Run with custom settings
  python run_ablations.py --phase 1 --gpus 4 --results-dir my_results
        """
    )
    
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4, 5],
        help="Which phase of experiments to run"
    )
    parser.add_argument(
        "--stage", type=str, choices=['screening', 'medium', 'full'],
        default='screening',
        help="Stage: screening (500 steps), medium (2000 steps), or full (5100 steps)"
    )
    parser.add_argument(
        "--experiment", type=str,
        help="Run a specific experiment by name"
    )
    parser.add_argument(
        "--gpus", type=int, default=8,
        help="Number of GPUs to use (default: 8)"
    )
    parser.add_argument(
        "--train-script", type=str, default="train_tw.py",
        help="Path to the training script (default: train_tw.py)"
    )
    parser.add_argument(
        "--results-dir", type=str, default="ablation_results",
        help="Directory for storing results (default: ablation_results)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List experiments without running them"
    )
    
    # Phase 3+ parameters (best from previous phases)
    parser.add_argument(
        "--best-function", type=str, default="linear",
        choices=['linear', 'sqrt', 'log', 'focal'],
        help="Best weighting function from Phase 2 (for Phase 3+)"
    )
    parser.add_argument(
        "--best-clamp-min", type=float, default=0.5,
        help="Best clamp minimum from Phase 3 (for Phase 4+)"
    )
    parser.add_argument(
        "--best-clamp-max", type=float, default=2.0,
        help="Best clamp maximum from Phase 3 (for Phase 4+)"
    )
    parser.add_argument(
        "--best-schedule", type=str, default="constant",
        choices=['constant', 'warmup', 'anneal', 'cyclical', 'adaptive'],
        help="Best schedule from Phase 4 (for Phase 5)"
    )
    
    args = parser.parse_args()
    
    # Generate experiments based on arguments
    experiments = []
    
    if args.experiment:
        # Find specific experiment across all phases
        all_experiments = (
            generate_phase1_experiments() +
            generate_phase2_experiments('screening') +
            generate_phase2_experiments('full') +
            generate_phase3_experiments(args.best_function, 'screening') +
            generate_phase4_experiments(args.best_function, args.best_clamp_min, args.best_clamp_max, 'screening') +
            generate_phase5_experiments(args.best_function, args.best_clamp_min, args.best_clamp_max, args.best_schedule)
        )
        experiments = [e for e in all_experiments if e.name == args.experiment]
        if not experiments:
            print(f"Error: Experiment '{args.experiment}' not found")
            print("\nAvailable experiments:")
            for e in all_experiments:
                print(f"  - {e.name}")
            sys.exit(1)
    
    elif args.phase:
        if args.phase == 1:
            experiments = generate_phase1_experiments()
        elif args.phase == 2:
            experiments = generate_phase2_experiments(args.stage)
        elif args.phase == 3:
            experiments = generate_phase3_experiments(args.best_function, args.stage)
        elif args.phase == 4:
            experiments = generate_phase4_experiments(
                args.best_function, args.best_clamp_min, args.best_clamp_max, args.stage
            )
        elif args.phase == 5:
            experiments = generate_phase5_experiments(
                args.best_function, args.best_clamp_min, args.best_clamp_max, args.best_schedule
            )
    
    else:
        parser.print_help()
        print("\nError: Must specify --phase or --experiment")
        sys.exit(1)
    
    # Create runner and execute
    runner = AblationRunner(
        train_script=args.train_script,
        results_dir=args.results_dir,
        gpus=args.gpus,
        dry_run=args.dry_run,
    )
    
    if args.dry_run:
        runner.list_experiments(experiments)
    else:
        runner.run_experiments(experiments)


if __name__ == "__main__":
    main()