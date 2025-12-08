#!/usr/bin/env python3
"""
Ablation Study Runner for Auxiliary Prediction Heads

This script manages and executes ablation experiments to test the hypothesis that
auxiliary prediction heads at intermediate layers accelerate training.

Experimental Design:
====================

Phase 1: Baseline Establishment (3 runs)
    - No auxiliary heads, establish baseline variance

Phase 2: Layer Position Ablation
    - Where should auxiliary heads be placed?
    - Test: layers 2, 4, 6, 8, 10 individually
    - Hypothesis: Middle layers (4-8) benefit most

Phase 3: Number of Heads Ablation  
    - How many auxiliary heads are optimal?
    - Test: 1 head vs 2 heads vs 3 heads
    - Configurations: [6], [4,8], [3,6,9]

Phase 4: Loss Weight Ablation
    - What auxiliary loss weight works best?
    - Test: 0.01, 0.05, 0.1, 0.2, 0.5
    - Use best layer config from Phase 2/3

Phase 5: Loss Schedule Ablation
    - Should auxiliary loss weight change during training?
    - Test: constant, linear_decay, cosine_decay, warmup_decay

Phase 6: Best Configuration Validation
    - Run best config multiple times for statistical significance

Usage:
    python run_aux.py --phase baseline --runs 3
    python run_aux.py --phase layer_position --runs 1
    python run_aux.py --phase all --runs 3
    python run_aux.py --experiment aux_4_8_w0.1 --runs 5
    python run_aux.py --list  # List all available experiments
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import shutil


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    aux_head_layers: str  # Comma-separated, e.g., "4,8" or "" for baseline
    aux_loss_weight: float
    aux_loss_schedule: str
    description: str
    phase: str
    
    # Training hyperparameters (can override defaults)
    num_iterations: int = 5100
    val_loss_every: int = 125
    learning_rate: float = 0.0036
    batch_size: int = 512
    device_batch_size: int = 64
    
    def to_args(self) -> List[str]:
        """Convert config to command-line arguments."""
        args = [
            f"--num_iterations={self.num_iterations}",
            f"--val_loss_every={self.val_loss_every}",
            f"--learning_rate={self.learning_rate}",
            f"--batch_size={self.batch_size}",
            f"--device_batch_size={self.device_batch_size}",
        ]
        if self.aux_head_layers:
            args.extend([
                f"--aux_head_layers={self.aux_head_layers}",
                f"--aux_loss_weight={self.aux_loss_weight}",
                f"--aux_loss_schedule={self.aux_loss_schedule}",
            ])
        return args


def get_all_experiments() -> Dict[str, ExperimentConfig]:
    """
    Define all experiments for the ablation study.
    
    Returns a dictionary mapping experiment names to their configurations.
    """
    experiments = {}
    
    # =========================================================================
    # Phase 1: Baseline
    # =========================================================================
    experiments["baseline"] = ExperimentConfig(
        name="baseline",
        aux_head_layers="",
        aux_loss_weight=0.0,
        aux_loss_schedule="constant",
        description="Baseline: No auxiliary heads",
        phase="baseline",
    )
    
    # =========================================================================
    # Phase 2: Layer Position Ablation
    # Single auxiliary head at different layers
    # =========================================================================
    for layer in [2, 4, 6, 8, 10]:
        experiments[f"aux_{layer}"] = ExperimentConfig(
            name=f"aux_{layer}",
            aux_head_layers=str(layer),
            aux_loss_weight=0.1,
            aux_loss_schedule="constant",
            description=f"Single aux head at layer {layer}",
            phase="layer_position",
        )
    
    # =========================================================================
    # Phase 3: Number of Heads Ablation
    # Different numbers and positions of auxiliary heads
    # =========================================================================
    
    # Two heads
    experiments["aux_4_8"] = ExperimentConfig(
        name="aux_4_8",
        aux_head_layers="4,8",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Two aux heads at layers 4 and 8",
        phase="num_heads",
    )
    
    experiments["aux_3_9"] = ExperimentConfig(
        name="aux_3_9",
        aux_head_layers="3,9",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Two aux heads at layers 3 and 9 (wider spacing)",
        phase="num_heads",
    )
    
    experiments["aux_5_7"] = ExperimentConfig(
        name="aux_5_7",
        aux_head_layers="5,7",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Two aux heads at layers 5 and 7 (tighter spacing)",
        phase="num_heads",
    )
    
    # Three heads
    experiments["aux_3_6_9"] = ExperimentConfig(
        name="aux_3_6_9",
        aux_head_layers="3,6,9",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Three aux heads at layers 3, 6, and 9",
        phase="num_heads",
    )
    
    experiments["aux_2_6_10"] = ExperimentConfig(
        name="aux_2_6_10",
        aux_head_layers="2,6,10",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Three aux heads at layers 2, 6, and 10",
        phase="num_heads",
    )
    
    experiments["aux_4_6_8"] = ExperimentConfig(
        name="aux_4_6_8",
        aux_head_layers="4,6,8",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Three aux heads at layers 4, 6, and 8 (middle focus)",
        phase="num_heads",
    )
    
    # Four heads (more aggressive deep supervision)
    experiments["aux_2_4_6_8"] = ExperimentConfig(
        name="aux_2_4_6_8",
        aux_head_layers="2,4,6,8",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Four aux heads at layers 2, 4, 6, and 8",
        phase="num_heads",
    )
    
    # =========================================================================
    # Phase 4: Loss Weight Ablation
    # Using the promising [4,8] configuration
    # =========================================================================
    for weight in [0.01, 0.05, 0.1, 0.2, 0.5]:
        weight_str = str(weight).replace(".", "p")
        experiments[f"aux_4_8_w{weight_str}"] = ExperimentConfig(
            name=f"aux_4_8_w{weight_str}",
            aux_head_layers="4,8",
            aux_loss_weight=weight,
            aux_loss_schedule="constant",
            description=f"Aux heads [4,8] with weight {weight}",
            phase="loss_weight",
        )
    
    # =========================================================================
    # Phase 5: Loss Schedule Ablation
    # Using [4,8] with weight 0.1
    # =========================================================================
    for schedule in ["constant", "linear_decay", "cosine_decay", "warmup_decay"]:
        experiments[f"aux_4_8_{schedule}"] = ExperimentConfig(
            name=f"aux_4_8_{schedule}",
            aux_head_layers="4,8",
            aux_loss_weight=0.1,
            aux_loss_schedule=schedule,
            description=f"Aux heads [4,8] with {schedule} schedule",
            phase="loss_schedule",
        )
    
    # =========================================================================
    # Phase 6: Combined Best Configurations (to be determined after Phase 2-5)
    # =========================================================================
    # These are hypothetical "best" configs - adjust based on results
    experiments["best_v1"] = ExperimentConfig(
        name="best_v1",
        aux_head_layers="4,8",
        aux_loss_weight=0.1,
        aux_loss_schedule="linear_decay",
        description="Best config v1: [4,8], w=0.1, linear_decay",
        phase="best",
    )
    
    experiments["best_v2"] = ExperimentConfig(
        name="best_v2",
        aux_head_layers="3,6,9",
        aux_loss_weight=0.05,
        aux_loss_schedule="cosine_decay",
        description="Best config v2: [3,6,9], w=0.05, cosine_decay",
        phase="best",
    )
    
    # =========================================================================
    # Quick Test Experiments (shorter runs for debugging)
    # =========================================================================
    experiments["quick_baseline"] = ExperimentConfig(
        name="quick_baseline",
        aux_head_layers="",
        aux_loss_weight=0.0,
        aux_loss_schedule="constant",
        description="Quick baseline (500 iters)",
        phase="quick",
        num_iterations=500,
        val_loss_every=50,
    )
    
    experiments["quick_aux_4_8"] = ExperimentConfig(
        name="quick_aux_4_8",
        aux_head_layers="4,8",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Quick aux [4,8] (500 iters)",
        phase="quick",
        num_iterations=500,
        val_loss_every=50,
    )
    
    return experiments


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_name: str
    run_id: int
    config: Dict[str, Any]
    final_val_loss: float
    final_train_loss: float
    training_time_ms: float
    step_avg_ms: float
    log_file: str
    aux_log_file: Optional[str]
    success: bool
    error_message: Optional[str] = None
    
    # Extracted metrics
    val_loss_history: List[float] = field(default_factory=list)
    train_loss_history: List[float] = field(default_factory=list)


class ExperimentRunner:
    """Manages running experiments and collecting results."""
    
    def __init__(
        self,
        output_dir: str = "experiments",
        num_gpus: int = 8,
        train_script: str = "aux_train_gpt.py",
    ):
        self.output_dir = Path(output_dir)
        self.num_gpus = num_gpus
        self.train_script = train_script
        self.experiments = get_all_experiments()
        
        # Create output directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"aux_heads_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[ExperimentResult] = []
        
        # Save experiment definitions
        self._save_experiment_definitions()
    
    def _save_experiment_definitions(self):
        """Save all experiment configurations to JSON."""
        configs = {name: asdict(exp) for name, exp in self.experiments.items()}
        with open(self.run_dir / "experiment_configs.json", "w") as f:
            json.dump(configs, f, indent=2)
    
    def run_experiment(
        self,
        experiment_name: str,
        run_id: int = 0,
        dry_run: bool = False,
    ) -> ExperimentResult:
        """Run a single experiment."""
        
        if experiment_name not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        config = self.experiments[experiment_name]
        
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name} (run {run_id})")
        print(f"Description: {config.description}")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={self.num_gpus}",
            self.train_script,
        ] + config.to_args()
        
        print(f"Command: {' '.join(cmd)}")
        
        if dry_run:
            print("[DRY RUN] Would execute the above command")
            return ExperimentResult(
                experiment_name=experiment_name,
                run_id=run_id,
                config=asdict(config),
                final_val_loss=0.0,
                final_train_loss=0.0,
                training_time_ms=0.0,
                step_avg_ms=0.0,
                log_file="",
                aux_log_file=None,
                success=True,
            )
        
        # Create experiment-specific output directory
        exp_dir = self.run_dir / experiment_name / f"run_{run_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(exp_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # Run experiment
        log_file = exp_dir / "output.log"
        start_time = time.time()
        
        try:
            with open(log_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=7200,  # 2 hour timeout
                )
            
            success = result.returncode == 0
            error_message = None if success else f"Return code: {result.returncode}"
            
        except subprocess.TimeoutExpired:
            success = False
            error_message = "Experiment timed out after 2 hours"
        except Exception as e:
            success = False
            error_message = str(e)
        
        elapsed_time = time.time() - start_time
        
        # Parse results from log file
        parsed = self._parse_log_file(log_file)
        
        # Find the most recent log files in logs/ directory
        main_log, aux_log = self._find_latest_logs()
        
        # Copy logs to experiment directory
        if main_log and os.path.exists(main_log):
            shutil.copy(main_log, exp_dir / "train_log.txt")
        if aux_log and os.path.exists(aux_log):
            shutil.copy(aux_log, exp_dir / "aux_log.txt")
        
        result = ExperimentResult(
            experiment_name=experiment_name,
            run_id=run_id,
            config=asdict(config),
            final_val_loss=parsed.get("final_val_loss", float("nan")),
            final_train_loss=parsed.get("final_train_loss", float("nan")),
            training_time_ms=parsed.get("training_time_ms", elapsed_time * 1000),
            step_avg_ms=parsed.get("step_avg_ms", float("nan")),
            log_file=str(log_file),
            aux_log_file=str(aux_log) if aux_log else None,
            success=success,
            error_message=error_message,
            val_loss_history=parsed.get("val_loss_history", []),
            train_loss_history=parsed.get("train_loss_history", []),
        )
        
        # Save result
        with open(exp_dir / "result.json", "w") as f:
            json.dump(asdict(result), f, indent=2)
        
        self.results.append(result)
        
        # Print summary
        print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
        if success:
            print(f"  Final val loss: {result.final_val_loss:.4f}")
            print(f"  Training time: {result.training_time_ms/1000:.1f}s")
            print(f"  Step avg: {result.step_avg_ms:.2f}ms")
        else:
            print(f"  Error: {error_message}")
        
        return result
    
    def _parse_log_file(self, log_file: Path) -> Dict[str, Any]:
        """Parse experiment output log to extract metrics."""
        result = {
            "val_loss_history": [],
            "train_loss_history": [],
        }
        
        if not log_file.exists():
            return result
        
        with open(log_file) as f:
            lines = f.readlines()
        
        for line in lines:
            # Parse validation loss lines
            # Format: step:X/Y val_loss:Z.ZZZZ train_time:Xms step_avg:Yms
            if "val_loss:" in line and "step:" in line:
                try:
                    parts = line.strip().split()
                    for part in parts:
                        if part.startswith("val_loss:"):
                            val_loss = float(part.split(":")[1])
                            result["val_loss_history"].append(val_loss)
                            result["final_val_loss"] = val_loss
                        elif part.startswith("train_time:"):
                            result["training_time_ms"] = float(part.split(":")[1].replace("ms", ""))
                        elif part.startswith("step_avg:"):
                            result["step_avg_ms"] = float(part.split(":")[1].replace("ms", ""))
                except:
                    pass
            
            # Parse training loss lines
            # Format: step:X/Y train_loss:Z.ZZZZ ...
            elif "train_loss:" in line and "val_loss" not in line:
                try:
                    parts = line.strip().split()
                    for part in parts:
                        if part.startswith("train_loss:"):
                            train_loss = float(part.split(":")[1])
                            result["train_loss_history"].append(train_loss)
                            result["final_train_loss"] = train_loss
                except:
                    pass
        
        return result
    
    def _find_latest_logs(self) -> tuple:
        """Find the most recently created log files."""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return None, None
        
        # Find the most recent .txt file
        txt_files = list(logs_dir.glob("*.txt"))
        if not txt_files:
            return None, None
        
        latest = max(txt_files, key=lambda p: p.stat().st_mtime)
        
        # Check for corresponding aux log
        aux_log = latest.parent / f"{latest.stem}_aux.txt"
        
        return str(latest), str(aux_log) if aux_log.exists() else None
    
    def run_phase(
        self,
        phase: str,
        num_runs: int = 1,
        dry_run: bool = False,
    ) -> List[ExperimentResult]:
        """Run all experiments in a phase."""
        
        phase_experiments = [
            name for name, exp in self.experiments.items()
            if exp.phase == phase
        ]
        
        if not phase_experiments:
            print(f"No experiments found for phase: {phase}")
            return []
        
        print(f"\n{'#'*60}")
        print(f"# Phase: {phase}")
        print(f"# Experiments: {len(phase_experiments)}")
        print(f"# Runs per experiment: {num_runs}")
        print(f"{'#'*60}")
        
        results = []
        for exp_name in phase_experiments:
            for run_id in range(num_runs):
                result = self.run_experiment(exp_name, run_id, dry_run)
                results.append(result)
        
        return results
    
    def run_all(
        self,
        num_runs: int = 1,
        dry_run: bool = False,
        phases: Optional[List[str]] = None,
    ) -> List[ExperimentResult]:
        """Run all experiments or specified phases."""
        
        if phases is None:
            # Default order for systematic ablation
            phases = [
                "baseline",
                "layer_position",
                "num_heads",
                "loss_weight",
                "loss_schedule",
            ]
        
        results = []
        for phase in phases:
            phase_results = self.run_phase(phase, num_runs, dry_run)
            results.extend(phase_results)
        
        # Save all results
        self._save_all_results()
        
        return results
    
    def _save_all_results(self):
        """Save all collected results."""
        results_file = self.run_dir / "all_results.json"
        with open(results_file, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        # Generate summary
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate a summary report of all results."""
        summary_file = self.run_dir / "summary.txt"
        
        with open(summary_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("AUXILIARY HEAD EXPERIMENT SUMMARY\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
            
            # Group by phase
            phases = {}
            for result in self.results:
                phase = self.experiments[result.experiment_name].phase
                if phase not in phases:
                    phases[phase] = []
                phases[phase].append(result)
            
            for phase, results in phases.items():
                f.write(f"\n## Phase: {phase}\n")
                f.write("-"*60 + "\n")
                f.write(f"{'Experiment':<25} {'Val Loss':>10} {'Time(s)':>10} {'Status':>10}\n")
                f.write("-"*60 + "\n")
                
                for r in results:
                    status = "OK" if r.success else "FAIL"
                    f.write(f"{r.experiment_name:<25} {r.final_val_loss:>10.4f} "
                           f"{r.training_time_ms/1000:>10.1f} {status:>10}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\nSummary saved to: {summary_file}")
    
    def list_experiments(self):
        """Print all available experiments."""
        print("\nAvailable Experiments:")
        print("="*80)
        
        # Group by phase
        phases = {}
        for name, exp in self.experiments.items():
            if exp.phase not in phases:
                phases[exp.phase] = []
            phases[exp.phase].append((name, exp))
        
        for phase, exps in sorted(phases.items()):
            print(f"\n## Phase: {phase}")
            print("-"*60)
            for name, exp in sorted(exps):
                layers = exp.aux_head_layers if exp.aux_head_layers else "none"
                print(f"  {name:<25} layers={layers:<12} w={exp.aux_loss_weight}")
                print(f"    └─ {exp.description}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run auxiliary head ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        choices=["baseline", "layer_position", "num_heads", "loss_weight", 
                 "loss_schedule", "best", "quick", "all"],
        help="Run all experiments in a specific phase",
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        help="Run a specific experiment by name",
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per experiment (default: 1)",
    )
    
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Output directory for results (default: experiments)",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments",
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    
    parser.add_argument(
        "--train_script",
        type=str,
        default="train_gpt_aux_heads.py",
        help="Path to training script",
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        train_script=args.train_script,
    )
    
    if args.list:
        runner.list_experiments()
        return
    
    if args.experiment:
        # Run specific experiment
        for run_id in range(args.runs):
            runner.run_experiment(args.experiment, run_id, args.dry_run)
    elif args.phase:
        if args.phase == "all":
            runner.run_all(args.runs, args.dry_run)
        else:
            runner.run_phase(args.phase, args.runs, args.dry_run)
    else:
        parser.print_help()
        print("\n\nUse --list to see available experiments")


if __name__ == "__main__":
    main()