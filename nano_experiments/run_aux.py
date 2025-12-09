#!/usr/bin/env python3
"""
Ablation Study Runner for Auxiliary Prediction Heads

This script manages and executes ablation experiments to test the hypothesis that
auxiliary prediction heads at intermediate layers accelerate training.

STAGED ABLATION STRAT:
==========================================

Problem: We only have around ~7 hours worth of compute on 8xH100
Full run time: ~15 minutes per experiment
Maximum full runs: ~28 total

Stage 1: Quick val
    - Verify hypothesis has merit with shortened runs
    - Test the baseline, aux_6, aux_4_8, aux_3_6_9

Stage 2: Layer pos screening ~6 runs
    - Try to identify best layer positions
    - Test baseline (full), aux_4, aux_6, aux_8, aux_4_8
    - Decision: Need to select best performing layer configuration

Stage 3: Targeted runs ~10 runs
    - Loss weight ablation: 0.05, 0.1, 0.2 (3 runs)
    - Loss schedule: constant, linear_decay, cosine_decay (3 runs)  
    - Validation runs: Repeat 4 runs of best config so far
    - Decision: Try to finalize the best configuration

Stage 4: Final validation ~8 runs
    - Multiple runs of best configuration for statistical significance
    - Target: p < 0.05 for claimed improvements

Usage:
    # Run staged ablation automatically with decision points
    python run_aux.py --staged --max_budget_hours 7
    
    # Run individual stages
    python run_aux.py --stage 1  # Quick validation
    python run_aux.py --stage 2  # Layer position
    python run_aux.py --stage 3  # Deep dive
    python run_aux.py --stage 4  # Final validation
    
    # Manual control (legacy)
    python run_aux.py --phase screening --runs 1
    python run_aux.py --experiment aux_4_8 --runs 5
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
DEFAULT_SEEDS = [42, 1337, 2603, 4242, 7777]

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


def get_screening_experiments() -> Dict[str, ExperimentConfig]:
    """
    Stage 1: Quick validation experiments (1000 iters, ~3 min each)
    """
    experiments = {}
    
    experiments["screen_baseline"] = ExperimentConfig(
        name="screen_baseline",
        aux_head_layers="",
        aux_loss_weight=0.0,
        aux_loss_schedule="constant",
        description="Screening baseline (1000 iters)",
        phase="screening",
        num_iterations=1000,
        val_loss_every=50,
    )
    
    experiments["screen_aux_6"] = ExperimentConfig(
        name="screen_aux_6",
        aux_head_layers="6",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Quick test: single aux head at middle layer",
        phase="screening",
        num_iterations=1000,
        val_loss_every=50,
    )
    
    experiments["screen_aux_4_8"] = ExperimentConfig(
        name="screen_aux_4_8",
        aux_head_layers="4,8",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Quick test: original hypothesis [4,8]",
        phase="screening",
        num_iterations=1000,
        val_loss_every=50,
    )
    
    experiments["screen_aux_3_6_9"] = ExperimentConfig(
        name="screen_aux_3_6_9",
        aux_head_layers="3,6,9",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Quick test: three heads distributed",
        phase="screening",
        num_iterations=1000,
        val_loss_every=50,
    )
    return experiments


def get_layer_position_experiments() -> Dict[str, ExperimentConfig]:
    """
    Stage 2: Layer position ablation (full length runs).
    Identify which layer positions benefit most from aux heads.
    """
    experiments = {}
    
    # Full baseline for statistical reference
    experiments["baseline"] = ExperimentConfig(
        name="baseline",
        aux_head_layers="",
        aux_loss_weight=0.0,
        aux_loss_schedule="constant",
        description="Full baseline: no auxiliary heads",
        phase="layer_position",
    )
    
    # Single layer experiments
    for layer in [4, 6, 8]:
        experiments[f"aux_{layer}"] = ExperimentConfig(
            name=f"aux_{layer}",
            aux_head_layers=str(layer),
            aux_loss_weight=0.1,
            aux_loss_schedule="constant",
            description=f"Single aux head at layer {layer}",
            phase="layer_position",
        )
    
    # Original hypothesis
    experiments["aux_4_8"] = ExperimentConfig(
        name="aux_4_8",
        aux_head_layers="4,8",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Two aux heads at layers 4 and 8 (original hypothesis)",
        phase="layer_position",
    )
    
    # Repeat for variance estimation
    experiments["aux_6_repeat"] = ExperimentConfig(
        name="aux_6_repeat",
        aux_head_layers="6",
        aux_loss_weight=0.1,
        aux_loss_schedule="constant",
        description="Repeat aux_6 to estimate variance",
        phase="layer_position",
    )
    
    return experiments


def get_deep_dive_experiments(best_layer_config: str) -> Dict[str, ExperimentConfig]:
    """
    Stage 3: Deep dive into best configuration from Stage 2.
    Optimize loss weight and schedule for best layer configuration.
    
    Args:
        best_layer_config: Layer configuration from Stage 2 (e.g., "6" or "4,8")
    """
    experiments = {}
    
    # Loss weight ablation
    for weight in [0.05, 0.1, 0.2]:
        weight_str = str(weight).replace(".", "p")
        experiments[f"dive_w{weight_str}"] = ExperimentConfig(
            name=f"dive_w{weight_str}",
            aux_head_layers=best_layer_config,
            aux_loss_weight=weight,
            aux_loss_schedule="constant",
            description=f"Loss weight {weight} with best layers [{best_layer_config}]",
            phase="deep_dive",
        )
    
    # Loss schedule ablation (using weight=0.1 or best from above)
    for schedule in ["constant", "linear_decay", "cosine_decay"]:
        experiments[f"dive_{schedule}"] = ExperimentConfig(
            name=f"dive_{schedule}",
            aux_head_layers=best_layer_config,
            aux_loss_weight=0.1,
            aux_loss_schedule=schedule,
            description=f"Schedule {schedule} with best layers [{best_layer_config}]",
            phase="deep_dive",
        )
    
    return experiments


def get_all_experiments() -> Dict[str, ExperimentConfig]:
    """
    Define all experiments for ablation studies, includes legacy experiments
    
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
    # Phase 2: Layer Position Ablation (reduced to key layers)
    # =========================================================================
    for layer in [4, 6, 8]:
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
    
    experiments = experiments | get_screening_experiments()
    experiments = experiments | get_layer_position_experiments()
    
    return experiments


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_name: str
    run_id: int
    seed: int
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

# =============================================================================
# Experiment Runner (original class, kept for backward compatibility)
# =============================================================================

class ExperimentRunner:
    """Manages running experiments and collecting results."""
    
    def __init__(
        self,
        output_dir: str = "experiments",
        num_gpus: int = 8,
        train_script: str = "aux_train_gpt.py",
        seeds: Optional[List[int]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.num_gpus = num_gpus
        self.train_script = train_script
        self.experiments = get_all_experiments()
        self.seeds = seeds or DEFAULT_SEEDS
        
        # Create output directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"aux_heads_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Persist the seed plan for reproducibility
        with open(self.run_dir / "seed_plan.json", "w") as f:
            json.dump({"seeds": self.seeds}, f, indent=2)
        
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
        seed = self.seeds[run_id % len(self.seeds)]
        
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name} (run {run_id})")
        print(f"Description: {config.description}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={self.num_gpus}",
            self.train_script,
        ] + config.to_args() + [f"--seed={seed}"]
        
        print(f"Command: {' '.join(cmd)}")
        
        if dry_run:
            print("[DRY RUN] Would execute the above command")
            return ExperimentResult(
                experiment_name=experiment_name,
                run_id=run_id,
                seed=seed,
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
        config_dict = asdict(config)
        config_dict["seed"] = seed
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Run experiment
        log_file = exp_dir / "output.log"
        start_time = time.time()

        env = os.environ.copy()
        env["SEED"] = str(seed)
        
        try:
            with open(log_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
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
            seed=seed,
            config=config_dict,
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
                print(f"    â””â”€ {exp.description}")

class StagedAblationRunner(ExperimentRunner):
    """
    Runs experiments in stages with manual or automatic decision points.
    Optimized for limited compute budget.
    """
    
    def __init__(
        self,
        output_dir: str = "experiments",
        num_gpus: int = 8,
        train_script: str = "aux_train_gpt.py",
        seeds: Optional[List[int]] = None,
        max_budget_hours: float = 7.0,
    ):
        # Initialize the parent class with required arguments
        super().__init__(
            output_dir=output_dir,
            num_gpus=num_gpus,
            train_script=train_script,
            seeds=seeds,
        )
        
        # Now initialize StagedAblationRunner specific attributes
        self.max_budget_hours = max_budget_hours
        self.budget_used_hours = 0.0
        self.stage_results = {}
        self.best_config = None

        # Add state file for persistence between manual runs
        self.state_file = self.run_dir / "ablation_state.json"
        self._load_state()    
    
    def _load_state(self):
        """Load state from previous runs."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                self.budget_used_hours = state.get("budget_used_hours", 0.0)
                self.stage_results = state.get("stage_results", {})
                self.best_config = state.get("best_config")
                print(f"📂 Loaded state: {self.budget_used_hours:.2f}h used, stage {len(self.stage_results)} complete")
    
    def _save_state(self):
        """Save state for resuming later."""
        state = {
            "budget_used_hours": self.budget_used_hours,
            "stage_results": self.stage_results,
            "best_config": self.best_config,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)
        print(f"💾 State saved: {self.budget_used_hours:.2f}h used")

    def estimate_time_hours(self, num_iterations: int, num_runs: int = 1) -> float:
        """Estimate time in hours for experiments."""
        # Full run (5100 iters) takes ~15 min = 0.25 hours
        # Time scales roughly linearly with iterations
        time_per_run = (num_iterations / 5100) * 0.25
        return time_per_run * num_runs
    
    def check_budget(self, estimated_hours: float) -> bool:
        """Check if we have enough budget remaining (informational only)."""
        if self.budget_used_hours + estimated_hours > self.max_budget_hours:
            print(f"\n⚠️  Budget will exceed limit: {self.budget_used_hours + estimated_hours:.2f}h > {self.max_budget_hours:.2f}h")
            print(f"   Continuing anyway...")
        return True
    
    def run_stage_1_screening(self) -> Dict[str, Any]:
        """
        Stage 1: Quick validation (1 hour, ~4 runs @ 1000 iters each).
        
        Decision criteria:
        - If ANY aux config shows >2% improvement in val loss trajectory, proceed
        - Otherwise, hypothesis likely invalid, reconsider approach
        
        Returns:
            Dict with stage results and decision to proceed
        """
        print("\n" + "="*70)
        print("STAGE 1: QUICK VALIDATION")
        print("="*70)
        print("Budget: ~1 hour (4 experiments @ 1000 iters each)")
        print("Goal: Verify hypothesis has merit before full compute commitment")
        print()
        
        experiments = get_screening_experiments()
        estimated_time = self.estimate_time_hours(1000, num_runs=len(experiments))
        
        # Display estimated time (no interruption)
        print(f"Estimated time: {estimated_time:.2f}h")
        print(f"Current budget used: {self.budget_used_hours:.2f}h / {self.max_budget_hours:.2f}h\n")
        
        # Run screening experiments
        results = []
        for exp_name in sorted(experiments.keys()):
            result = self.run_experiment(exp_name, run_id=0)
            results.append(result)
        
        self.budget_used_hours += estimated_time
        
        # Analyze results
        baseline_loss = next(r.final_val_loss for r in results if r.experiment_name == "screen_baseline")
        aux_results = [(r.experiment_name, r.final_val_loss) for r in results if "aux" in r.experiment_name]
        
        best_aux_name, best_aux_loss = min(aux_results, key=lambda x: x[1])
        improvement = (baseline_loss - best_aux_loss) / baseline_loss * 100
        
        print(f"\n📊 STAGE 1 RESULTS:")
        print(f"  Baseline loss: {baseline_loss:.4f}")
        print(f"  Best aux loss: {best_aux_loss:.4f} ({best_aux_name})")
        print(f"  Improvement: {improvement:+.2f}%")
        
        # Decision
        proceed = improvement > 2.0  # Require >2% improvement to proceed
        
        if proceed:
            print(f"\n✅ DECISION: PROCEED to Stage 2")
            print(f"  Aux heads show promise ({improvement:.2f}% improvement)")
        else:
            print(f"\n❌ DECISION: STOP")
            print(f"  Insufficient improvement ({improvement:.2f}% < 2.0%)")
            print(f"  Hypothesis may not be valid for this architecture")
        
        return {
            "proceed": proceed,
            "improvement_pct": improvement,
            "best_screening_config": best_aux_name,
            "baseline_loss": baseline_loss,
            "best_aux_loss": best_aux_loss,
            "results": results,
        }
    
    def run_stage_2_layer_position(self, baseline_only: bool = False) -> Dict[str, Any]:
        """
        Stage 2: Layer position screening (1.5 hours, ~6 full runs).
        
        Args:
            baseline_only: If True, only run baseline for comparison
        
        Decision criteria:
        - Identify best performing layer configuration
        - Estimate variance from repeated run
        
        Returns:
            Dict with best layer configuration
        """
        print("\n" + "="*70)
        print("STAGE 2: LAYER POSITION SCREENING")
        print("="*70)
        print("Budget: ~1.5 hours (6 full-length experiments)")
        print("Goal: Identify which layer positions benefit most")
        print()
        
        experiments = get_layer_position_experiments()
        
        # Allow running only baseline if needed
        if baseline_only:
            experiments = {k: v for k, v in experiments.items() if k == "baseline"}
        
        estimated_time = self.estimate_time_hours(5100, num_runs=len(experiments))
        
        # Display estimated time (no interruption)
        print(f"Estimated time: {estimated_time:.2f}h")
        print(f"Current budget used: {self.budget_used_hours:.2f}h / {self.max_budget_hours:.2f}h\n")
        
        # Run experiments
        results = []
        for exp_name in sorted(experiments.keys()):
            result = self.run_experiment(exp_name, run_id=0)
            results.append(result)
        
        self.budget_used_hours += estimated_time
        
        # Analyze results
        baseline_result = next((r for r in results if r.experiment_name == "baseline"), None)
        if not baseline_result:
            print("⚠️  No baseline found - run with baseline_only=False first")
            return {"proceed": False, "reason": "No baseline"}
        
        aux_results = [(r.experiment_name, r.final_val_loss, r.training_time_ms) 
                       for r in results if "aux" in r.experiment_name and "repeat" not in r.experiment_name]
        
        if aux_results:
            best_name, best_loss, best_time = min(aux_results, key=lambda x: x[1])
            
            # Extract layer config from best experiment
            best_config = get_layer_position_experiments()[best_name]
            best_layers = best_config.aux_head_layers
            
            improvement = (baseline_result.final_val_loss - best_loss) / baseline_result.final_val_loss * 100
            time_delta = (best_time - baseline_result.training_time_ms) / 1000  # seconds
            
            print(f"\n📊 STAGE 2 RESULTS:")
            print(f"  Baseline: {baseline_result.final_val_loss:.4f} ({baseline_result.training_time_ms/1000:.1f}s)")
            print(f"  Best config: {best_name}")
            print(f"    Loss: {best_loss:.4f} ({improvement:+.2f}%)")
            print(f"    Time: {best_time/1000:.1f}s ({time_delta:+.1f}s)")
            print(f"    Layers: {best_layers}")
            
            self.best_config = best_layers
        else:
            best_layers = None
            improvement = 0
            
        # Estimate variance from repeat experiment
        repeat_name = "aux_6_repeat"
        if repeat_name in [r.experiment_name for r in results]:
            repeat_loss = next(r.final_val_loss for r in results if r.experiment_name == repeat_name)
            original_loss = next((r.final_val_loss for r in results if r.experiment_name == "aux_6"), None)
            if original_loss:
                variance_est = abs(repeat_loss - original_loss)
                print(f"\n  Variance estimate (aux_6 vs repeat): {variance_est:.4f}")
        
        stage_result = {
            "proceed": True,
            "best_layers": best_layers,
            "best_name": best_name if aux_results else None,
            "improvement_pct": improvement,
            "time_delta_s": time_delta if aux_results else 0,
            "results": results,
        }
        
        self.stage_results["stage2"] = stage_result
        self._save_state()
        
        return stage_result
    
    def run_stage_3_deep_dive(self, best_layers: Optional[str] = None, 
                              weight_only: bool = False,
                              schedule_only: bool = False) -> Dict[str, Any]:
        """
        Stage 3: Targeted deep dive (2.5 hours, ~10 runs).
        
        Args:
            best_layers: Best layer configuration (uses saved state if None)
            weight_only: Only run weight ablation experiments
            schedule_only: Only run schedule ablation experiments
        
        Returns:
            Dict with optimized configuration
        """
        # Use saved state if no layers provided
        if best_layers is None:
            if self.best_config:
                best_layers = self.best_config
            elif "stage2" in self.stage_results:
                best_layers = self.stage_results["stage2"].get("best_layers")
            
            if not best_layers:
                print("⚠️  No best_layers found. Please provide or run Stage 2 first.")
                return {"proceed": False, "reason": "No best_layers"}
        
        print("\n" + "="*70)
        print("STAGE 3: TARGETED DEEP DIVE")
        print("="*70)
        print(f"Budget: ~2.5 hours (10 full-length experiments)")
        print(f"Goal: Optimize loss weight and schedule for layers [{best_layers}]")
        print()
        
        experiments = get_deep_dive_experiments(best_layers)
        
        # Filter experiments based on flags
        if weight_only:
            experiments = {k: v for k, v in experiments.items() if "dive_w" in k}
        elif schedule_only:
            experiments = {k: v for k, v in experiments.items() if "dive_" in k and "w" not in k}
        
        # Add validation runs if running full set
        if not weight_only and not schedule_only:
            for i in range(4):
                experiments[f"validation_{i}"] = ExperimentConfig(
                    name=f"validation_{i}",
                    aux_head_layers=best_layers,
                    aux_loss_weight=0.1,
                    aux_loss_schedule="constant",
                    description=f"Validation run {i} of current best",
                    phase="deep_dive",
                )
        
        estimated_time = self.estimate_time_hours(5100, num_runs=len(experiments))
        
        # Display estimated time (no interruption)
        print(f"Estimated time: {estimated_time:.2f}h")
        print(f"Current budget used: {self.budget_used_hours:.2f}h / {self.max_budget_hours:.2f}h\n")
        
        # Run experiments
        results = []
        for exp_name in sorted(experiments.keys()):
            result = self.run_experiment(exp_name, run_id=0)
            results.append(result)
        
        self.budget_used_hours += estimated_time
        
        # Analyze results
        weight_results = [(r.experiment_name, r.final_val_loss) 
                          for r in results if "dive_w" in r.experiment_name]
        schedule_results = [(r.experiment_name, r.final_val_loss) 
                            for r in results if "dive_" in r.experiment_name and "w" not in r.experiment_name]
        validation_results = [r.final_val_loss for r in results if "validation" in r.experiment_name]
        
        best_weight_name, best_weight_loss = min(weight_results, key=lambda x: x[1]) if weight_results else (None, None)
        best_schedule_name, best_schedule_loss = min(schedule_results, key=lambda x: x[1]) if schedule_results else (None, None)
        
        print(f"\n📊 STAGE 3 RESULTS:")
        if best_weight_name:
            print(f"  Best weight config: {best_weight_name} ({best_weight_loss:.4f})")
        if best_schedule_name:
            print(f"  Best schedule config: {best_schedule_name} ({best_schedule_loss:.4f})")
        if validation_results:
            import numpy as np
            val_mean = np.mean(validation_results)
            val_std = np.std(validation_results)
            print(f"  Validation runs: {val_mean:.4f} ± {val_std:.4f}")
        
        # Determine overall best from this stage
        all_losses = []
        if weight_results:
            all_losses.extend(weight_results)
        if schedule_results:
            all_losses.extend(schedule_results)
        
        if all_losses:
            best_overall_name, best_overall_loss = min(all_losses, key=lambda x: x[1])
            best_config_obj = experiments[best_overall_name]
            
            self.best_config = {
                "layers": best_config_obj.aux_head_layers,
                "weight": best_config_obj.aux_loss_weight,
                "schedule": best_config_obj.aux_loss_schedule,
            }
            
            print(f"\n  Overall best: {best_overall_name}")
            print(f"    Layers: {self.best_config['layers']}")
            print(f"    Weight: {self.best_config['weight']}")
            print(f"    Schedule: {self.best_config['schedule']}")
        
        stage_result = {
            "proceed": True,
            "best_config": self.best_config,
            "results": results,
        }
        
        self.stage_results["stage3"] = stage_result
        self._save_state()
        
        return stage_result
    
    def run_stage_4_final_validation(self, best_config: Dict[str, Any], num_runs: int = 8) -> Dict[str, Any]:
        """
        Stage 4: Final validation (2 hours, ~8 runs).
        
        Multiple runs of best configuration for statistical significance.
        
        Args:
            best_config: Dict with 'layers', 'weight', 'schedule' keys
            num_runs: Number of validation runs
        
        Returns:
            Dict with final statistics and significance tests
        """
        print("\n" + "="*70)
        print("STAGE 4: FINAL VALIDATION")
        print("="*70)
        print(f"Budget: ~2 hours ({num_runs} full-length runs)")
        print(f"Goal: Statistical significance testing of best configuration")
        print()
        
        estimated_time = self.estimate_time_hours(5100, num_runs=num_runs * 2)  # best + baseline
        
        # Display estimated time (no interruption)
        print(f"Estimated time: {estimated_time:.2f}h")
        print(f"Current budget used: {self.budget_used_hours:.2f}h / {self.max_budget_hours:.2f}h\n")
        
        # Create experiment configs
        best_exp = ExperimentConfig(
            name="final_best",
            aux_head_layers=best_config["layers"],
            aux_loss_weight=best_config["weight"],
            aux_loss_schedule=best_config["schedule"],
            description="Final best configuration",
            phase="final_validation",
        )
        
        baseline_exp = ExperimentConfig(
            name="final_baseline",
            aux_head_layers="",
            aux_loss_weight=0.0,
            aux_loss_schedule="constant",
            description="Final baseline",
            phase="final_validation",
        )
        
        # Save configs temporarily
        self.experiments["final_best"] = best_exp
        self.experiments["final_baseline"] = baseline_exp
        
        # Run experiments
        best_results = []
        baseline_results = []
        
        for run_id in range(num_runs):
            print(f"\n--- Run {run_id + 1}/{num_runs} ---")
            best_result = self.run_experiment("final_best", run_id)
            baseline_result = self.run_experiment("final_baseline", run_id)
            best_results.append(best_result)
            baseline_results.append(baseline_result)
        
        self.budget_used_hours += estimated_time
        
        # Statistical analysis
        from scipy import stats as scipy_stats
        
        best_losses = [r.final_val_loss for r in best_results if r.success]
        baseline_losses = [r.final_val_loss for r in baseline_results if r.success]
        
        best_times = [r.training_time_ms for r in best_results if r.success]
        baseline_times = [r.training_time_ms for r in baseline_results if r.success]
        
        # T-tests
        loss_ttest = scipy_stats.ttest_ind(best_losses, baseline_losses)
        time_ttest = scipy_stats.ttest_ind(best_times, baseline_times)
        
        # Effect sizes
        loss_mean_best = np.mean(best_losses)
        loss_mean_baseline = np.mean(baseline_losses)
        loss_improvement_pct = (loss_mean_baseline - loss_mean_best) / loss_mean_baseline * 100
        
        time_mean_best = np.mean(best_times) / 1000
        time_mean_baseline = np.mean(baseline_times) / 1000
        time_delta = time_mean_best - time_mean_baseline
        
        print(f"\n📊 STAGE 4 FINAL RESULTS:")
        print(f"\n  Best Configuration:")
        print(f"    Layers: {best_config['layers']}")
        print(f"    Weight: {best_config['weight']}")
        print(f"    Schedule: {best_config['schedule']}")
        print(f"\n  Validation Loss:")
        print(f"    Baseline: {loss_mean_baseline:.4f} ± {np.std(baseline_losses):.4f}")
        print(f"    Best:     {loss_mean_best:.4f} ± {np.std(best_losses):.4f}")
        print(f"    Improvement: {loss_improvement_pct:+.2f}%")
        print(f"    p-value: {loss_ttest.pvalue:.6f}")
        print(f"    Significant (p<0.05): {loss_ttest.pvalue < 0.05}")
        print(f"\n  Training Time:")
        print(f"    Baseline: {time_mean_baseline:.1f}s ± {np.std(baseline_times)/1000:.1f}s")
        print(f"    Best:     {time_mean_best:.1f}s ± {np.std(best_times)/1000:.1f}s")
        print(f"    Delta: {time_delta:+.1f}s")
        print(f"    p-value: {time_ttest.pvalue:.6f}")
        
        # Overall conclusion
        print(f"\n{'='*70}")
        if loss_ttest.pvalue < 0.05 and loss_improvement_pct > 0:
            print(f"✅ CONCLUSION: Auxiliary heads provide SIGNIFICANT improvement")
            print(f"   Val loss improved by {loss_improvement_pct:.2f}% (p={loss_ttest.pvalue:.4f})")
        elif loss_improvement_pct > 0:
            print(f"⚠️  CONCLUSION: Auxiliary heads show improvement but NOT significant")
            print(f"   Val loss improved by {loss_improvement_pct:.2f}% (p={loss_ttest.pvalue:.4f})")
            print(f"   May need more runs for statistical power")
        else:
            print(f"❌ CONCLUSION: Auxiliary heads do NOT improve performance")
            print(f"   Val loss changed by {loss_improvement_pct:+.2f}% (p={loss_ttest.pvalue:.4f})")
        print(f"{'='*70}\n")
        
        print(f"\n💰 TOTAL BUDGET USED: {self.budget_used_hours:.2f} / {self.max_budget_hours:.2f} hours")
        
        return {
            "best_config": best_config,
            "loss_improvement_pct": loss_improvement_pct,
            "loss_pvalue": loss_ttest.pvalue,
            "time_delta_s": time_delta,
            "time_pvalue": time_ttest.pvalue,
            "best_results": best_results,
            "baseline_results": baseline_results,
            "significant": loss_ttest.pvalue < 0.05 and loss_improvement_pct > 0,
        }
    
    def run_staged_ablation(self) -> Dict[str, Any]:
        """
        Run complete staged ablation with automatic decision points.
        
        Returns:
            Dict with results from all stages
        """
        print("\n" + "="*80)
        print("STAGED ABLATION: AUXILIARY PREDICTION HEADS")
        print("="*80)
        print(f"Max budget: {self.max_budget_hours:.1f} hours")
        print(f"Estimated full run time: 0.25 hours")
        print(f"Maximum total runs: ~{int(self.max_budget_hours / 0.25)}")
        print("="*80)
        
        all_results = {}
        
        # Stage 1: Screening
        stage1 = self.run_stage_1_screening()
        all_results["stage1"] = stage1
        
        if not stage1["proceed"]:
            print(f"\n⚠️  Stopping after Stage 1: hypothesis does not show promise")
            return all_results
        
        # Stage 2: Layer Position
        stage2 = self.run_stage_2_layer_position()
        all_results["stage2"] = stage2
        
        # Stage 3: Deep Dive
        stage3 = self.run_stage_3_deep_dive(stage2["best_layers"])
        all_results["stage3"] = stage3
        
        # Stage 4: Final Validation
        stage4 = self.run_stage_4_final_validation(stage3["best_config"])
        all_results["stage4"] = stage4
        
        # Save all results
        self._save_staged_results(all_results)
        
        return all_results
    
    def _save_staged_results(self, all_results: Dict[str, Any]):
        """Save staged ablation results to file."""
        results_file = self.run_dir / "staged_ablation_results.json"
        
        # Convert results to JSON-serializable format
        serializable = {}
        for stage, stage_results in all_results.items():
            serializable[stage] = {}
            for key, value in stage_results.items():
                if key == "results" or key.endswith("_results"):
                    # Convert result objects to dicts
                    if isinstance(value, list):
                        serializable[stage][key] = [asdict(r) if hasattr(r, '__dict__') else r for r in value]
                    else:
                        serializable[stage][key] = value
                else:
                    serializable[stage][key] = value
        
        with open(results_file, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        
        print(f"\n💾 Staged results saved to: {results_file}")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run auxiliary head ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Staged ablation mode
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Run complete staged ablation with automatic decision points",
    )
    
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific stage manually (1=screening, 2=layer_position, 3=deep_dive, 4=validation)",
    )
    
    # Stage-specific options
    parser.add_argument(
        "--best_layers",
        type=str,
        help="Best layer configuration for Stage 3 (e.g., '4,8'). Uses saved state if not provided.",
    )
    
    parser.add_argument(
        "--best_weight",
        type=float,
        help="Best weight for Stage 4. Uses saved state if not provided.",
    )
    
    parser.add_argument(
        "--best_schedule",
        type=str,
        help="Best schedule for Stage 4. Uses saved state if not provided.",
    )
    
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Stage 2: Only run baseline experiment",
    )
    
    parser.add_argument(
        "--weight_only",
        action="store_true",
        help="Stage 3: Only run weight ablation experiments",
    )
    
    parser.add_argument(
        "--schedule_only",
        action="store_true",
        help="Stage 3: Only run schedule ablation experiments",
    )
    
    parser.add_argument(
        "--validation_runs",
        type=int,
        default=8,
        help="Stage 4: Number of validation runs (default: 8)",
    )
    
    parser.add_argument(
        "--max_budget_hours",
        type=float,
        default=7.0,
        help="Maximum compute budget in hours (default: 7.0)",
    )

    # Legacy mode
    parser.add_argument(
        "--phase",
        type=str,
        choices=["baseline", "layer_position", "num_heads", "loss_weight", 
                 "loss_schedule", "best", "quick", "screening", "all"],
        help="LEGACY: Run all experiments in a specific phase",
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
        default="aux_train_gpt.py",
        help="Path to training script",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds to cycle through for runs (default: 42,1337,2603,4242,7777)",
    )
    
    args = parser.parse_args()

    # Parse seeds once so staged and legacy modes share the same plan
    seed_list = DEFAULT_SEEDS if not args.seeds else [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    
    # STAGED ABLATION MODE
    if args.staged or args.stage:
        runner = StagedAblationRunner(
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            train_script=args.train_script,
            max_budget_hours=args.max_budget_hours,
            seeds=seed_list,
        )
        
        if args.staged:
            # Run complete staged ablation
            runner.run_staged_ablation()
        elif args.stage == 1:
            runner.run_stage_1_screening()
            runner._save_state()
        elif args.stage == 2:
            runner.run_stage_2_layer_position(baseline_only=args.baseline_only)
        elif args.stage == 3:
            runner.run_stage_3_deep_dive(
                best_layers=args.best_layers,
                weight_only=args.weight_only,
                schedule_only=args.schedule_only,
            )
        elif args.stage == 4:
            # Build best config from args or saved state
            if args.best_layers or args.best_weight or args.best_schedule:
                best_config = {
                    "layers": args.best_layers or runner.best_config.get("layers") if runner.best_config else None,
                    "weight": args.best_weight or runner.best_config.get("weight") if runner.best_config else 0.1,
                    "schedule": args.best_schedule or runner.best_config.get("schedule") if runner.best_config else "constant",
                }
            elif runner.best_config:
                best_config = runner.best_config
            else:
                print("⚠️  No best config found. Please provide --best_layers, --best_weight, --best_schedule")
                print("   or run Stage 3 first.")
                return
            
            runner.run_stage_4_final_validation(best_config, num_runs=args.validation_runs)
        
        return

    # LEGACY MODE (backward compatibility)
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        train_script=args.train_script,
        seeds=seed_list,
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
        print("   Example: python run_aux.py --staged --max_budget_hours 7")
        print("\n Use --list to see available experiments")


if __name__ == "__main__":
    main()