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
import signal
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import shutil

# =============================================================================
# GPU Cleanup Utilities
# =============================================================================

def cleanup_gpu_processes(verbose: bool = True):
    """Kill any lingering GPU processes and clear CUDA cache."""
    # Give processes time to clean up naturally
    time.sleep(2)
    
    # Try to kill any orphaned torchrun/python processes using GPUs
    try:
        # Get PIDs of processes using NVIDIA GPUs
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            current_pid = os.getpid()
            for pid in pids:
                pid = pid.strip()
                if pid and pid.isdigit() and int(pid) != current_pid:
                    try:
                        # Check if it's a python/torch process before killing
                        proc_check = subprocess.run(
                            ['ps', '-p', pid, '-o', 'comm='],
                            capture_output=True, text=True, timeout=5
                        )
                        if 'python' in proc_check.stdout.lower():
                            if verbose:
                                print(f"    Killing orphaned GPU process: {pid}")
                            os.kill(int(pid), signal.SIGTERM)
                            time.sleep(1)
                            # Force kill if still running
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                            except ProcessLookupError:
                                pass  # Already dead
                    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Additional wait for processes to terminate
    time.sleep(2)
    
    # Clear CUDA cache via a quick Python invocation
    try:
        subprocess.run(
            ['python', '-c', 'import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None'],
            timeout=10, capture_output=True
        )
    except Exception:
        pass


def wait_for_gpus_free(timeout: int = 60, verbose: bool = True) -> bool:
    """Wait for GPUs to be free before starting next experiment."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
                if not pids:
                    if verbose:
                        print("    GPUs are free.")
                    return True
                if verbose:
                    print(f"    Waiting for {len(pids)} GPU process(es) to finish...")
        except Exception:
            pass
        time.sleep(5)
    
    if verbose:
        print("    Warning: Timeout waiting for GPUs to be free")
    return False


def kill_process_tree(pid: int):
    """Kill a process and all its children."""
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(3)
        # Force kill if still running
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    except (ProcessLookupError, PermissionError, OSError):
        pass

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
    
    def to_script_modifications(self) -> Dict[str, Any]:
        """
        Convert config to modifications for the training script.
        Returns a dict that can be used to modify the Hyperparameters dataclass.
        """
        mods = {
            'num_iterations': self.num_iterations,
            'val_loss_every': self.val_loss_every,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'device_batch_size': self.device_batch_size,
            'aux_head_layers': self.aux_head_layers,
            'aux_loss_weight': self.aux_loss_weight,
            'aux_loss_schedule': self.aux_loss_schedule,
        }
        return mods


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_name: str
    run_id: int
    log_seed: int
    config: Dict[str, Any]
    final_val_loss: float
    final_train_loss: float
    training_time_ms: float
    step_avg_ms: float
    log_file: str
    aux_log_file: Optional[str]
    success: bool
    error_message: Optional[str] = None
    val_loss_history: List[float] = field(default_factory=list)
    train_loss_history: List[float] = field(default_factory=list)


# =============================================================================
# Experiment Definitions
# =============================================================================

def get_screening_experiments() -> Dict[str, ExperimentConfig]:
    """Stage 1: Quick validation experiments (1000 iters, ~3 min each)"""
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
    """Stage 2: Layer position ablation (full length runs)"""
    experiments = {}
    
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
    
    return experiments


def get_deep_dive_experiments(best_layer_config: str) -> Dict[str, ExperimentConfig]:
    """Stage 3: Deep dive into best configuration from Stage 2"""
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
    
    # Loss schedule ablation
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
    """Get all experiments including screening and layer position"""
    experiments = {}
    experiments.update(get_screening_experiments())
    experiments.update(get_layer_position_experiments())
    return experiments


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Manages and executes ablation experiments."""
    
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
        self.results = []
        
        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Experiment directory: {self.run_dir}")
    
    def run_experiment(
        self,
        experiment_name: str,
        run_id: int = 0,
    ) -> ExperimentResult:
        """Run a single experiment."""
        
        if experiment_name not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        config = self.experiments[experiment_name]
        # Generate a random seed for logging purposes only
        import random
        log_seed = random.randint(0, 2**31 - 1)
        
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name} (run {run_id})")
        print(f"Description: {config.description}")
        print(f"Log seed: {log_seed}")
        print(f"{'='*60}")
        
        # Create modified training script
        modified_script = self._create_modified_script(config)
        
        # Build torchrun command
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={self.num_gpus}",
            str(modified_script),
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Ensure GPUs are clean before starting
        print("  Ensuring GPUs are clean...")
        cleanup_gpu_processes(verbose=False)
        if not wait_for_gpus_free(timeout=30, verbose=True):
            print("  Warning: Starting experiment with potentially busy GPUs")
        
        # Create experiment-specific output directory
        exp_dir = self.run_dir / experiment_name / f"run_{run_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = asdict(config)
        config_dict["log_seed"] = log_seed
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Run experiment using Popen for better process control
        log_file = exp_dir / "output.log"
        start_time = time.time()
        process = None
        
        try:
            with open(log_file, "w") as f:
                # Create new process group so we can kill all children
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # Create new process group
                )
                
                # Wait for completion with timeout
                try:
                    return_code = process.wait(timeout=7200)  # 2 hour timeout
                    success = return_code == 0
                    error_message = None if success else f"Return code: {return_code}"
                except subprocess.TimeoutExpired:
                    success = False
                    error_message = "Experiment timed out after 2 hours"
                    # Kill the entire process group
                    print("  Timeout! Killing process tree...")
                    kill_process_tree(process.pid)
            
        except Exception as e:
            success = False
            error_message = str(e)
            if process and process.poll() is None:
                print(f"  Exception! Killing process tree: {e}")
                kill_process_tree(process.pid)
        finally:
            # Clean up modified script
            if modified_script.exists():
                modified_script.unlink()
            
            # Clean up any orphaned GPU processes
            print("  Cleaning up GPU processes...")
            cleanup_gpu_processes(verbose=True)
            wait_for_gpus_free(timeout=30, verbose=True)
        
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
            log_seed=log_seed,
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
        print(f"\nResult: {'✓ SUCCESS' if success else '✗ FAILED'}")
        if success:
            print(f"  Final val loss: {result.final_val_loss:.4f}")
            print(f"  Training time: {result.training_time_ms/1000:.1f}s")
            print(f"  Step avg: {result.step_avg_ms:.2f}ms")
        else:
            print(f"  Error: {error_message}")
        
        return result
    
    def _create_modified_script(self, config: ExperimentConfig) -> Path:
        """
        Create a modified version of the training script with experiment parameters.
        The clean script uses a Hyperparameters dataclass at module level, so we modify it.
        """
        # Read the original script
        with open(self.train_script, 'r') as f:
            script_content = f.read()
        
        # Create modifications for the Hyperparameters dataclass
        mods = config.to_script_modifications()
        
        # Find the Hyperparameters initialization and modify it
        # We'll add modifications right after "args = Hyperparameters()"
        
        modifications = []
        for key, value in mods.items():
            if isinstance(value, str):
                modifications.append(f'args.{key} = "{value}"')
            else:
                modifications.append(f'args.{key} = {value}')
        
        mod_block = '\n'.join(modifications)
        
        # Insert modifications after "args = Hyperparameters()"
        marker = "args = Hyperparameters()"
        if marker in script_content:
            script_content = script_content.replace(
                marker,
                f"{marker}\n# Experiment modifications\n{mod_block}\n"
            )
        else:
            raise ValueError(f"Could not find '{marker}' in training script")
        
        # Write modified script
        modified_path = self.run_dir / f"modified_script_{config.name}.py"
        with open(modified_path, 'w') as f:
            f.write(script_content)
        
        return modified_path
    
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
        
        # Exclude aux logs from main search
        txt_files = [f for f in txt_files if not f.name.endswith("_aux.txt")]
        if not txt_files:
            return None, None
        
        latest = max(txt_files, key=lambda p: p.stat().st_mtime)
        
        # Check for corresponding aux log
        aux_log = latest.parent / f"{latest.stem}_aux.txt"
        
        return str(latest), str(aux_log) if aux_log.exists() else None
    
    def run_phase(self, phase: str, num_runs: int = 1) -> List[ExperimentResult]:
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
                result = self.run_experiment(exp_name, run_id)
                results.append(result)
        
        return results
    
    def list_experiments(self):
        """Print all available experiments."""
        print("\n📋 Available Experiments:")
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
# Staged Ablation Runner
# =============================================================================

class StagedAblationRunner(ExperimentRunner):
    """Runs experiments in stages with decision points."""
    
    def __init__(
        self,
        output_dir: str = "experiments",
        num_gpus: int = 8,
        train_script: str = "aux_train_gpt.py",
        max_budget_hours: float = 7.0,
    ):
        super().__init__(output_dir, num_gpus, train_script)
        
        self.max_budget_hours = max_budget_hours
        self.budget_used_hours = 0.0
        self.stage_results = {}
        self.best_config = None
        
        # State file for persistence
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
                print(f"📂 Loaded state: {self.budget_used_hours:.2f}h used")
    
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
        time_per_run = (num_iterations / 5100) * 0.25
        return time_per_run * num_runs
    
    def run_stage_1_screening(self) -> Dict[str, Any]:
        """Stage 1: Quick validation (1 hour, ~4 runs @ 1000 iters each)"""
        print("\n" + "="*80)
        print("STAGE 1: QUICK VALIDATION")
        print("="*80)
        
        screening_exps = get_screening_experiments()
        results = []
        
        for exp_name in screening_exps:
            result = self.run_experiment(exp_name, run_id=0)
            results.append(result)
        
        # Analyze results
        baseline = next(r for r in results if r.experiment_name == "screen_baseline")
        aux_results = [r for r in results if r.experiment_name != "screen_baseline"]
        
        best_improvement = 0
        best_config = None
        
        for r in aux_results:
            if r.success and baseline.success:
                improvement = (baseline.final_val_loss - r.final_val_loss) / baseline.final_val_loss * 100
                print(f"{r.experiment_name}: {improvement:+.2f}% vs baseline")
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_config = r.experiment_name
        
        proceed = best_improvement > 2.0
        
        stage_result = {
            "results": results,
            "best_improvement_pct": best_improvement,
            "best_config": best_config,
            "decision": "proceed" if proceed else "stop",
        }
        
        self.stage_results["stage1"] = stage_result
        self._save_state()
        
        print(f"\n{'='*80}")
        print(f"STAGE 1 DECISION: {'✓ PROCEED' if proceed else '✗ STOP'}")
        print(f"Best improvement: {best_improvement:.2f}%")
        print(f"{'='*80}\n")
        
        return stage_result
    
    def run_stage_2_layer_position(self, baseline_only: bool = False) -> Dict[str, Any]:
        """Stage 2: Layer position ablation"""
        print("\n" + "="*80)
        print("STAGE 2: LAYER POSITION SCREENING")
        print("="*80)
        
        layer_exps = get_layer_position_experiments()
        results = []
        
        if baseline_only:
            result = self.run_experiment("baseline", run_id=0)
            results.append(result)
        else:
            for exp_name in layer_exps:
                result = self.run_experiment(exp_name, run_id=0)
                results.append(result)
        
        if not baseline_only:
            # Analyze results
            baseline = next(r for r in results if r.experiment_name == "baseline")
            aux_results = [r for r in results if r.experiment_name != "baseline"]
            
            best_improvement = 0
            best_layers = None
            
            for r in aux_results:
                if r.success and baseline.success:
                    improvement = (baseline.final_val_loss - r.final_val_loss) / baseline.final_val_loss * 100
                    print(f"{r.experiment_name}: {improvement:+.2f}% vs baseline")
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_layers = self.experiments[r.experiment_name].aux_head_layers
            
            print(f"\n{'='*80}")
            print(f"STAGE 2 RESULT: Best layers = {best_layers}")
            print(f"Best improvement: {best_improvement:.2f}%")
            print(f"{'='*80}\n")
            
            stage_result = {
                "results": results,
                "best_layers": best_layers,
                "best_improvement_pct": best_improvement,
            }
        else:
            stage_result = {
                "results": results,
                "baseline_only": True,
            }
        
        self.stage_results["stage2"] = stage_result
        self._save_state()
        
        return stage_result
    
    def run_stage_3_deep_dive(
        self,
        best_layers: Optional[str] = None,
        weight_only: bool = False,
        schedule_only: bool = False,
    ) -> Dict[str, Any]:
        """Stage 3: Deep dive into best configuration"""
        print("\n" + "="*80)
        print("STAGE 3: DEEP DIVE OPTIMIZATION")
        print("="*80)
        
        if best_layers is None:
            if "stage2" in self.stage_results:
                best_layers = self.stage_results["stage2"].get("best_layers")
            if best_layers is None:
                raise ValueError("No best_layers provided and no stage2 results found")
        
        print(f"Using best layers from Stage 2: {best_layers}")
        
        dive_exps = get_deep_dive_experiments(best_layers)
        results = []
        
        for exp_name, exp in dive_exps.items():
            # Skip based on flags
            if weight_only and not exp_name.startswith("dive_w"):
                continue
            if schedule_only and not exp_name.startswith("dive_") or exp_name.startswith("dive_w"):
                continue
            
            result = self.run_experiment(exp_name, run_id=0)
            results.append(result)
        
        # Find best configuration
        best_result = min(results, key=lambda r: r.final_val_loss if r.success else float('inf'))
        best_exp = self.experiments[best_result.experiment_name]
        
        self.best_config = {
            "layers": best_exp.aux_head_layers,
            "weight": best_exp.aux_loss_weight,
            "schedule": best_exp.aux_loss_schedule,
        }
        
        print(f"\n{'='*80}")
        print(f"STAGE 3 RESULT: Best config = {self.best_config}")
        print(f"Best val loss: {best_result.final_val_loss:.4f}")
        print(f"{'='*80}\n")
        
        stage_result = {
            "results": results,
            "best_config": self.best_config,
        }
        
        self.stage_results["stage3"] = stage_result
        self._save_state()
        
        return stage_result
    
    def run_stage_4_final_validation(
        self,
        best_config: Optional[Dict] = None,
        num_runs: int = 8,
    ) -> Dict[str, Any]:
        """Stage 4: Final validation with multiple runs"""
        print("\n" + "="*80)
        print("STAGE 4: FINAL VALIDATION")
        print("="*80)
        
        if best_config is None:
            best_config = self.best_config
        if best_config is None:
            raise ValueError("No best_config provided and no saved config found")
        
        print(f"Running {num_runs} validation runs of: {best_config}")
        
        # Create validation experiment
        val_exp = ExperimentConfig(
            name="final_validation",
            aux_head_layers=best_config["layers"],
            aux_loss_weight=best_config["weight"],
            aux_loss_schedule=best_config["schedule"],
            description="Final validation of best configuration",
            phase="validation",
        )
        
        self.experiments["final_validation"] = val_exp
        
        results = []
        for run_id in range(num_runs):
            result = self.run_experiment("final_validation", run_id)
            results.append(result)
        
        # Compute statistics
        val_losses = [r.final_val_loss for r in results if r.success]
        if val_losses:
            import statistics
            mean_loss = statistics.mean(val_losses)
            std_loss = statistics.stdev(val_losses) if len(val_losses) > 1 else 0
            
            print(f"\n{'='*80}")
            print(f"STAGE 4 RESULTS:")
            print(f"  Mean val loss: {mean_loss:.4f} ± {std_loss:.4f}")
            print(f"  Successful runs: {len(val_losses)}/{num_runs}")
            print(f"{'='*80}\n")
        
        stage_result = {
            "results": results,
            "statistics": {
                "mean": mean_loss if val_losses else None,
                "std": std_loss if val_losses else None,
                "n_success": len(val_losses),
                "n_total": num_runs,
            }
        }
        
        self.stage_results["stage4"] = stage_result
        self._save_state()
        
        return stage_result


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
    parser.add_argument("--staged", action="store_true",
                       help="Run complete staged ablation")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4],
                       help="Run specific stage (1=screening, 2=layer_position, 3=deep_dive, 4=validation)")
    
    # Stage-specific options
    parser.add_argument("--best_layers", type=str,
                       help="Best layer config for Stage 3 (e.g., '4,8')")
    parser.add_argument("--best_weight", type=float,
                       help="Best weight for Stage 4")
    parser.add_argument("--best_schedule", type=str,
                       help="Best schedule for Stage 4")
    parser.add_argument("--baseline_only", action="store_true",
                       help="Stage 2: Only run baseline")
    parser.add_argument("--weight_only", action="store_true",
                       help="Stage 3: Only run weight ablation")
    parser.add_argument("--schedule_only", action="store_true",
                       help="Stage 3: Only run schedule ablation")
    parser.add_argument("--validation_runs", type=int, default=8,
                       help="Stage 4: Number of validation runs")
    
    # Legacy mode
    parser.add_argument("--phase", type=str,
                       choices=["screening", "layer_position"],
                       help="Run all experiments in a phase")
    parser.add_argument("--experiment", type=str,
                       help="Run specific experiment by name")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per experiment")
    
    # General options
    parser.add_argument("--num_gpus", type=int, default=8,
                       help="Number of GPUs to use")
    parser.add_argument("--output_dir", type=str, default="experiments",
                       help="Output directory for results")
    parser.add_argument("--train_script", type=str, default="aux_train_gpt.py",
                       help="Path to training script")
    parser.add_argument("--max_budget_hours", type=float, default=7.0,
                       help="Maximum compute budget in hours")
    parser.add_argument("--list", action="store_true",
                       help="List all available experiments")
    
    args = parser.parse_args()
    
    # STAGED ABLATION MODE
    if args.staged or args.stage:
        runner = StagedAblationRunner(
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            train_script=args.train_script,
            max_budget_hours=args.max_budget_hours,
        )
        
        if args.stage == 1:
            runner.run_stage_1_screening()
        elif args.stage == 2:
            runner.run_stage_2_layer_position(baseline_only=args.baseline_only)
        elif args.stage == 3:
            runner.run_stage_3_deep_dive(
                best_layers=args.best_layers,
                weight_only=args.weight_only,
                schedule_only=args.schedule_only,
            )
        elif args.stage == 4:
            if args.best_layers or args.best_weight or args.best_schedule:
                best_config = {
                    "layers": args.best_layers or "6",
                    "weight": args.best_weight or 0.1,
                    "schedule": args.best_schedule or "constant",
                }
            else:
                best_config = runner.best_config
            
            if best_config is None:
                print("⚠️  No best config found. Please provide --best_layers, --best_weight, --best_schedule")
                return
            
            runner.run_stage_4_final_validation(best_config, num_runs=args.validation_runs)
        
        return
    
    # LEGACY MODE
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        train_script=args.train_script,
    )
    
    if args.list:
        runner.list_experiments()
        return
    
    if args.experiment:
        for run_id in range(args.runs):
            runner.run_experiment(args.experiment, run_id)
    elif args.phase:
        runner.run_phase(args.phase, args.runs)
    else:
        parser.print_help()
        print("\n💡 Examples:")
        print("   python run_aux.py --staged")
        print("   python run_aux.py --stage 1")
        print("   python run_aux.py --list")


if __name__ == "__main__":
    main()