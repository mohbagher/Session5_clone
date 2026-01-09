"""
Experiment Runner - Interactive Menu System for RIS Probe-Based ML Experiments

This script provides an interactive interface to run different research tasks
related to probe design, limited probing analysis, scaling studies, etc.
"""

import argparse
import os
import select
import sys
import time

# Add experiments to path
sys.path.append(os.path.dirname(__file__))

from experiments.tasks.task_a1_binary import run_task_a1
from experiments.tasks.task_a2_hadamard import run_task_a2
from experiments.tasks.task_a3_diversity import run_task_a3
from experiments.tasks.task_a4_sobol import run_task_a4
from experiments.tasks.task_a5_halton import run_task_a5
from experiments.tasks.task_b1_m_variation import run_task_b1
from experiments.tasks.task_b2_top_m import run_task_b2
from experiments.tasks.task_b3_baselines import run_task_b3
from experiments.tasks.task_c1_scale_k import run_task_c1
from experiments.tasks.task_c2_phase_resolution import run_task_c2
from experiments.tasks.task_d1_seed_variation import run_task_d1
from experiments.tasks.task_d2_sanity_checks import run_task_d2
from experiments.tasks.task_e1_summary import run_task_e1
from experiments.tasks.task_e2_comparison_plots import run_task_e2


# Task registry
TASKS = {
    1: {'name': 'A1: Binary Probes', 'func': run_task_a1, 'phase': 'A'},
    2: {'name': 'A2: Hadamard Probes', 'func': run_task_a2, 'phase': 'A'},
    3: {'name': 'A3: Probe Diversity Analysis', 'func': run_task_a3, 'phase': 'A'},
    4: {'name': 'A4: Sobol Probes', 'func': run_task_a4, 'phase': 'A'},
    5: {'name': 'A5: Halton Probes', 'func': run_task_a5, 'phase': 'A'},
    6: {'name': 'B1: M Variation Study', 'func': run_task_b1, 'phase': 'B'},
    7: {'name': 'B2: Top-m Selection', 'func': run_task_b2, 'phase': 'B'},
    8: {'name': 'B3: Baseline Comparison', 'func': run_task_b3, 'phase': 'B'},
    9: {'name': 'C1: Scale K', 'func': run_task_c1, 'phase': 'C'},
    10: {'name': 'C2: Phase Resolution', 'func': run_task_c2, 'phase': 'C'},
    11: {'name': 'D1: Seed Variation', 'func': run_task_d1, 'phase': 'D'},
    12: {'name': 'D2: Sanity Checks', 'func': run_task_d2, 'phase': 'D'},
    13: {'name': 'E1: Summary', 'func': run_task_e1, 'phase': 'E'},
    14: {'name': 'E2: Comparison Plots', 'func': run_task_e2, 'phase': 'E'},
}


class ExperimentSettings:
    """Settings for experiments."""
    def __init__(self, N=32, K=64, M=8, seed=42):
        self.N = N
        self.K = K
        self.M = M
        self.seed = seed
    
    def __str__(self):
        return f"N={self.N}, K={self.K}, M={self.M}, Seed={self.seed}"


def print_menu(settings: ExperimentSettings):
    """Print the interactive menu."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           RIS Probe-Based ML Experiment Framework                    ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Current Settings: {str(settings):49s} ║")
    print("║  [S] Change Settings                                                 ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  PHASE A: Probe Design                                               ║")
    print("║    [1] A1: Binary Probes                                             ║")
    print("║    [2] A2: Hadamard Probes                                           ║")
    print("║    [3] A3: Probe Diversity Analysis                                  ║")
    print("║    [4] A4: Sobol Probes                                              ║")
    print("║    [5] A5: Halton Probes                                             ║")
    print("║  PHASE B: Limited Probing Analysis                                   ║")
    print("║    [6] B1: M Variation Study                                         ║")
    print("║    [7] B2: Top-m Selection                                           ║")
    print("║    [8] B3: Baseline Comparison                                       ║")
    print("║  PHASE C: Scaling Study                                              ║")
    print("║    [9] C1: Scale K                                                   ║")
    print("║    [10] C2: Phase Resolution                                         ║")
    print("║  PHASE D: Quality Control                                            ║")
    print("║    [11] D1: Seed Variation                                           ║")
    print("║    [12] D2: Sanity Checks                                            ║")
    print("║  PHASE E: Documentation                                              ║")
    print("║    [13] E1: Summary                                                  ║")
    print("║    [14] E2: Comparison Plots                                         ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  [0] Run ALL Tasks                                                   ║")
    print("║  [99] Custom Selection (e.g., '1,4,6')                               ║")
    print("║  [Q] Quit                                                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")


def change_settings(settings: ExperimentSettings) -> ExperimentSettings:
    """Interactive settings change."""
    print("\n" + "="*70)
    print("Change Settings")
    print("="*70)
    print(f"Current: {settings}")
    print("\nPress Enter to keep current value")
    
    try:
        N_str = input(f"N (RIS elements) [{settings.N}]: ").strip()
        if N_str:
            settings.N = int(N_str)
        
        K_str = input(f"K (total probes) [{settings.K}]: ").strip()
        if K_str:
            settings.K = int(K_str)
        
        M_str = input(f"M (sensing budget) [{settings.M}]: ").strip()
        if M_str:
            settings.M = int(M_str)
        
        seed_str = input(f"Seed [{settings.seed}]: ").strip()
        if seed_str:
            settings.seed = int(seed_str)
        
        # Validate M <= K
        if settings.M > settings.K:
            print(f"\nWarning: M ({settings.M}) > K ({settings.K}). Setting M = K.")
            settings.M = settings.K
        
        print(f"\nNew settings: {settings}")
    except ValueError as e:
        print(f"Invalid input: {e}")
        print("Settings unchanged.")
    
    return settings


def run_task(task_id: int, settings: ExperimentSettings):
    """Run a single task."""
    if task_id not in TASKS:
        print(f"Error: Task {task_id} not found.")
        return
    
    task = TASKS[task_id]
    print("\n" + "="*70)
    print(f"Running Task {task_id}: {task['name']}")
    print("="*70)
    print(f"Settings: {settings}")
    print("="*70)
    
    try:
        result = task['func'](
            N=settings.N,
            K=settings.K,
            M=settings.M,
            seed=settings.seed,
            verbose=True
        )
        print("\n✓ Task completed successfully!")
        return result
    except Exception as e:
        print(f"\n✗ Task failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_tasks(settings: ExperimentSettings):
    """Run all tasks sequentially."""
    print("\n" + "="*70)
    print("Running ALL Tasks")
    print("="*70)
    
    results = {}
    for task_id in sorted(TASKS.keys()):
        result = run_task(task_id, settings)
        results[task_id] = result
        
        # Brief pause between tasks
        wait_for_next_task()
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED")
    print("="*70)
    return results


def run_custom_selection(settings: ExperimentSettings):
    """Run custom selection of tasks."""
    selection = input("\nEnter task numbers separated by commas (e.g., 1,4,6): ").strip()
    
    try:
        task_ids = [int(x.strip()) for x in selection.split(',')]
        
        results = {}
        for task_id in task_ids:
            if task_id in TASKS:
                result = run_task(task_id, settings)
                results[task_id] = result
                
                if len(task_ids) > 1:
                    wait_for_next_task()
            else:
                print(f"Warning: Task {task_id} not found. Skipping.")
        
        print("\n" + "="*70)
        print("SELECTED TASKS COMPLETED")
        print("="*70)
        return results
    
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None


def interactive_mode():
    """Run in interactive mode with menu."""
    settings = ExperimentSettings()
    
    while True:
        print_menu(settings)
        choice = input("\nEnter your choice: ").strip().upper()
        
        if choice == 'Q':
            print("\nExiting experiment runner. Goodbye!")
            break
        elif choice == 'S':
            settings = change_settings(settings)
        elif choice == '0':
            run_all_tasks(settings)
        elif choice == '99':
            run_custom_selection(settings)
        elif choice.isdigit():
            task_id = int(choice)
            if 1 <= task_id <= 14:
                run_task(task_id, settings)
            else:
                print("Invalid task number. Please choose 1-14.")
        else:
            print("Invalid choice. Please try again.")


def wait_for_next_task(timeout_seconds: int = 5):
    """Wait for Enter or auto-advance after timeout."""
    message = f"\nWaiting {timeout_seconds}s to continue (press Enter to skip)..."
    print(message)
    if not sys.stdin.isatty():
        time.sleep(timeout_seconds)
        return
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if ready:
        sys.stdin.readline()


def cli_mode(args):
    """Run in CLI mode with arguments."""
    settings = ExperimentSettings(N=args.N, K=args.K, M=args.M, seed=args.seed)
    
    print("="*70)
    print("RIS Probe-Based ML Experiment Framework (CLI Mode)")
    print("="*70)
    print(f"Settings: {settings}")
    print("="*70)
    
    if args.task == 'all':
        run_all_tasks(settings)
    else:
        # Parse task list
        task_ids = [int(x.strip()) for x in args.task.split(',')]
        results = {}
        for task_id in task_ids:
            if task_id in TASKS:
                result = run_task(task_id, settings)
                results[task_id] = result
            else:
                print(f"Warning: Task {task_id} not found. Skipping.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RIS Probe-Based ML Experiment Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--task', type=str, default=None,
                        help='Task number(s) to run (e.g., "1" or "1,4,6" or "all"). '
                             'If not specified, runs in interactive mode.')
    parser.add_argument('--N', type=int, default=32,
                        help='Number of RIS elements')
    parser.add_argument('--K', type=int, default=64,
                        help='Total number of probes in bank')
    parser.add_argument('--M', type=int, default=8,
                        help='Sensing budget (probes measured per sample)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Validate M <= K
    if args.M > args.K:
        print(f"Error: M ({args.M}) cannot be greater than K ({args.K})")
        sys.exit(1)
    
    if args.task is not None:
        # CLI mode
        cli_mode(args)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
