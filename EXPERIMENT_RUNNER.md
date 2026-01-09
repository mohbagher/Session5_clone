# Experiment Runner Documentation

## Overview

The Experiment Runner provides an interactive menu system and CLI interface for running comprehensive experiments on the RIS probe-based ML system. It includes 14 different tasks organized into 5 phases:

- **Phase A: Probe Design** - Analyzing different probe types (binary, Hadamard, diversity)
- **Phase B: Limited Probing Analysis** - M variation, top-m selection, baseline comparisons
- **Phase C: Scaling Study** - Testing different K values and phase resolutions
- **Phase D: Quality Control** - Seed variation and sanity checks
- **Phase E: Documentation** - Summary reports and comparison plots

## Quick Start

### Interactive Mode

Simply run the experiment runner without arguments to enter interactive mode:

```bash
python experiment_runner.py
```

You'll see a menu like this:

```
╔══════════════════════════════════════════════════════════════════════╗
║           RIS Probe-Based ML Experiment Framework                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Current Settings: N=32, K=64, M=8, Seed=42                          ║
║  [S] Change Settings                                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  PHASE A: Probe Design                                               ║
║    [1] A1: Binary Probes                                             ║
║    [2] A2: Hadamard Probes                                           ║
║    [3] A3: Probe Diversity Analysis                                  ║
...
```

### CLI Mode

Run specific tasks with command-line arguments:

```bash
# Run a single task
python experiment_runner.py --task 1 --N 32 --K 64 --M 8

# Run multiple tasks
python experiment_runner.py --task 1,3,6 --N 32 --K 64 --M 8

# Run all tasks
python experiment_runner.py --task all --N 64 --K 128 --M 16

# Custom configuration
python experiment_runner.py --task 6 --N 64 --K 128 --M 32 --seed 123
```

## Task Descriptions

### Phase A: Probe Design

#### Task 1: A1 - Binary Probes
- Generates binary probes with phases {0, π}
- Compares with continuous random probes
- Creates phase heatmaps and histograms
- Computes diversity metrics

**Outputs:**
- `results/A1_binary_probes/plots/phase_heatmap.png`
- `results/A1_binary_probes/plots/phase_histogram.png`
- `results/A1_binary_probes/metrics.txt`

#### Task 2: A2 - Hadamard Probes
- Generates structured Hadamard-based binary probes
- Compares with random binary probes
- Analyzes Hamming distance distributions
- Visualizes structured patterns

**Outputs:**
- `results/A2_hadamard_probes/plots/phase_heatmap.png`
- `results/A2_hadamard_probes/plots/hamming_distance_distribution.png`
- `results/A2_hadamard_probes/plots/distance_matrices.png`
- `results/A2_hadamard_probes/metrics.txt`

#### Task 3: A3 - Probe Diversity Analysis
- Compares all probe types: continuous, binary, 2-bit, Hadamard
- Generates pairwise similarity/distance distributions
- Creates summary comparison table
- Analyzes phase distributions

**Outputs:**
- `results/A3_diversity_analysis/diversity_summary.csv`
- `results/A3_diversity_analysis/plots/pairwise_distributions.png`
- `results/A3_diversity_analysis/plots/diversity_comparison.png`
- `results/A3_diversity_analysis/plots/phase_distributions.png`
- `results/A3_diversity_analysis/metrics.txt`

#### Task 4: A4 - Sobol Probes
- Generates Sobol low-discrepancy probes
- Compares with continuous random probes
- Creates phase heatmaps and histograms
- Computes diversity metrics

**Outputs:**
- `results/A4_sobol_probes/plots/phase_heatmap.png`
- `results/A4_sobol_probes/plots/phase_histogram.png`
- `results/A4_sobol_probes/metrics.txt`

#### Task 5: A5 - Halton Probes
- Generates Halton low-discrepancy probes
- Compares with continuous random probes
- Creates phase heatmaps and histograms
- Computes diversity metrics

**Outputs:**
- `results/A5_halton_probes/plots/phase_heatmap.png`
- `results/A5_halton_probes/plots/phase_histogram.png`
- `results/A5_halton_probes/metrics.txt`

### Phase B: Limited Probing Analysis

#### Task 6: B1 - M Variation Study
- Tests different sensing budgets: M ∈ {2, 4, 8, 16, 32}
- Trains ML model for each M value
- Plots η (power ratio) vs M
- Compares continuous and binary probes

**Outputs:**
- `results/B1_M_variation/plots/eta_vs_M.png`
- `results/B1_M_variation/plots/eta_comparison_bar.png`
- `results/B1_M_variation/metrics.txt`

**Note:** This task trains multiple models and takes longer to run.

#### Task 7: B2 - Top-m Selection
- Evaluates Top-1, 2, 4, 8 performance
- Plots η vs top-m curves
- Creates accuracy comparison
- Generates summary table

**Outputs:**
- `results/B2_top_m_selection/plots/top_m_curves.png`
- `results/B2_top_m_selection/plots/top_m_summary_table.png`
- `results/B2_top_m_selection/metrics.txt`

#### Task 8: B3 - Baseline Comparison
- Implements baselines: Random 1/K, Random M/K, Best Observed
- Trains ML model
- Creates bar plot comparing ML vs all baselines
- Reports improvement percentages

**Outputs:**
- `results/B3_baselines/plots/baseline_comparison.png`
- `results/B3_baselines/metrics.txt`

### Phase C: Scaling Study

#### Task 9: C1 - Scale K (Placeholder)
- Tests K = 32, 64, 128
- Plots η vs K for fixed M/K ratio
- (Full implementation coming soon)

#### Task 10: C2 - Phase Resolution (Placeholder)
- Compares continuous, 1-bit, 2-bit, Hadamard
- Summary plot of best η for each type vs M
- (Full implementation coming soon)

### Phase D: Quality Control

#### Task 11: D1 - Seed Variation (Placeholder)
- Trains with seeds 1, 2, 3, 4, 5
- Creates boxplot of η distribution
- Reports mean ± std
- (Full implementation coming soon)

#### Task 12: D2 - Sanity Checks (Placeholder)
- Verifies training loss decreases
- Verifies validation η increases
- Flags any issues
- (Full implementation coming soon)

### Phase E: Documentation

#### Task 13: E1 - Summary (Placeholder)
- Generates one-page markdown summary
- Includes key findings and figures
- (Full implementation coming soon)

#### Task 14: E2 - Comparison Plots (Placeholder)
- Generates master comparison figures
- η vs M for all probe types
- Top-m performance curves
- (Full implementation coming soon)

## Command-Line Options

```
--task TASK     Task number(s) to run (e.g., "1" or "1,6,8" or "all")
                If not specified, runs in interactive mode
--N N           Number of RIS elements (default: 32)
--K K           Total number of probes in bank (default: 64)
--M M           Sensing budget (probes measured per sample) (default: 8)
--seed SEED     Random seed (default: 42)
```

## Results Directory Structure

Each task saves results to its own folder:

```
results/
├── A1_binary_probes/
│   ├── plots/
│   │   ├── phase_heatmap.png
│   │   └── phase_histogram.png
│   └── metrics.txt
├── A2_hadamard_probes/
│   ├── plots/
│   └── metrics.txt
├── A3_diversity_analysis/
│   ├── plots/
│   ├── diversity_summary.csv
│   └── metrics.txt
├── A4_sobol_probes/
│   ├── plots/
│   └── metrics.txt
├── A5_halton_probes/
│   ├── plots/
│   └── metrics.txt
├── B1_M_variation/
│   ├── plots/
│   └── metrics.txt
...
```

## Example Workflows

### Quick Test with Small Parameters

```bash
# Test probe diversity with small parameters
python experiment_runner.py --task 3 --N 8 --K 16 --M 4

# Test baseline comparison (trains a model)
python experiment_runner.py --task 8 --N 16 --K 32 --M 8
```

### Full Probe Analysis Suite

```bash
# Run all Phase A tasks (probe design)
python experiment_runner.py --task 1,2,3,4,5 --N 32 --K 64 --M 8
```

### M Variation Study

```bash
# Test different sensing budgets
python experiment_runner.py --task 6 --N 32 --K 64
# Will test M = 2, 4, 8, 16, 32 automatically
```

### Comprehensive Experiment

```bash
# Run all implemented tasks with custom settings
python experiment_runner.py --task all --N 64 --K 128 --M 16 --seed 123
```

## Tips

1. **Start Small**: Test with small N and K values first (e.g., N=8, K=16) to verify everything works before running full experiments.

2. **Training Tasks**: Tasks B1, B2, B3 involve ML training and take longer. Adjust data sizes in the code if needed for faster testing.

3. **Interactive Mode**: Use interactive mode to explore tasks one at a time and adjust settings between runs.

4. **Parallel Execution**: Currently tasks run sequentially. For parallel execution, run multiple CLI commands in different terminals.

5. **Auto-advance**: When running multiple tasks in sequence, the runner waits 5 seconds between tasks (press Enter to skip the wait).

6. **Memory**: Large K values with full training datasets can use significant memory. Monitor system resources.

## Dependencies

All dependencies are listed in `requirements.txt`:
- numpy >= 1.21.0
- torch >= 1.12.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- tqdm >= 4.62.0
- pandas >= 1.3.0
- scipy >= 1.7.0

Install with:
```bash
pip install -r requirements.txt
```

## Extending the Framework

To add a new task:

1. Create a new file in `experiments/tasks/` (e.g., `task_x1_mytask.py`)
2. Implement a `run_task_x1(N, K, M, seed, results_dir, verbose)` function
3. Add the task to the `TASKS` dictionary in `experiment_runner.py`
4. The menu will automatically include the new task

## Troubleshooting

**Issue**: Import errors
- **Solution**: Make sure you're in the repository root directory and dependencies are installed

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in task code or reduce training data size (n_train, n_val, n_test)

**Issue**: Tasks take too long
- **Solution**: Use smaller N, K values for testing, or reduce num_epochs in training tasks

**Issue**: Plots not displaying
- **Solution**: Plots are saved to files automatically. Check the results directory for PNG files.

## Contributing

When implementing placeholder tasks (C, D, E phases):
1. Follow the same pattern as existing tasks
2. Create results directory structure
3. Save plots and metrics consistently
4. Return a dictionary with results
5. Update this README with task details
