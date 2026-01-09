#!/bin/bash

echo "========================================================================"
echo "  RIS Probe-Based ML Experiment Framework - Demo"
echo "========================================================================"
echo ""
echo "This demo will run several quick tasks to demonstrate the framework."
echo "All tasks use small parameters (N=8, K=16, M=4) for fast execution."
echo ""

# Task 1: Binary Probes
echo "Running Task 1: Binary Probes..."
python experiment_runner.py --task 1 --N 8 --K 16 --M 4 --seed 42
echo ""
echo "Press Enter to continue..."
read

# Task 2: Hadamard Probes
echo "Running Task 2: Hadamard Probes..."
python experiment_runner.py --task 2 --N 8 --K 16 --M 4 --seed 42
echo ""
echo "Press Enter to continue..."
read

# Task 3: Diversity Analysis
echo "Running Task 3: Diversity Analysis..."
python experiment_runner.py --task 3 --N 8 --K 16 --M 4 --seed 42
echo ""

echo "========================================================================"
echo "  Demo Complete!"
echo "========================================================================"
echo ""
echo "Check the results/ directory for generated plots and metrics."
echo "To run the full interactive menu, use: python experiment_runner.py"
echo ""
