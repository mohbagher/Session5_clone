# PhD Research Dashboard - User Guide

## 📖 Overview

The **PhD_Research_Dashboard.ipynb** is a comprehensive, production-ready Jupyter notebook that provides complete control over all experimental parameters through an intuitive widget-based interface. This notebook is specifically designed for PhD research on RIS probe-based ML systems.

## 🚀 Quick Start

### Prerequisites

Ensure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- torch >= 1.12.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- tqdm >= 4.62.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- ipywidgets >= 8.0.0

### Running the Notebook

1. **Launch Jupyter**:
   ```bash
   jupyter notebook notebooks/PhD_Research_Dashboard.ipynb
   ```
   or for JupyterLab:
   ```bash
   jupyter lab notebooks/PhD_Research_Dashboard.ipynb
   ```

2. **Execute All Cells**: 
   - In Jupyter: `Cell → Run All`
   - In JupyterLab: `Run → Run All Cells`

3. **Configure Parameters**: Navigate through the 5 tabs and adjust parameters as needed

4. **Run Experiment**: Click the "🚀 RUN EXPERIMENT" button

5. **View Results**: Click "📊 Show Results Dashboard" after completion

6. **Export Results**: Click "💾 Export Results" to save outputs

## 📋 Notebook Structure

### Cell 1: Title & Description
- Project overview
- Key features
- Quick start instructions

### Cell 2: Imports & Setup
- All library imports
- Device detection (CPU/GPU)
- Environment validation

### Cell 3: Widget Control Panel (5 Tabs)

#### Tab 1: 🎛️ System & Physics
Configure physical system parameters:
- **N**: Number of RIS elements (8-256, step 8)
- **K**: Codebook size (16-512, step 16)
- **M**: Sensing budget (1-K)
- **P_tx**: Transmit power (0.1-10)
- **σ²_h, σ²_g**: Channel variances
- **Phase Mode**: Continuous or Discrete (1-8 bits)
- **Channel Type**: Rayleigh, Rician, Sparse, Correlated, Time-Varying
- **Probe Methods**: Multiple selection (continuous, binary, 2bit, hadamard, sobol, halton)

#### Tab 2: 🧠 Model Architecture
Choose and configure model architecture:
- **Model Types**: MLP, CNN, LSTM, GRU, ResNet-MLP, Attention, Transformer, Hybrid-CNN-LSTM
- **MLP Config**: Dynamic layer controls (1-10 layers, 16-2048 units each)
- **CNN Config**: Conv layers, pooling, FC layers
- **RNN Config**: Hidden size, layers, bidirectional
- **Transformer Config**: d_model, attention heads, layers
- **Regularization**: Dropout, batch norm, layer norm, weight initialization

#### Tab 3: ⚙️ Training Configuration
Configure training process:
- **Data**: Train/val/test samples, seed, normalization
- **Training**: Epochs, batch size, early stopping
- **Optimizer**: Adam, AdamW, SGD, RMSprop, AdaGrad, Adadelta, Adamax
- **Learning Rate**: LogSlider (1e-5 to 1e-1)
- **LR Scheduler**: StepLR, MultiStepLR, CosineAnnealing, OneCycleLR, etc.
- **Loss Function**: CrossEntropy, Focal Loss, Label Smoothing
- **Advanced Reg**: Mixup, gradient clipping, L1 regularization

#### Tab 4: 📊 Evaluation & Comparison
Configure evaluation and analysis:
- **Multi-Model Comparison**: Compare multiple architectures
- **Top-m Values**: Specify accuracy metrics
- **Baselines**: Random, Best Observed, Oracle
- **Statistical Analysis**: Multi-seed runs, confidence intervals
- **Performance Profiling**: Time tracking, memory usage, checkpoints

#### Tab 5: 🎨 Visualization & Export
Configure outputs:
- **Plot Selection**: 22+ plot types
  - Training curves, CDF, PDF, Box plots, Violin plots
  - Heatmaps, Scatter plots, Bar comparisons
  - ROC curves, Confusion matrix, 3D surfaces
- **Plot Customization**: Figure size, DPI, style, color palette
- **Export Formats**: CSV, JSON, YAML, Excel, HDF5, Pickle
- **Report Generation**: PDF reports with all results
- **Verbosity**: Output level, progress bars, logging

### Cell 4: Dynamic Widget Interactions
Implements smart widget behavior:
- Auto-update M max based on K
- Phase bits enabled only in discrete mode
- Channel-specific parameters show/hide
- Model architecture adapts to model type
- Optimizer/scheduler/loss parameters adapt dynamically
- Configuration validation

### Cell 5: Configuration Builder
Collects all widget values into Config object

### Cell 6: Experiment Runner
Main execution engine:
- Large "🚀 RUN EXPERIMENT" button
- Progress tracking
- Error handling
- Multi-seed and multi-probe support

### Cell 7: Real-time Progress Display
Shows live experiment progress:
- Current status
- Training metrics
- Time estimates

### Cell 8: Results Dashboard
Comprehensive results visualization:
- Summary statistics table
- Best model identification
- Interactive plots
- Performance comparisons

### Cell 9: Export & Save Functions
Export results in multiple formats:
- CSV for Excel/R/MATLAB
- JSON for structured data
- YAML for configs
- Pickle for Python objects

### Cell 10: Helper Functions
Utility functions:
- `reset_to_defaults()`: Reset all widgets
- `load_preset('quick_test')`: Load quick test config
- `load_preset('full_training')`: Load full training config
- `load_preset('paper_reproduction')`: Load publication config
- `estimate_training_time()`: Predict training duration
- `get_system_info()`: Display hardware info
- `load_config_from_yaml(path)`: Load saved configs

## 🎯 Usage Scenarios

### Scenario 1: Quick Exploration (2 minutes)
```
1. Open notebook and run all cells
2. Change: N=16, K=32, epochs=10
3. Click RUN EXPERIMENT
4. View results
```

### Scenario 2: Model Comparison
```
1. Navigate to Tab 4 (Evaluation)
2. Enable "Compare Multiple Models"
3. Select models: Baseline_MLP, Deep_MLP, Tiny_MLP
4. Set epochs=50
5. Run experiment
6. Compare results in dashboard
```

### Scenario 3: Probe Method Analysis
```
1. Tab 1: Select multiple probe methods (continuous, hadamard, sobol)
2. Tab 3: Set sufficient epochs (50+)
3. Run experiment
4. View CDF and Bar Comparison plots
```

### Scenario 4: Publication-Ready Results
```
1. Click "Full Training Preset" button
2. Tab 4: Enable "Multi-Seed Run"
3. Tab 4: Enter seeds: "42,43,44,45,46,47,48,49,50,51"
4. Tab 5: Select all desired plots
5. Tab 5: Enable "Generate PDF Report"
6. Run experiment (may take hours)
7. Export results
```

### Scenario 5: Hyperparameter Sweep (Manual)
```
For each learning rate in [1e-4, 1e-3, 1e-2]:
  1. Set learning rate in Tab 3
  2. Run experiment
  3. Note results
  4. Export
```

## ⚙️ Dynamic Features

### Auto-Updates
- **M ≤ K**: M slider automatically adjusts when K changes
- **Phase Bits**: Only enabled when Discrete mode selected
- **Channel Params**: Show/hide based on channel type
- **Model Params**: Adapt to selected model architecture
- **Optimizer Params**: Show relevant parameters only
- **Scheduler Params**: Show relevant parameters only

### Validation
- Configuration validation before running
- Warning for unusual parameter combinations
- Error prevention for invalid inputs
- Helpful error messages

## 📊 Output Structure

When saving results, the following structure is created:

```
results/experiment_2024-01-07_15-30-45/
├── config.yaml                 # Full configuration
├── results.csv                 # Tabular results
├── results.json                # Structured results
├── results.yaml                # YAML results
├── results.pkl                 # Python pickle
├── plots/                      # Generated plots
│   ├── training_curves.png
│   ├── cdf.png
│   ├── bar_comparison.png
│   └── ...
└── report.pdf                  # Optional PDF report
```

## 🎓 Research Best Practices

### For Reproducibility
1. Always set a fixed seed
2. Enable "Save Results" 
3. Save config.yaml for every experiment
4. Use multi-seed runs for statistical significance
5. Export in multiple formats

### For Publication
1. Use "Full Training Preset"
2. Enable multi-seed runs (10+ seeds)
3. Select all relevant plots
4. Enable PDF report generation
5. Use high DPI (300) for plots
6. Save in PDF/SVG formats

### For Development
1. Use "Quick Test Preset" for rapid iteration
2. Start with small N, K values
3. Increase epochs gradually
4. Monitor training curves for convergence
5. Use validation interval = 1

## 🐛 Troubleshooting

### Widgets Not Displaying
- Ensure ipywidgets is installed: `pip install ipywidgets`
- For JupyterLab: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
- Restart Jupyter kernel

### CUDA Out of Memory
- Reduce batch size
- Reduce model size (fewer/smaller layers)
- Reduce dataset size
- Enable memory tracking to identify issues

### Training Too Slow
- Reduce training samples
- Reduce epochs
- Reduce model complexity
- Use larger batch size (if memory allows)
- Ensure GPU is being used (check Cell 2 output)

### Import Errors
- Check all requirements: `pip install -r requirements.txt`
- Verify Python version >= 3.8
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

## 💡 Tips & Tricks

1. **Save Configs**: After finding good parameters, export config and reload later
2. **Presets**: Use presets for common configurations
3. **Incremental Testing**: Start small, gradually increase complexity
4. **Progress Monitoring**: Watch training curves to catch issues early
5. **Compare Methods**: Use multi-model comparison to identify best approach
6. **Statistical Significance**: Use multi-seed runs for reliable conclusions
7. **Export Early**: Save results after each successful run
8. **Documentation**: Use output directory names to track experiments

## 📚 Additional Resources

- **Main README**: `/README.md` - Project overview
- **Experiment Runner**: `/EXPERIMENT_RUNNER.md` - CLI interface
- **Implementation Details**: `/IMPLEMENTATION_SUMMARY.md` - Technical details
- **Usage Examples**: `/USAGE_EXAMPLES.md` - Code examples

## 🤝 Contributing

Found a bug or have a feature request? Please open an issue on GitHub.

## 📄 License

MIT License - See LICENSE file for details

## 📧 Contact

For questions or collaboration, open an issue on the GitHub repository.

---

**Happy Researching! 🎓🚀**
