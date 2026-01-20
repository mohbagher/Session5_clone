# Quick Start: New Dashboard Architecture

## Immediate Usage (No Migration Needed)

### Option 1: Test in Isolation
```python
# In a new notebook cell - test without affecting your current setup
exec(open('test_complete_dashboard.py').read())
```

This runs all tests and shows you the new dashboard is working.

### Option 2: Display New Dashboard
```python
from dashboard import create_complete_interface

# Create interface
ui, widgets = create_complete_interface()

# Display it
display(ui)

# Get current configuration
from dashboard.main import get_widget_values
config = get_widget_values(widgets)
print(config)
```

### Option 3: Use With Existing Callbacks
```python
from dashboard import create_complete_interface
from callbacks import setup_callbacks  # Your existing callbacks

# Create interface
ui, widgets = create_complete_interface()

# Setup callbacks (your existing code works!)
setup_callbacks(widgets)

# Display
display(ui)
```

## Key Features

### 6 Clean Tabs

1. **System** - N, K, M, probes, channel params
2. **Physics & Realism** - Channel sources, impairment profiles
3. **Model** - Architecture, transfer learning
4. **Training** - Hyperparameters, optimizer
5. **Evaluation** - Top-m, comparison, multi-seed
6. **Visualization** - Plot selection, styling

### Components Always Visible

- **Stack Manager** (below tabs) - Add/remove experiments
- **Action Buttons** - Run, save, load, reset
- **Status Display** - Progress, metrics, logs
- **Results Display** - Summary, plots, export

## File Organization
```
dashboard/
├── tabs/              # Each tab = separate file
│   ├── tab_system.py
│   ├── tab_physics.py
│   ├── tab_model.py
│   ├── tab_training.py
│   ├── tab_evaluation.py
│   └── tab_visualization.py
│
├── components/        # Reusable UI components
│   ├── stack_manager.py
│   ├── buttons.py
│   ├── status_display.py
│   └── results_display.py
│
└── main.py           # Orchestrator (brings it all together)
```

## Common Tasks

### Get All Widget Values
```python
from dashboard.main import get_widget_values

config = get_widget_values(widgets)
# Returns dict with all parameters
```

### Access Individual Widgets
```python
# All widgets accessible via dictionary
widgets['N'].value = 64
widgets['realism_profile'].value = 'moderate_impairments'
widgets['button_run_experiment'].on_click(my_callback)
```

### Modify Tab Content

Each tab file is standalone. Edit any tab without touching others:
```python
# Edit dashboard/tabs/tab_system.py to customize System tab
# Edit dashboard/tabs/tab_physics.py to customize Physics tab
# etc.
```

## Comparison with Old Dashboard

| Feature | Old | New |
|---------|-----|-----|
| Total files | 1 (widgets.py) | 11+ (organized) |
| Lines per file | 2000+ | ~200 each |
| Tab editing | Scroll through one file | Open specific tab file |
| Widget access | Same dict | Same dict ✓ |
| Callbacks | Compatible | Compatible ✓ |
| Configs | Compatible | Compatible ✓ |

## Phase 1 Physics Integration

The new dashboard is fully integrated with Phase 1 physics:
```python
# Automatic physics tracking
widgets['channel_source'].value = 'python_synthetic'
widgets['realism_profile'].value = 'moderate_impairments'

# Configuration includes physics metadata
config = get_widget_values(widgets)
print(config['channel_source'])     # 'python_synthetic'
print(config['realism_profile'])    # 'moderate_impairments'
```

## Testing

Run comprehensive tests:
```python
exec(open('test_complete_dashboard.py').read())
```

Expected output:
```
[TEST 1] Importing all modules... [OK]
[TEST 2] Creating individual tabs... [OK]
[TEST 3] Creating components... [OK]
[TEST 4] Creating unified dashboard... [OK]
[TEST 5] Creating complete interface... [OK]
[TEST 6] Extracting configuration... [OK]
[TEST 7] Testing widget access... [OK]
[TEST 8] Testing Phase 1 physics integration... [OK]

ALL TESTS PASSED!
```

## Need Help?

1. **See working example**: `test_complete_dashboard.py`
2. **Read migration guide**: `docs/MIGRATION_GUIDE.md`
3. **Check individual tabs**: `dashboard/tabs/tab_*.py`
4. **Review main file**: `dashboard/main.py`

---

**The new architecture is ready to use immediately - no migration required to test it!**
