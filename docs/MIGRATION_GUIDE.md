# Migration Guide: Old → New Dashboard Architecture

## Overview

This guide helps you transition from the monolithic `widgets.py` to the clean, modular architecture.

## Architecture Comparison

### Old Structure (Monolithic)
```
Session5_clone/
├── widgets.py                    # Everything in one file (2000+ lines)
├── callbacks.py                  # Event handlers
├── experiment_runner.py          # Experiment logic
├── config.py                     # Configuration
└── [other scattered files]
```

### New Structure (Modular)
```
Session5_clone/
├── dashboard/
│   ├── tabs/                    # Each tab = separate file
│   │   ├── tab_system.py
│   │   ├── tab_physics.py
│   │   ├── tab_model.py
│   │   ├── tab_training.py
│   │   ├── tab_evaluation.py
│   │   └── tab_visualization.py
│   ├── components/              # Reusable components
│   │   ├── stack_manager.py
│   │   ├── buttons.py
│   │   ├── status_display.py
│   │   └── results_display.py
│   ├── main.py                  # Dashboard orchestrator
│   └── callbacks.py             # Event handlers (same)
├── physics/                     # Phase 1 physics module
├── models/                      # Neural network models
├── data/                        # Data generation
└── [other organized folders]
```

## Migration Steps

### Step 1: Test New Dashboard (NO CHANGES TO OLD CODE YET)
```python
# In a new notebook cell:
exec(open('test_complete_dashboard.py').read())
```

Expected output: All tests pass ✓

### Step 2: Display New Dashboard (Side-by-Side Comparison)
```python
# Test new dashboard
from dashboard import create_complete_interface
ui_new, widgets_new = create_complete_interface()
display(ui_new)

# Old dashboard still works
# (your existing notebook code unchanged)
```

### Step 3: Update Your Notebook (When Ready)

**OLD CODE (in your notebook):**
```python
from widgets import create_unified_dashboard, get_all_widgets
from callbacks import setup_callbacks

dashboard = create_unified_dashboard()
widgets = get_all_widgets()
setup_callbacks(widgets)
display(dashboard)
```

**NEW CODE (replace with this):**
```python
from dashboard import create_complete_interface
from callbacks import setup_callbacks  # Same file, works as-is

ui, widgets = create_complete_interface()
setup_callbacks(widgets)  # Same callbacks work!
display(ui)
```

### Step 4: Update Callbacks (Minor Changes)

Your `callbacks.py` should work mostly as-is. Only change needed:

**OLD:**
```python
def config_to_dict(wd):
    config = {
        'N': wd['N'].value,
        'K': wd['K'].value,
        # ... etc
    }
```

**NEW:**
```python
from dashboard.main import get_widget_values

def config_to_dict(wd):
    # Option 1: Use helper function (recommended)
    config = get_widget_values(wd)

    # Option 2: Manual (same as before, still works)
    config = {
        'N': wd['N'].value,
        'K': wd['K'].value,
        # ... etc
    }
```

The `get_widget_values()` helper is optional but recommended - it extracts all config automatically.

## What Stays The Same

✅ **callbacks.py** - Works as-is (widget dict keys unchanged)
✅ **experiment_runner.py** - No changes needed
✅ **All your existing saved configs** - Compatible
✅ **All your experiment results** - Compatible

## What's Different

🔄 **Tab organization** - Cleaner, each in separate file
🔄 **Import statements** - `from dashboard import ...` instead of `from widgets import ...`
✅ **Everything else** - Same functionality

## Testing Checklist

Before switching completely:

- [ ] Run `test_complete_dashboard.py` - all tests pass
- [ ] Display new dashboard - UI renders correctly
- [ ] Click through all 6 tabs - no errors
- [ ] Test "Add to Stack" - works
- [ ] Extract config with `get_widget_values()` - returns dict
- [ ] Physics profile selection - updates info box
- [ ] Custom impairments toggle - shows/hides advanced settings

## Rollback Plan

If anything breaks, you can instantly rollback:
```python
# Use old dashboard (no changes needed to your code)
from widgets import create_unified_dashboard, get_all_widgets
dashboard = create_unified_dashboard()
widgets = get_all_widgets()
display(dashboard)
```

Your old `widgets.py` file is untouched and still works!

## Benefits of New Architecture

1. **Maintainability** - Each tab ~200 lines vs. one 2000+ line file
2. **Collaboration** - Multiple people can edit different tabs
3. **Testing** - Test individual tabs/components
4. **Reusability** - Components used across multiple dashboards
5. **Scalability** - Easy to add new tabs/features
6. **Professional** - Industry-standard Python package structure

## File Mapping Reference

| Old Location | New Location |
|--------------|-------------|
| `widgets.py` (Tab 1) | `dashboard/tabs/tab_system.py` |
| `widgets.py` (Tab 2 - Physics) | `dashboard/tabs/tab_physics.py` |
| `widgets.py` (Tab 3) | `dashboard/tabs/tab_model.py` |
| `widgets.py` (Tab 4) | `dashboard/tabs/tab_training.py` |
| `widgets.py` (Tab 5) | `dashboard/tabs/tab_evaluation.py` |
| `widgets.py` (Tab 6) | `dashboard/tabs/tab_visualization.py` |
| `widgets.py` (Stack) | `dashboard/components/stack_manager.py` |
| `widgets.py` (Buttons) | `dashboard/components/buttons.py` |
| `widgets.py` (Status) | `dashboard/components/status_display.py` |
| `widgets.py` (Results) | `dashboard/components/results_display.py` |
| `widgets.py` (Main) | `dashboard/main.py` |

## Next Steps

1. **Test** - Run `test_complete_dashboard.py`
2. **Compare** - Display both old and new dashboards side-by-side
3. **Migrate** - Update your notebook when ready
4. **Enjoy** - Clean, maintainable code structure!

## Questions?

- Check `test_complete_dashboard.py` for working examples
- Review individual tab files in `dashboard/tabs/`
- Look at `dashboard/main.py` to see how it all connects

---

**Remember:** The old dashboard still works! Take your time to test the new one.
