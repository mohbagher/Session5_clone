# Phase 2: MATLAB Backend Integration

## Overview

Phase 2 adds MATLAB as a verified computational backend for channel generation, using industry-standard MathWorks toolboxes.

## Features

✅ **Persistent MATLAB Engine** - Single session across runs  
✅ **Verified Toolboxes** - Communications, 5G, Phased Array  
✅ **Multiple Scenarios** - Rayleigh, CDL-RIS, Rician, TDL  
✅ **Graceful Fallback** - Falls back to Python if MATLAB unavailable  
✅ **Full Metadata** - Every result tagged with backend, toolbox, version  

## Quick Start

### 1. Install MATLAB Engine
```bash
# Locate MATLAB installation
cd /path/to/MATLAB/R2023b/extern/engines/python

# Install
python setup.py install
```

### 2. Run Setup
```bash
python setup_phase2.py
```

### 3. Test Integration
```bash
python tests/test_matlab_backend.py
```

### 4. Use in Dashboard

1. Open dashboard: `notebooks/RIS_PhD_Ultimate_Dashboard.ipynb`
2. Navigate to **Physics & Realism** tab
3. Set **Physics Backend** to **MATLAB**
4. Select scenario (e.g., **CDL-RIS Channel**)
5. Configure parameters (carrier frequency, CDL profile, etc.)
6. Run experiment!

## Available Scenarios

| Scenario | Toolbox | Description |
|----------|---------|-------------|
| `rayleigh_basic` | Communications | Basic Rayleigh fading |
| `cdl_ris` | 5G | 3GPP CDL channel with RIS |
| `rician_los` | Communications | Rician fading (LOS) |
| `tdl_urban` | 5G | TDL urban channel |

## Backend Selection Logic
```
User selects backend → Config updated → Experiment runner reads config
                                              ↓
                                      Backend == MATLAB?
                                       ↙            ↘
                                     Yes            No
                                      ↓              ↓
                              MATLAB Engine     Python numpy
                                      ↓              ↓
                              Scenario selected   Synthetic
                                      ↓              ↓
                                   Channels generated
                                           ↓
                              Flow through Phase 1 impairments
                                           ↓
                                   Training/evaluation
```

## Result Metadata

Every result includes backend information:
```json
{
  "metadata": {
    "backend": "matlab",
    "channel_generation": {
      "toolbox": "5g",
      "scenario": "cdl_ris",
      "function": "generate_cdl_ris_channel",
      "matlab_version": "9.14.0.2206163 (R2023a)",
      "reference": "https://mathworks.com/...",
      "carrier_frequency": 28000000000,
      "delay_profile": "CDL-C",
      "doppler_shift": 5
    }
  }
}
```

## Expanding to New Scenarios

Add new scenarios in `physics/matlab_backend/toolbox_registry.py`:
```python
SCENARIO_TEMPLATES['my_scenario'] = ScenarioTemplate(
    name='My Custom Scenario',
    toolbox='communications',
    description='Custom channel model',
    matlab_function='my_matlab_function',
    default_params={...},
    reference='https://...'
)
```

## Troubleshooting

See `docs/MATLAB_TROUBLESHOOTING.md`

## Future Roadmap

- **Phase 2.2**: Waveform generation (OFDM, 5G NR)
- **Phase 2.3**: Verified metrics (BER, EVM, capacity)
- **Phase 2.4**: Batch processing, parallel execution