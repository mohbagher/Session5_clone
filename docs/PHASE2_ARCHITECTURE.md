# Phase 2: Physics-First Architecture

## Overview

Phase 2 refactors the physics engine from a monolithic design to a **composable architecture** using dependency injection. This enables:

- **Ablation studies**: Swap physics components independently
- **Hardware realism**: Varactor coupling, thermal drift, channel aging
- **Backend abstraction**: MATLAB/Python support
- **Optimization foundation**: Bayesian and grid search optimizers

## Architecture

```
src/ris_platform/
├── core/
│   └── interfaces.py          # Abstract base classes (ABCs)
├── physics/
│   ├── models.py              # IdealPhysicsModel, RealisticPhysicsModel
│   └── components/
│       ├── unit_cell/         # IdealUnitCell, VaractorUnitCell
│       ├── coupling/          # NoCoupling, GeometricCoupling
│       ├── wavefront/         # PlanarWavefront, SphericalWavefront
│       └── aging/             # JakesAging
├── backend/
│   ├── matlab.py              # MATLABBackend (wraps session_manager)
│   └── python_synthetic.py   # PythonSyntheticBackend
├── probing/
│   └── structured.py          # RandomProbing, SobolProbing, HadamardProbing
└── optimization/
    ├── bayesian.py            # BayesianOptimizer (Optuna)
    └── grid_search.py         # GridSearchOptimizer
```

## Key Features

### 1. Composable Physics Components

```python
from src.ris_platform.physics.models import RealisticPhysicsModel
from src.ris_platform.physics.components.unit_cell import VaractorUnitCell
from src.ris_platform.physics.components.coupling import GeometricCoupling

# Create realistic model with dependency injection
physics = RealisticPhysicsModel(
    unit_cell=VaractorUnitCell(coupling_strength=0.3),
    coupling=GeometricCoupling(coupling_strength=0.5)
)

# Compute received power
power = physics.compute_received_power(h, g, phases)
```

### 2. Unit Cell Models

#### Ideal Unit Cell
```python
Γ(θ) = exp(jθ)
```
- Perfect amplitude: |Γ| = 1
- No thermal drift

#### Varactor Unit Cell (Hardware Realistic)
```python
A(θ) = 1 - α(1 + cos(θ))
Γ(θ, T) = A(θ) * exp(j(θ + β*(T - T0)))
```
- α: Coupling strength (0.3 typical)
- β: Thermal drift coefficient (0.02 rad/°C)
- **Reference**: Dai et al., IEEE Access 2020

### 3. Coupling Models

#### No Coupling
```python
Γ_coupled = Γ_uncoupled
```
- Baseline assumption
- Elements operate independently

#### Geometric Coupling (Bessel Function)
```python
C_ij = J₀(k * d_ij)
Γ_coupled = C @ Γ_uncoupled
```
- J₀: Bessel function of the first kind
- k: Wave number (2π/λ)
- d_ij: Distance between elements
- **Reference**: Pozar, Microwave Engineering, Ch. 8

### 4. Wavefront Models

#### Planar Wavefront (Far-Field)
```python
y = g^H @ (Γ ⊙ h)
```
- Standard textbook model
- Valid when distance >> Rayleigh distance

#### Spherical Wavefront (Near-Field 6G)
```python
y = Σ_n g_n * Γ_n * h_n * exp(-jk*r_n) / r_n
```
- Distance-dependent phase and path loss
- Critical for 6G near-field scenarios
- **Reference**: Björnson et al., arXiv:2110.06661

### 5. Channel Aging

#### Jakes' Doppler Model
```python
ρ(τ) = J₀(2π * f_d * τ)
h(t+Δt) = ρ * h(t) + √(1-ρ²) * innovation
```
- AR(1) temporal correlation
- Configurable Doppler frequency
- **Reference**: Jakes, 1974

## Backward Compatibility

The `dashboard/experiment_runner.py` includes **factory functions** that translate the old configuration format to the new components:

```python
def create_physics_model(config: Dict) -> PhysicsModel:
    """Translate config dict → instantiated physics components."""
    if config['realism_profile'] == 'ideal':
        return IdealPhysicsModel()
    else:
        unit_cell = VaractorUnitCell(...)
        coupling = GeometricCoupling(...)
        return RealisticPhysicsModel(unit_cell, coupling, ...)
```

The `run_single_experiment()` signature **remains unchanged**, ensuring full compatibility with the existing dashboard UI.

## Testing

Run comprehensive validation tests:

```bash
python tests/test_phase2_architecture.py
```

Test suites:
- ✓ Unit Cell Models (ideal, varactor)
- ✓ Coupling Models (none, geometric)
- ✓ Wavefront Models (planar, spherical)
- ✓ Channel Aging (Jakes)
- ✓ Physics Models (ideal, realistic)
- ✓ Channel Backends (Python, MATLAB)
- ✓ Probing Strategies (random, Sobol, Hadamard)
- ✓ Factory Functions (backward compatibility)

## Quick Start

### Demo Script
```bash
python examples/demo_phase2_architecture.py
```

### Basic Usage
```python
from src.ris_platform.physics.models import IdealPhysicsModel

# Create model
physics = IdealPhysicsModel()

# Generate channels
h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
g = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
phases = np.random.rand(N) * 2 * np.pi

# Compute power
power = physics.compute_received_power(h, g, phases)
```

### Ablation Study
```python
from src.ris_platform.physics.models import RealisticPhysicsModel
from src.ris_platform.physics.components.unit_cell import IdealUnitCell, VaractorUnitCell
from src.ris_platform.physics.components.coupling import NoCoupling, GeometricCoupling

# Compare configurations
configs = [
    RealisticPhysicsModel(IdealUnitCell(), NoCoupling()),
    RealisticPhysicsModel(VaractorUnitCell(0.3), NoCoupling()),
    RealisticPhysicsModel(VaractorUnitCell(0.3), GeometricCoupling(0.5)),
]

for physics in configs:
    power = physics.compute_received_power(h, g, phases)
    print(f"Power: {power:.4f}")
```

## Publication Enablement

| Paper | Component | Ablation Study |
|-------|-----------|----------------|
| Paper 1: Unit Cell | `VaractorUnitCell` | Amplitude-phase coupling impact |
| Paper 2: Coupling | `GeometricCoupling` | Bessel vs. no coupling |
| Paper 3: Near-Field 6G | `SphericalWavefront` | Spherical vs. planar |
| Paper 4: Adaptive Probing | `SobolProbing` | Low-discrepancy vs. random |
| Paper 5: AutoML | `BayesianOptimizer` | Hyperparameter sensitivity |

## Dependencies

```txt
numpy>=1.21.0
scipy>=1.7.0
torch>=1.12.0
optuna>=3.0.0  # Optional (for BayesianOptimizer)
```

## Protected Zones

✅ **NO MODIFICATIONS** to:
- `dashboard/components/` - UI components preserved
- `dashboard/tabs/` - Tab implementations preserved
- `physics/matlab_backend/` - Session manager wrapped, not replaced

## Scientific References

1. **Varactor Model**: Dai et al., "Hardware Impairments Adaptive Deep Learning for RIS-Aided Multi-User Downlink Communication Systems," IEEE Access, 2020
2. **Coupling Model**: Pozar, "Microwave Engineering," 4th Edition, Chapter 8
3. **Near-Field Model**: Björnson et al., "Power Scaling Laws and Near-Field Behaviors of Massive MIMO and Intelligent Reflecting Surfaces," arXiv:2110.06661, 2021
4. **Jakes Model**: Jakes, "Microwave Mobile Communications," Wiley, 1974

## Architecture Benefits

1. **Modularity**: Each component can be developed/tested independently
2. **Reusability**: Components can be mixed and matched
3. **Extensibility**: New components can be added without breaking existing code
4. **Testability**: Each component has focused unit tests
5. **Research**: Enables ablation studies and component comparison
6. **Production**: Clean interfaces for hardware deployment

## Version

Phase 2 Architecture v2.0.0
