# Usage Guide

## Installation

### Installing from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/MaxM-1/TDSE_test.git
   cd tdse_solver
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. Install with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Basic Usage

### Simple Example

```python
import numpy as np
from tdse_solver import TDSESolver, GaussianWavePacket
from tdse_solver.potentials import SquareBarrier

# Set up spatial grid
x = np.linspace(-50, 50, 1024)

# Create initial wave packet
psi0 = GaussianWavePacket(x, x0=-20, k0=2.0, sigma=2.0)

# Define potential
V = SquareBarrier(x, height=1.5, width=5.0)

# Create solver
solver = TDSESolver(x, psi0, V, dt=0.01)

# Evolve in time
times, psi_t = solver.evolve(t_max=20.0)

# Plot results
from tdse_solver.visualization import plot_evolution
plot_evolution(x, times, psi_t, V)
```

## Creating Initial Wave Packets

### Gaussian Wave Packet

The most common initial state:

```python
from tdse_solver import GaussianWavePacket

psi = GaussianWavePacket(
    x,
    x0=0.0,      # Center position
    k0=2.0,      # Wave number (momentum = ℏk)
    sigma=2.0    # Width parameter
)
```

### Plane Wave

```python
from tdse_solver.wavefunction import PlaneWave

psi = PlaneWave(x, k=2.0)
```

### Harmonic Oscillator Ground State

```python
from tdse_solver.wavefunction import GroundStateHO

psi = GroundStateHO(x, omega=1.0)
```

### Custom Wave Function

You can also create your own:

```python
import numpy as np
from tdse_solver.wavefunction import normalize_wavefunction

# Create custom wave function
psi = np.exp(-x**2) * np.exp(1j * x)

# Normalize it
psi = normalize_wavefunction(psi, x)
```

## Defining Potentials

### Built-in Potentials

```python
from tdse_solver.potentials import *

# Harmonic oscillator
V = HarmonicOscillator(x, omega=1.0)

# Square barrier
V = SquareBarrier(x, height=2.0, width=5.0, center=0.0)

# Square well
V = SquareWell(x, depth=-2.0, width=5.0)

# Gaussian barrier
V = GaussianBarrier(x, height=2.0, width=2.0)

# Double slit
V = DoubleSlit(x, slit_separation=8.0, slit_width=1.5)

# Free particle
V = ZeroPotential(x)
```

### Custom Potential

Define your own potential as a function or array:

```python
# As a function
def my_potential(x):
    return 0.5 * x**2 + 0.1 * x**4

V = my_potential(x)

# Or directly as an array
V = np.where(np.abs(x) < 5, -1.0, 0.0)
```

## Solver Configuration

### Basic Parameters

```python
solver = TDSESolver(
    x,           # Spatial grid
    psi0,        # Initial wave function
    V,           # Potential (array or callable)
    dt=0.01,     # Time step
    hbar=1.0,    # Reduced Planck constant (atomic units)
    m=1.0        # Particle mass (atomic units)
)
```

### Choosing Time Step

A good rule of thumb:
- `dt ≈ 0.01 * dx²` for typical problems
- Smaller `dt` for high potentials or high momentum
- Check energy conservation to verify `dt` is appropriate

## Time Evolution

### Evolve to a Maximum Time

```python
times, psi_t = solver.evolve(
    t_max=20.0,      # Maximum time
    store_every=5    # Store every 5 time steps
)
```

### Single Time Step

For more control, use individual steps:

```python
for i in range(n_steps):
    solver.step()
    
    # Access current state
    psi_current = solver.psi
    time = solver.time
```

## Calculating Observables

### Position and Momentum

```python
# Position expectation value
x_exp = solver.get_expectation_x()

# Position squared
x2_exp = solver.get_expectation_x2()

# Standard deviation
sigma_x = np.sqrt(x2_exp - x_exp**2)

# Momentum expectation value
p_exp = solver.get_expectation_p()
```

### Energy

```python
# Total energy (kinetic + potential)
E = solver.get_energy()
```

### Probability Density

```python
# Get probability density |ψ|²
prob = solver.get_probability_density()

# Probability in a region
prob_region = np.sum(prob[x > 0]) * dx
```

### Normalization

```python
# Check normalization (should be 1.0)
norm = solver.get_norm()
```

## Visualization

### Static Plots

```python
from tdse_solver.visualization import plot_wavefunction, plot_evolution

# Plot current wave function
fig, axes = plot_wavefunction(x, solver.psi, V)

# Plot snapshots of evolution
fig, axes = plot_evolution(x, times, psi_t, V)
```

### Animations

```python
from tdse_solver.visualization import animate_wavefunction

# Create animation
anim = animate_wavefunction(
    x, times, psi_t, V,
    filename='animation.gif',  # Save as GIF
    fps=30,                     # Frames per second
    title='My Simulation'
)
```

### Space-Time Plots

```python
from tdse_solver.visualization import plot_spacetime_density

# Create space-time density plot
fig, ax = plot_spacetime_density(x, times, psi_t)
```

### Momentum Space

```python
from tdse_solver.wavefunction import get_momentum_distribution
from tdse_solver.visualization import plot_momentum_space

# Transform to momentum space
k, phi = get_momentum_distribution(solver.psi, x)

# Plot
fig, ax = plot_momentum_space(k, phi)
```

## Best Practices

### Grid Setup

1. **Large enough domain**: Wave packet should not reach boundaries
   ```python
   x_extent = 10 * sigma  # At least 10 standard deviations
   x = np.linspace(-x_extent, x_extent, N)
   ```

2. **Sufficient resolution**: Resolve the shortest wavelength
   ```python
   k_max = k0 + 3/sigma  # Maximum wave number
   dx_required = 2*np.pi / (2*k_max)  # Nyquist criterion
   N = int(2 * x_extent / dx_required)
   ```

### Time Step Selection

Check energy conservation:

```python
E0 = solver.get_energy()
solver.evolve(t_max=10.0)
E_final = solver.get_energy()

relative_error = abs(E_final - E0) / abs(E0)
if relative_error > 0.01:
    print("Warning: Energy not conserved, use smaller dt")
```

### Memory Considerations

For long simulations, don't store every time step:

```python
# Store every 10th step to save memory
times, psi_t = solver.evolve(t_max=100.0, store_every=10)
```

## Advanced Usage

### Custom Evolution Loop

```python
# Initialize storage
n_steps = 1000
store_interval = 10
n_store = n_steps // store_interval

psi_t = np.zeros((n_store, len(x)), dtype=complex)
energies = np.zeros(n_store)

# Evolution loop
for i in range(n_steps):
    solver.step()
    
    if i % store_interval == 0:
        idx = i // store_interval
        psi_t[idx] = solver.psi.copy()
        energies[idx] = solver.get_energy()
        
        # Custom analysis here
        print(f"Time: {solver.time:.2f}, E: {energies[idx]:.4f}")
```

### Resetting the Solver

```python
# Reset with new initial condition
psi_new = GaussianWavePacket(x, x0=5, k0=1, sigma=2)
solver.reset(psi_new)
```

## Troubleshooting

### Wave Packet Reaches Boundary

**Problem**: Reflections from boundaries corrupt the solution.

**Solution**:
- Increase domain size
- Use absorbing boundary conditions (not yet implemented)
- End simulation before wave packet reaches boundary

### Energy Not Conserved

**Problem**: Energy changes significantly during evolution.

**Solution**:
- Decrease time step `dt`
- Increase grid resolution (more points)
- Check that potential doesn't have discontinuities

### Numerical Instabilities

**Problem**: Solution becomes unstable or explodes.

**Solution**:
- Reduce time step
- Check initial conditions are properly normalized
- Ensure potential is reasonable (not too large)

## Examples

See the `examples/` directory for complete working examples:
- `quantum_tunneling.py` - Barrier tunneling
- `harmonic_oscillator.py` - Coherent state oscillations
- `free_particle.py` - Wave packet dispersion
- `potential_well.py` - Bound state dynamics
- `double_slit.py` - Interference patterns
