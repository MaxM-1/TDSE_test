# Time-Dependent Schrödinger Equation Solver

A comprehensive Python package for numerically solving the time-dependent Schrödinger equation (TDSE) in one dimension using spectral and finite difference methods.

## Features

- **Multiple Numerical Methods**: Split-operator method with FFT for high accuracy
- **Flexible Potentials**: Built-in potentials (harmonic oscillator, square barrier, quantum well, Gaussian, etc.)
- **Wave Packet Tools**: Gaussian wave packets with customizable parameters
- **Visualization**: Real-time animation and static plotting capabilities
- **Physical Accuracy**: Conservation of probability and energy verification
- **Extensible Design**: Easy to add custom potentials and initial conditions

## Installation

### From Source

```bash
git clone https://github.com/MaxM-1/TDSE_test.git
cd tdse-solver
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## Quick Start

```python
from tdse_solver import TDSESolver, GaussianWavePacket
from tdse_solver.potentials import SquareBarrier
import numpy as np

# Set up spatial grid
x = np.linspace(-50, 50, 1024)
dx = x[1] - x[0]

# Create initial wave packet
psi0 = GaussianWavePacket(x, x0=-20, k0=2.0, sigma=2.0)

# Define potential (square barrier)
V = SquareBarrier(x, height=1.5, width=5.0, center=0.0)

# Create solver
solver = TDSESolver(x, psi0, V, dt=0.01)

# Evolve in time
times, psi_t = solver.evolve(t_max=20.0)

# Visualize
from tdse_solver.visualization import animate_wavefunction
animate_wavefunction(x, times, psi_t, V, filename='tunneling.gif')
```

## Examples

The `examples/` directory contains several demonstration scripts:

- `quantum_tunneling.py` - Barrier tunneling and reflection
- `harmonic_oscillator.py` - Oscillations in a parabolic potential
- `double_slit.py` - Wave packet interference
- `free_particle.py` - Dispersing Gaussian wave packet
- `potential_well.py` - Bound state dynamics

Run an example:
```bash
python examples/quantum_tunneling.py
```

## Theory

The time-dependent Schrödinger equation in one dimension is:

```
iℏ ∂ψ/∂t = [-ℏ²/2m ∂²/∂x² + V(x)] ψ
```

This package uses the split-operator method, which separates the Hamiltonian into kinetic and potential parts:

```
ψ(t + Δt) ≈ exp(-iV̂Δt/2ℏ) exp(-iT̂Δt/ℏ) exp(-iV̂Δt/2ℏ) ψ(t)
```

The kinetic operator is applied in momentum space using FFT for high accuracy.

## Documentation

See the `docs/` folder for detailed documentation:
- [Mathematical Background](docs/theory.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api.md)

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tdse_solver,
  author = {Max S.J. Miller},
  title = {Time-Dependent Schrödinger Equation Solver},
  year = {2025},
  url = {https://github.com/MaxM-1/TDSE_test}
}
```

## References

1. Press, W. H., et al. "Numerical Recipes" (2007)
2. Tannor, D. J. "Introduction to Quantum Mechanics: A Time-Dependent Perspective" (2007)
3. Feit, M. D., et al. "Solution of the Schrödinger equation by a spectral method" J. Comp. Phys. (1982)
