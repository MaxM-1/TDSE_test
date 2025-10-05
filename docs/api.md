# API Reference

## Core Module: `tdse_solver.solver`

### `TDSESolver`

Main class for solving the time-dependent Schrödinger equation.

#### Constructor

```python
TDSESolver(x, psi_initial, V, dt=0.01, hbar=1.0, m=1.0)
```

**Parameters:**
- `x` (ndarray): Spatial grid points
- `psi_initial` (ndarray): Initial complex wave function
- `V` (ndarray or callable): Potential energy (array or function of x)
- `dt` (float): Time step for evolution (default: 0.01)
- `hbar` (float): Reduced Planck constant in atomic units (default: 1.0)
- `m` (float): Particle mass in atomic units (default: 1.0)

**Attributes:**
- `x` (ndarray): Spatial grid
- `psi` (ndarray): Current wave function
- `V` (ndarray): Potential energy array
- `time` (float): Current time
- `dt` (float): Time step
- `dx` (float): Spatial grid spacing
- `N` (int): Number of grid points
- `k` (ndarray): Momentum space grid
- `hbar` (float): Reduced Planck constant
- `m` (float): Particle mass

#### Methods

##### `step()`
Perform one time step using the split-operator method.

**Returns:** None (modifies `self.psi` and `self.time` in place)

##### `evolve(t_max, store_every=1)`
Evolve the wave function up to a maximum time.

**Parameters:**
- `t_max` (float): Maximum time to evolve
- `store_every` (int): Store wave function every N steps (default: 1)

**Returns:**
- `times` (ndarray): Array of time points
- `psi_t` (ndarray): Wave function at each time point, shape: (n_times, N)

##### `get_probability_density()`
Get the probability density |ψ|².

**Returns:** ndarray of probability density

##### `get_expectation_x()`
Calculate expectation value of position ⟨x⟩.

**Returns:** float

##### `get_expectation_x2()`
Calculate expectation value of position squared ⟨x²⟩.

**Returns:** float

##### `get_expectation_p()`
Calculate expectation value of momentum ⟨p⟩.

**Returns:** float

##### `get_energy()`
Calculate total energy expectation value ⟨H⟩ (kinetic + potential).

**Returns:** float

##### `get_norm()`
Calculate the norm of the wave function (should be 1 if normalized).

**Returns:** float

##### `reset(psi_initial)`
Reset the solver with a new initial wave function.

**Parameters:**
- `psi_initial` (ndarray): New initial wave function

**Returns:** None

---

## Wave Function Module: `tdse_solver.wavefunction`

### Wave Packet Functions

#### `GaussianWavePacket(x, x0=0.0, k0=0.0, sigma=1.0, normalize=True)`
Create a Gaussian wave packet.

**Parameters:**
- `x` (ndarray): Spatial grid
- `x0` (float): Initial center position (default: 0.0)
- `k0` (float): Initial momentum/wave number (default: 0.0)
- `sigma` (float): Width parameter (standard deviation) (default: 1.0)
- `normalize` (bool): Whether to normalize (default: True)

**Returns:** ndarray (complex wave function)

#### `PlaneWave(x, k=1.0, amplitude=1.0, normalize=True)`
Create a plane wave state.

**Parameters:**
- `x` (ndarray): Spatial grid
- `k` (float): Wave number (default: 1.0)
- `amplitude` (float): Amplitude (default: 1.0)
- `normalize` (bool): Whether to normalize (default: True)

**Returns:** ndarray (complex wave function)

#### `GroundStateHO(x, omega=1.0, m=1.0, hbar=1.0, center=0.0)`
Ground state of the quantum harmonic oscillator.

**Parameters:**
- `x` (ndarray): Spatial grid
- `omega` (float): Angular frequency (default: 1.0)
- `m` (float): Mass (default: 1.0)
- `hbar` (float): Reduced Planck constant (default: 1.0)
- `center` (float): Center position (default: 0.0)

**Returns:** ndarray (real wave function)

#### `ExcitedStateHO(x, n, omega=1.0, m=1.0, hbar=1.0, center=0.0)`
Excited state of the quantum harmonic oscillator.

**Parameters:**
- `x` (ndarray): Spatial grid
- `n` (int): Quantum number (0, 1, 2, ...)
- `omega` (float): Angular frequency (default: 1.0)
- `m` (float): Mass (default: 1.0)
- `hbar` (float): Reduced Planck constant (default: 1.0)
- `center` (float): Center position (default: 0.0)

**Returns:** ndarray (real wave function)

#### `Superposition(x, psi1, psi2, c1=1.0, c2=1.0, normalize=True)`
Create a superposition of two wave functions.

**Parameters:**
- `x` (ndarray): Spatial grid
- `psi1` (ndarray): First wave function
- `psi2` (ndarray): Second wave function
- `c1` (complex): Coefficient for first state (default: 1.0)
- `c2` (complex): Coefficient for second state (default: 1.0)
- `normalize` (bool): Whether to normalize (default: True)

**Returns:** ndarray (complex wave function)

### Utility Functions

#### `normalize_wavefunction(psi, x)`
Normalize a wave function so that ∫|ψ|²dx = 1.

**Parameters:**
- `psi` (ndarray): Wave function to normalize
- `x` (ndarray): Spatial grid

**Returns:** ndarray (normalized wave function)

#### `calculate_overlap(psi1, psi2, x)`
Calculate the overlap between two wave functions: ⟨ψ₁|ψ₂⟩.

**Parameters:**
- `psi1` (ndarray): First wave function
- `psi2` (ndarray): Second wave function
- `x` (ndarray): Spatial grid

**Returns:** complex (overlap integral)

#### `get_momentum_distribution(psi, x)`
Calculate the momentum space wave function via Fourier transform.

**Parameters:**
- `psi` (ndarray): Position space wave function
- `x` (ndarray): Spatial grid

**Returns:**
- `k` (ndarray): Momentum space grid
- `phi` (ndarray): Momentum space wave function

#### `spreading_width(psi, x)`
Calculate the width (standard deviation) of a wave packet: Δx = √(⟨x²⟩ - ⟨x⟩²).

**Parameters:**
- `psi` (ndarray): Wave function
- `x` (ndarray): Spatial grid

**Returns:** float (width of the wave packet)

---

## Potentials Module: `tdse_solver.potentials`

### Potential Functions

#### `HarmonicOscillator(x, omega=1.0, m=1.0, hbar=1.0, center=0.0)`
Harmonic oscillator potential: V(x) = ½mω²(x - x₀)².

#### `SquareBarrier(x, height=1.0, width=2.0, center=0.0)`
Square potential barrier.

#### `SquareWell(x, depth=-1.0, width=2.0, center=0.0)`
Square potential well.

#### `DoubleSlit(x, slit_separation=10.0, slit_width=1.0, barrier_height=10.0, center=0.0)`
Double slit potential.

#### `GaussianBarrier(x, height=1.0, width=2.0, center=0.0)`
Gaussian-shaped potential barrier.

#### `GaussianWell(x, depth=-1.0, width=2.0, center=0.0)`
Gaussian-shaped potential well.

#### `DoubleWell(x, separation=5.0, depth=-1.0, width=1.0)`
Double well potential (two Gaussian wells).

#### `PeriodicPotential(x, amplitude=1.0, period=2π, phase=0.0)`
Periodic potential: V(x) = A cos(2πx/λ + φ).

#### `StepPotential(x, height=1.0, position=0.0)`
Step potential (sudden change in potential).

#### `LinearPotential(x, slope=0.1, offset=0.0)`
Linear potential: V(x) = ax + b.

#### `CoulombPotential(x, charge=-1.0, epsilon=0.1, center=0.0)`
Coulomb-like potential with regularization.

#### `ZeroPotential(x)`
Zero potential (free particle).

#### `CustomPotential(x, func)`
Create custom potential from user-defined function.

---

## Visualization Module: `tdse_solver.visualization`

### Plotting Functions

#### `plot_wavefunction(x, psi, V=None, title="Wave Function", figsize=(10, 6))`
Plot the wave function and probability density.

**Parameters:**
- `x` (ndarray): Spatial grid
- `psi` (ndarray): Complex wave function
- `V` (ndarray, optional): Potential energy to plot
- `title` (str): Plot title (default: "Wave Function")
- `figsize` (tuple): Figure size (default: (10, 6))

**Returns:** fig, axes (Matplotlib figure and axes objects)

#### `plot_evolution(x, times, psi_t, V=None, figsize=(12, 8))`
Plot snapshots of wave function evolution.

**Parameters:**
- `x` (ndarray): Spatial grid
- `times` (ndarray): Time points
- `psi_t` (ndarray): Wave function at each time, shape: (n_times, N)
- `V` (ndarray, optional): Potential energy
- `figsize` (tuple): Figure size (default: (12, 8))

**Returns:** fig, axes

#### `animate_wavefunction(x, times, psi_t, V=None, filename=None, fps=30, title="Wave Function Evolution", figsize=(10, 6))`
Create an animation of wave function evolution.

**Parameters:**
- `x` (ndarray): Spatial grid
- `times` (ndarray): Time points
- `psi_t` (ndarray): Wave function at each time
- `V` (ndarray, optional): Potential energy
- `filename` (str, optional): Filename to save animation (e.g., 'anim.gif')
- `fps` (int): Frames per second (default: 30)
- `title` (str): Animation title (default: "Wave Function Evolution")
- `figsize` (tuple): Figure size (default: (10, 6))

**Returns:** animation (Matplotlib animation object)

#### `plot_spacetime_density(x, times, psi_t, figsize=(10, 6))`
Create a space-time plot of probability density.

**Parameters:**
- `x` (ndarray): Spatial grid
- `times` (ndarray): Time points
- `psi_t` (ndarray): Wave function at each time
- `figsize` (tuple): Figure size (default: (10, 6))

**Returns:** fig, ax

#### `plot_probability_current(x, times, psi_t, hbar=1.0, m=1.0, figsize=(10, 6))`
Plot the probability current density j(x,t) = (ℏ/m) Im(ψ* ∂ψ/∂x).

**Parameters:**
- `x` (ndarray): Spatial grid
- `times` (ndarray): Time points
- `psi_t` (ndarray): Wave function at each time
- `hbar` (float): Reduced Planck constant (default: 1.0)
- `m` (float): Mass (default: 1.0)
- `figsize` (tuple): Figure size (default: (10, 6))

**Returns:** fig, ax

#### `plot_momentum_space(k, phi, title="Momentum Space Wave Function", figsize=(10, 6))`
Plot the momentum space wave function.

**Parameters:**
- `k` (ndarray): Momentum grid
- `phi` (ndarray): Momentum space wave function
- `title` (str): Plot title (default: "Momentum Space Wave Function")
- `figsize` (tuple): Figure size (default: (10, 6))

**Returns:** fig, ax
