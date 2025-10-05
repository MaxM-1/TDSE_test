# Mathematical Background

## The Time-Dependent Schrödinger Equation

The time-dependent Schrödinger equation (TDSE) describes how quantum states evolve in time:

```
iℏ ∂ψ/∂t = Ĥψ
```

where:
- `ψ(x,t)` is the wave function (complex-valued)
- `ℏ` is the reduced Planck constant
- `Ĥ` is the Hamiltonian operator
- `i` is the imaginary unit

## One-Dimensional Case

In one dimension with a time-independent potential `V(x)`, the Hamiltonian is:

```
Ĥ = T̂ + V̂ = -ℏ²/(2m) ∂²/∂x² + V(x)
```

The full equation becomes:

```
iℏ ∂ψ/∂t = [-ℏ²/(2m) ∂²/∂x² + V(x)] ψ
```

## Physical Interpretation

### Wave Function
The wave function `ψ(x,t)` is a complex-valued function that contains all information about the quantum state.

### Probability Density
The probability of finding the particle at position `x` at time `t` is:

```
P(x,t) = |ψ(x,t)|²
```

### Normalization
The wave function must be normalized:

```
∫_{-∞}^{∞} |ψ(x,t)|² dx = 1
```

### Expectation Values
The expectation value of an observable `A` is:

```
<A> = ∫_{-∞}^{∞} ψ*(x,t) Â ψ(x,t) dx
```

For position and momentum:

```
<x> = ∫_{-∞}^{∞} x|ψ|² dx

<p> = ∫_{-∞}^{∞} ψ* (-iℏ ∂/∂x) ψ dx
```

## Conservation Laws

### Probability Conservation
The total probability is conserved:

```
d/dt ∫|ψ|² dx = 0
```

### Energy Conservation
For time-independent Hamiltonians, energy is conserved:

```
d/dt <H> = 0
```

## The Split-Operator Method

The split-operator method is a highly accurate technique for solving the TDSE numerically.

### Time Evolution Operator
The formal solution to the TDSE is:

```
ψ(t + Δt) = exp(-iĤΔt/ℏ) ψ(t)
```

### Splitting the Hamiltonian
We split the Hamiltonian into kinetic and potential parts:

```
Ĥ = T̂ + V̂
```

### Trotter Formula
Using the Trotter formula (to second order):

```
exp(-iĤΔt/ℏ) ≈ exp(-iV̂Δt/2ℏ) exp(-iT̂Δt/ℏ) exp(-iV̂Δt/2ℏ) + O(Δt³)
```

### Applying the Operators

1. **Potential operator** (position space):
   ```
   ψ → exp(-iV(x)Δt/2ℏ) ψ
   ```
   This is a simple multiplication.

2. **Kinetic operator** (momentum space):
   ```
   T̂ = -ℏ²/(2m) ∂²/∂x² → (ℏk)²/(2m) in momentum space
   ```
   
   We use FFT to transform to momentum space:
   ```
   φ(k) = FFT[ψ(x)]
   φ → exp(-i(ℏk)²Δt/(2mℏ)) φ
   ψ = IFFT[φ(k)]
   ```

3. **Second potential step**:
   ```
   ψ → exp(-iV(x)Δt/2ℏ) ψ
   ```

### Advantages
- **High accuracy**: O(Δt³) error per step
- **Unconditionally stable**: Works for any Δt
- **Unitary**: Preserves normalization
- **Efficient**: Uses FFT for derivatives

## Numerical Considerations

### Grid Spacing
The spatial grid spacing `dx` must resolve the wave function:

```
dx ≪ λ = 2π/k
```

where `k` is the typical wave number.

### Time Step
The time step should satisfy:

```
Δt ≪ ℏ/E
```

where `E` is the characteristic energy scale.

### Boundary Conditions
The wave function should decay to zero at the boundaries to minimize reflections.

## Common Quantum Systems

### Free Particle (V = 0)
- Wave packet spreads: `σ(t) = σ₀√(1 + (t/τ)²)` where `τ = 2mσ₀²/ℏ`
- Center moves with constant velocity: `<x> = x₀ + (ℏk₀/m)t`

### Harmonic Oscillator (V = ½mω²x²)
- Coherent states oscillate with period `T = 2π/ω`
- Energy levels: `Eₙ = ℏω(n + 1/2)`

### Potential Barrier
- Quantum tunneling: particles can pass through classically forbidden regions
- Transmission coefficient depends on barrier height and width

### Double Slit
- Wave packet diffracts through slits
- Interference pattern emerges beyond the slits

## References

1. Tannor, D. J. "Introduction to Quantum Mechanics: A Time-Dependent Perspective" (2007)
2. Feit, M. D., Fleck, J. A., & Steiger, A. "Solution of the Schrödinger equation by a spectral method" J. Comp. Phys. 47, 412 (1982)
3. Press, W. H., et al. "Numerical Recipes" Cambridge University Press (2007)
4. Sakurai, J. J. "Modern Quantum Mechanics" (2017)
