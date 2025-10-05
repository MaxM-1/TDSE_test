"""
Harmonic Oscillator Example

Demonstrates coherent oscillations of a wave packet in a parabolic potential.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tdse_solver import TDSESolver, GaussianWavePacket
from tdse_solver.potentials import HarmonicOscillator
from tdse_solver.wavefunction import GroundStateHO
from tdse_solver.visualization import animate_wavefunction, plot_evolution


def main():
    print("=" * 60)
    print("Quantum Harmonic Oscillator Simulation")
    print("=" * 60)
    
    # Set up spatial grid
    x = np.linspace(-20, 20, 512)
    dx = x[1] - x[0]
    
    # Harmonic oscillator parameters
    omega = 0.5  # Angular frequency
    m = 1.0      # Mass
    hbar = 1.0   # Reduced Planck constant
    
    print(f"\nHarmonic Oscillator Parameters:")
    print(f"  Angular frequency: ω = {omega}")
    print(f"  Classical period: T = 2π/ω = {2*np.pi/omega:.3f}")
    print(f"  Ground state energy: E0 = ℏω/2 = {hbar*omega/2:.3f}")
    
    # Create potential
    V = HarmonicOscillator(x, omega=omega, m=m, hbar=hbar, center=0.0)
    
    # Option 1: Displaced Gaussian wave packet (coherent state)
    x0 = 5.0   # Displaced from center
    k0 = 0.0   # Initially at rest
    sigma = 1.0 / np.sqrt(m * omega / hbar)  # Minimum uncertainty
    
    print(f"\nInitial Wave Packet (Coherent State):")
    print(f"  Displacement: x0 = {x0}")
    print(f"  Initial momentum: k0 = {k0}")
    print(f"  Width: σ = {sigma:.3f}")
    
    psi0 = GaussianWavePacket(x, x0=x0, k0=k0, sigma=sigma)
    
    # Create solver
    dt = 0.05
    solver = TDSESolver(x, psi0, V, dt=dt, hbar=hbar, m=m)
    
    print(f"\nSimulation Parameters:")
    print(f"  Time step: dt = {dt}")
    print(f"  Initial energy: E = {solver.get_energy():.4f}")
    
    # Evolve for multiple periods
    n_periods = 3
    t_max = n_periods * 2 * np.pi / omega
    store_every = 5
    
    print(f"\nEvolving for {n_periods} classical periods (t = {t_max:.2f})...")
    times, psi_t = solver.evolve(t_max=t_max, store_every=store_every)
    
    # Calculate expectation values over time
    x_exp = np.zeros(len(times))
    energy = np.zeros(len(times))
    
    print("Calculating expectation values...")
    for i, psi in enumerate(psi_t):
        solver.psi = psi
        solver.time = times[i]
        x_exp[i] = solver.get_expectation_x()
        energy[i] = solver.get_energy()
    
    # Theoretical solution for coherent state
    x_theory = x0 * np.cos(omega * times)
    
    print(f"\nFinal Properties:")
    print(f"  Final energy: E = {energy[-1]:.4f}")
    print(f"  Energy change: ΔE/E0 = {abs(energy[-1] - energy[0]) / energy[0] * 100:.4f}%")
    print(f"  Final norm: ||ψ|| = {solver.get_norm():.6f}")
    
    # Create visualizations
    print("\nCreating plots...")
    
    # Plot position expectation value
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(times, x_exp, 'b-', linewidth=2, label='Simulation')
    ax1.plot(times, x_theory, 'r--', linewidth=2, alpha=0.7, label='Theory')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('<x>')
    ax1.set_title('Position Expectation Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, energy, 'g-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy')
    ax2.set_title('Total Energy (Should be Conserved)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('harmonic_oscillator_observables.png', dpi=150, bbox_inches='tight')
    print("  Saved: harmonic_oscillator_observables.png")
    
    # Plot snapshots
    fig2, _ = plot_evolution(x, times, psi_t, V)
    fig2.suptitle('Harmonic Oscillator: Wave Packet Evolution')
    plt.savefig('harmonic_oscillator_snapshots.png', dpi=150, bbox_inches='tight')
    print("  Saved: harmonic_oscillator_snapshots.png")
    
    # Create animation
    print("  Creating animation (this may take a moment)...")
    anim = animate_wavefunction(x, times, psi_t, V,
                                filename='harmonic_oscillator_animation.gif',
                                fps=30,
                                title='Quantum Harmonic Oscillator')
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
