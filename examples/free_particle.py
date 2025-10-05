"""
Free Particle Example

Demonstrates the dispersion (spreading) of a free Gaussian wave packet.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tdse_solver import TDSESolver, GaussianWavePacket
from tdse_solver.potentials import ZeroPotential
from tdse_solver.wavefunction import spreading_width
from tdse_solver.visualization import animate_wavefunction, plot_spacetime_density


def main():
    print("=" * 60)
    print("Free Particle Dispersion Simulation")
    print("=" * 60)
    
    # Set up spatial grid
    x = np.linspace(-50, 50, 1024)
    dx = x[1] - x[0]
    
    # Create initial wave packet
    x0 = 0.0    # Center position
    k0 = 1.0    # Initial momentum
    sigma0 = 2.0  # Initial width
    
    print(f"\nInitial Wave Packet:")
    print(f"  Center: x0 = {x0}")
    print(f"  Momentum: k0 = {k0}")
    print(f"  Initial width: σ0 = {sigma0}")
    print(f"  Initial kinetic energy: E = k²/2 = {0.5 * k0**2:.3f}")
    
    psi0 = GaussianWavePacket(x, x0=x0, k0=k0, sigma=sigma0)
    
    # Zero potential (free particle)
    V = ZeroPotential(x)
    
    # Create solver
    dt = 0.02
    hbar = 1.0
    m = 1.0
    solver = TDSESolver(x, psi0, V, dt=dt, hbar=hbar, m=m)
    
    print(f"\nSimulation Parameters:")
    print(f"  Grid points: N = {len(x)}")
    print(f"  Time step: dt = {dt}")
    
    # Theoretical spreading
    # For a free Gaussian: σ(t) = σ0 * sqrt(1 + (ℏt/(2mσ0²))²)
    print(f"\nTheoretical spreading rate:")
    print(f"  τ = 2mσ0²/ℏ = {2 * m * sigma0**2 / hbar:.3f}")
    
    # Evolve in time
    t_max = 30.0
    store_every = 5
    
    print(f"\nEvolving system to t = {t_max}...")
    times, psi_t = solver.evolve(t_max=t_max, store_every=store_every)
    
    # Calculate width over time
    widths = np.zeros(len(times))
    positions = np.zeros(len(times))
    
    for i, psi in enumerate(psi_t):
        widths[i] = spreading_width(psi, x)
        prob = np.abs(psi)**2
        positions[i] = np.sum(x * prob) * dx
    
    # Theoretical width
    tau = 2 * m * sigma0**2 / hbar
    widths_theory = sigma0 * np.sqrt(1 + (times / tau)**2)
    
    # Theoretical position
    positions_theory = x0 + (hbar * k0 / m) * times
    
    print(f"\nFinal Properties:")
    print(f"  Final width: σ(t) = {widths[-1]:.3f}")
    print(f"  Width increase: σ(t)/σ0 = {widths[-1]/sigma0:.3f}x")
    print(f"  Final position: <x> = {positions[-1]:.3f}")
    print(f"  Expected position: {positions_theory[-1]:.3f}")
    
    # Create visualizations
    print("\nCreating plots...")
    
    # Plot width and position evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(times, widths, 'b-', linewidth=2, label='Simulation')
    ax1.plot(times, widths_theory, 'r--', linewidth=2, alpha=0.7, label='Theory')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Width σ(t)')
    ax1.set_title('Wave Packet Spreading')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, positions, 'b-', linewidth=2, label='Simulation')
    ax2.plot(times, positions_theory, 'r--', linewidth=2, alpha=0.7, label='Theory')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position <x>')
    ax2.set_title('Center of Mass Motion')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('free_particle_observables.png', dpi=150, bbox_inches='tight')
    print("  Saved: free_particle_observables.png")
    
    # Space-time density plot
    fig2, _ = plot_spacetime_density(x, times, psi_t)
    fig2.suptitle('Free Particle: Space-Time Evolution')
    plt.savefig('free_particle_spacetime.png', dpi=150, bbox_inches='tight')
    print("  Saved: free_particle_spacetime.png")
    
    # Create animation
    print("  Creating animation (this may take a moment)...")
    anim = animate_wavefunction(x, times, psi_t, V,
                                filename='free_particle_animation.gif',
                                fps=30,
                                title='Free Particle Dispersion')
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
