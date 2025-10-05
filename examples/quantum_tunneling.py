"""
Quantum Tunneling Example

Demonstrates a Gaussian wave packet encountering a potential barrier,
showing both transmission and reflection.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tdse_solver import TDSESolver, GaussianWavePacket
from tdse_solver.potentials import SquareBarrier
from tdse_solver.visualization import animate_wavefunction, plot_evolution


def main():
    print("=" * 60)
    print("Quantum Tunneling Simulation")
    print("=" * 60)
    
    # Set up spatial grid
    x_min, x_max = -50, 50
    N = 1024
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    
    # Create initial wave packet (moving to the right)
    x0 = -20.0  # Start position
    k0 = 2.0    # Wave number (momentum)
    sigma = 2.0  # Width
    
    print(f"\nInitial Wave Packet:")
    print(f"  Position: x0 = {x0}")
    print(f"  Momentum: k0 = {k0}")
    print(f"  Width: σ = {sigma}")
    print(f"  Energy: E ≈ {0.5 * k0**2:.3f}")
    
    psi0 = GaussianWavePacket(x, x0=x0, k0=k0, sigma=sigma)
    
    # Define potential barrier
    barrier_height = 1.5
    barrier_width = 5.0
    barrier_center = 0.0
    
    print(f"\nPotential Barrier:")
    print(f"  Height: V0 = {barrier_height}")
    print(f"  Width: w = {barrier_width}")
    print(f"  Center: x = {barrier_center}")
    
    V = SquareBarrier(x, height=barrier_height, width=barrier_width, center=barrier_center)
    
    # Create solver
    dt = 0.01
    solver = TDSESolver(x, psi0, V, dt=dt)
    
    print(f"\nSimulation Parameters:")
    print(f"  Grid points: N = {N}")
    print(f"  Grid spacing: dx = {dx:.4f}")
    print(f"  Time step: dt = {dt}")
    
    # Calculate initial properties
    E0 = solver.get_energy()
    print(f"  Initial energy: E = {E0:.4f}")
    print(f"  Initial norm: ||ψ|| = {solver.get_norm():.6f}")
    
    # Evolve in time
    t_max = 20.0
    store_every = 5
    
    print(f"\nEvolving system to t = {t_max}...")
    times, psi_t = solver.evolve(t_max=t_max, store_every=store_every)
    
    # Check conservation
    print(f"\nFinal Properties:")
    print(f"  Final energy: E = {solver.get_energy():.4f}")
    print(f"  Final norm: ||ψ|| = {solver.get_norm():.6f}")
    print(f"  Energy change: ΔE/E0 = {abs(solver.get_energy() - E0) / E0 * 100:.4f}%")
    
    # Calculate transmission and reflection coefficients
    # (probability to the right and left of the barrier)
    final_prob = np.abs(solver.psi)**2
    barrier_end = barrier_center + barrier_width / 2
    
    transmitted = np.sum(final_prob[x > barrier_end]) * dx
    reflected = np.sum(final_prob[x < -10]) * dx  # Back to starting region
    
    print(f"\nTransmission/Reflection:")
    print(f"  Transmitted: {transmitted * 100:.2f}%")
    print(f"  Reflected: {reflected * 100:.2f}%")
    
    # Create visualizations
    print("\nCreating plots...")
    
    # Plot snapshots
    fig1, _ = plot_evolution(x, times, psi_t, V)
    fig1.suptitle('Quantum Tunneling: Wave Packet Evolution')
    plt.savefig('tunneling_snapshots.png', dpi=150, bbox_inches='tight')
    print("  Saved: tunneling_snapshots.png")
    
    # Create animation
    print("  Creating animation (this may take a moment)...")
    anim = animate_wavefunction(x, times, psi_t, V, 
                                filename='tunneling_animation.gif',
                                fps=30,
                                title='Quantum Tunneling')
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
