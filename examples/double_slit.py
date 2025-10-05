"""
Double Slit Example

Demonstrates interference patterns from a wave packet passing through two slits.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tdse_solver import TDSESolver, GaussianWavePacket
from tdse_solver.potentials import DoubleSlit
from tdse_solver.visualization import animate_wavefunction, plot_spacetime_density


def main():
    print("=" * 60)
    print("Double Slit Interference Simulation")
    print("=" * 60)
    
    # Set up spatial grid
    x = np.linspace(-40, 40, 1024)
    dx = x[1] - x[0]
    
    # Double slit parameters
    slit_separation = 8.0
    slit_width = 1.5
    barrier_height = 50.0  # High barrier to approximate hard walls
    slit_position = 0.0
    
    print(f"\nDouble Slit Parameters:")
    print(f"  Slit separation: d = {slit_separation}")
    print(f"  Slit width: w = {slit_width}")
    print(f"  Barrier height: V = {barrier_height}")
    
    V = DoubleSlit(x, slit_separation=slit_separation, slit_width=slit_width,
                   barrier_height=barrier_height, center=slit_position)
    
    # Create initial wave packet (approaching from left)
    x0 = -20.0
    k0 = 3.0  # Need sufficient momentum to pass through
    sigma = 5.0  # Wide enough to cover both slits
    
    print(f"\nInitial Wave Packet:")
    print(f"  Position: x0 = {x0}")
    print(f"  Momentum: k0 = {k0}")
    print(f"  Width: σ = {sigma}")
    print(f"  Energy: E ≈ {0.5 * k0**2:.3f}")
    
    psi0 = GaussianWavePacket(x, x0=x0, k0=k0, sigma=sigma)
    
    # Create solver
    dt = 0.005  # Smaller time step for high barrier
    solver = TDSESolver(x, psi0, V, dt=dt)
    
    print(f"\nSimulation Parameters:")
    print(f"  Time step: dt = {dt}")
    print(f"  Initial energy: E = {solver.get_energy():.4f}")
    
    # Evolve in time
    t_max = 15.0
    store_every = 10
    
    print(f"\nEvolving system to t = {t_max}...")
    times, psi_t = solver.evolve(t_max=t_max, store_every=store_every)
    
    # Analyze interference pattern beyond the slits
    analysis_position = 20.0  # Position where we analyze the pattern
    analysis_idx = np.argmin(np.abs(x - analysis_position))
    
    final_prob = np.abs(psi_t[-1])**2
    
    print(f"\nFinal Properties:")
    print(f"  Final norm: ||ψ|| = {solver.get_norm():.6f}")
    print(f"  Total probability beyond slits: {np.sum(final_prob[x > slit_position]) * dx * 100:.2f}%")
    
    # Create visualizations
    print("\nCreating plots...")
    
    # Plot final probability distribution with detail
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full view
    ax1.fill_between(x, final_prob, alpha=0.6, color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, V / np.max(V) * np.max(final_prob) * 0.3, 'k-', 
                  linewidth=2, alpha=0.5, label='Double slit')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Probability Density |ψ|²')
    ax1.set_title('Final Probability Distribution')
    ax1.grid(True, alpha=0.3)
    ax1_twin.set_ylabel('Barrier (scaled)')
    
    # Zoom in on interference region beyond slits
    zoom_start = slit_position + 5
    zoom_end = analysis_position + 10
    zoom_mask = (x >= zoom_start) & (x <= zoom_end)
    
    ax2.fill_between(x[zoom_mask], final_prob[zoom_mask], alpha=0.6, color='green')
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Probability Density |ψ|²')
    ax2.set_title('Interference Pattern (Zoomed)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('double_slit_interference.png', dpi=150, bbox_inches='tight')
    print("  Saved: double_slit_interference.png")
    
    # Space-time plot
    fig2, _ = plot_spacetime_density(x, times, psi_t)
    fig2.suptitle('Double Slit: Space-Time Evolution')
    plt.savefig('double_slit_spacetime.png', dpi=150, bbox_inches='tight')
    print("  Saved: double_slit_spacetime.png")
    
    # Create animation
    print("  Creating animation (this may take a moment)...")
    anim = animate_wavefunction(x, times, psi_t, V,
                                filename='double_slit_animation.gif',
                                fps=30,
                                title='Double Slit Interference')
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("Note: Interference fringes should be visible in the region")
    print("      beyond the slits if the wave packet was wide enough")
    print("      to pass through both slits simultaneously.")
    print("=" * 60)


if __name__ == "__main__":
    main()
