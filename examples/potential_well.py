"""
Potential Well Example

Demonstrates bound state dynamics in a square or Gaussian potential well.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tdse_solver import TDSESolver, GaussianWavePacket
from tdse_solver.potentials import SquareWell, GaussianWell
from tdse_solver.visualization import animate_wavefunction, plot_evolution


def main():
    print("=" * 60)
    print("Potential Well Simulation")
    print("=" * 60)
    
    # Set up spatial grid
    x = np.linspace(-30, 30, 512)
    dx = x[1] - x[0]
    
    # Choose potential type
    use_square = True  # Set to False for Gaussian well
    
    if use_square:
        well_depth = -2.0
        well_width = 10.0
        V = SquareWell(x, depth=well_depth, width=well_width, center=0.0)
        well_type = "Square"
    else:
        well_depth = -2.0
        well_width = 3.0
        V = GaussianWell(x, depth=well_depth, width=well_width, center=0.0)
        well_type = "Gaussian"
    
    print(f"\n{well_type} Potential Well:")
    print(f"  Depth: V0 = {well_depth}")
    print(f"  Width: w = {well_width}")
    
    # Create initial wave packet (centered in well, but with some momentum)
    x0 = -3.0   # Slightly off-center
    k0 = 0.5    # Small momentum
    sigma = 2.0
    
    print(f"\nInitial Wave Packet:")
    print(f"  Position: x0 = {x0}")
    print(f"  Momentum: k0 = {k0}")
    print(f"  Width: σ = {sigma}")
    
    psi0 = GaussianWavePacket(x, x0=x0, k0=k0, sigma=sigma)
    
    # Create solver
    dt = 0.05
    solver = TDSESolver(x, psi0, V, dt=dt)
    
    E0 = solver.get_energy()
    print(f"\nInitial energy: E = {E0:.4f}")
    
    # Check if state is bound (E < 0 for well bottom at V0)
    if E0 < 0:
        print("  State is BOUND (E < 0)")
    else:
        print("  State has energy above well rim")
    
    # Evolve in time
    t_max = 50.0
    store_every = 5
    
    print(f"\nEvolving system to t = {t_max}...")
    times, psi_t = solver.evolve(t_max=t_max, store_every=store_every)
    
    # Calculate probability inside and outside well
    if use_square:
        inside_mask = np.abs(x) <= well_width / 2
    else:
        # For Gaussian, use region where V < well_depth/2
        inside_mask = V < well_depth / 2
    
    prob_inside = np.zeros(len(times))
    prob_outside = np.zeros(len(times))
    energy = np.zeros(len(times))
    
    for i, psi in enumerate(psi_t):
        prob = np.abs(psi)**2
        prob_inside[i] = np.sum(prob[inside_mask]) * dx
        prob_outside[i] = np.sum(prob[~inside_mask]) * dx
        
        solver.psi = psi
        energy[i] = solver.get_energy()
    
    print(f"\nFinal Properties:")
    print(f"  Probability inside well: {prob_inside[-1] * 100:.2f}%")
    print(f"  Probability outside well: {prob_outside[-1] * 100:.2f}%")
    print(f"  Energy conservation: ΔE/E0 = {abs(energy[-1] - energy[0]) / abs(energy[0]) * 100:.4f}%")
    
    # Create visualizations
    print("\nCreating plots...")
    
    # Plot probability distribution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(times, prob_inside, 'b-', linewidth=2, label='Inside well')
    ax1.plot(times, prob_outside, 'r-', linewidth=2, label='Outside well')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.plot(times, energy, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Well rim')
    ax2.axhline(y=well_depth, color='r', linestyle='--', alpha=0.5, label='Well bottom')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy')
    ax2.set_title('Total Energy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('potential_well_observables.png', dpi=150, bbox_inches='tight')
    print("  Saved: potential_well_observables.png")
    
    # Plot snapshots
    fig2, _ = plot_evolution(x, times, psi_t, V)
    fig2.suptitle(f'{well_type} Potential Well: Wave Packet Evolution')
    plt.savefig('potential_well_snapshots.png', dpi=150, bbox_inches='tight')
    print("  Saved: potential_well_snapshots.png")
    
    # Create animation
    print("  Creating animation (this may take a moment)...")
    anim = animate_wavefunction(x, times, psi_t, V,
                                filename='potential_well_animation.gif',
                                fps=30,
                                title=f'{well_type} Potential Well')
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
