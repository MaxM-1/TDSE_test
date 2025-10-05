"""
Visualization tools for wave function evolution and probability densities.

This module provides functions for plotting and animating quantum states.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_wavefunction(x, psi, V=None, title="Wave Function", figsize=(10, 6)):
    """
    Plot the wave function and probability density.
    
    Parameters:
        x (ndarray): Spatial grid
        psi (ndarray): Complex wave function
        V (ndarray, optional): Potential energy to plot
        title (str): Plot title
        figsize (tuple): Figure size
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot real and imaginary parts
    ax1.plot(x, np.real(psi), label='Re(ψ)', color='blue', alpha=0.7)
    ax1.plot(x, np.imag(psi), label='Im(ψ)', color='red', alpha=0.7)
    ax1.plot(x, np.abs(psi), label='|ψ|', color='black', linewidth=2)
    ax1.set_ylabel('Wave Function')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    
    # Plot probability density
    prob = np.abs(psi)**2
    ax2.fill_between(x, prob, alpha=0.6, color='green', label='|ψ|²')
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Probability Density')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Add potential if provided
    if V is not None:
        ax2_twin = ax2.twinx()
        # Scale potential for visibility
        V_scaled = V / np.max(np.abs(V)) * np.max(prob) * 0.5 if np.max(np.abs(V)) > 0 else V
        ax2_twin.plot(x, V_scaled, 'k--', alpha=0.5, linewidth=2, label='V(x)')
        ax2_twin.set_ylabel('Potential (scaled)')
        ax2_twin.legend(loc='upper left')
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_evolution(x, times, psi_t, V=None, figsize=(12, 8)):
    """
    Plot snapshots of wave function evolution.
    
    Parameters:
        x (ndarray): Spatial grid
        times (ndarray): Time points
        psi_t (ndarray): Wave function at each time (shape: [n_times, N])
        V (ndarray, optional): Potential energy
        figsize (tuple): Figure size
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    n_snapshots = min(6, len(times))
    indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(n_snapshots, 1, figsize=figsize, sharex=True)
    
    if n_snapshots == 1:
        axes = [axes]
    
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        psi = psi_t[idx]
        prob = np.abs(psi)**2
        
        ax.plot(x, np.real(psi), 'b-', alpha=0.5, label='Re(ψ)')
        ax.plot(x, np.imag(psi), 'r-', alpha=0.5, label='Im(ψ)')
        ax.fill_between(x, prob, alpha=0.3, color='green', label='|ψ|²')
        
        if V is not None and i == 0:
            ax_twin = ax.twinx()
            V_scaled = V / np.max(np.abs(V)) * np.max(prob) * 0.5 if np.max(np.abs(V)) > 0 else V
            ax_twin.plot(x, V_scaled, 'k--', alpha=0.5, linewidth=1.5)
            ax_twin.set_ylabel('V(x)', fontsize=8)
            ax_twin.tick_params(labelsize=8)
        
        ax.set_ylabel(f't = {times[idx]:.2f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Position (x)')
    plt.tight_layout()
    return fig, axes


def animate_wavefunction(x, times, psi_t, V=None, filename=None, fps=30, 
                        title="Wave Function Evolution", figsize=(10, 6)):
    """
    Create an animation of wave function evolution.
    
    Parameters:
        x (ndarray): Spatial grid
        times (ndarray): Time points
        psi_t (ndarray): Wave function at each time (shape: [n_times, N])
        V (ndarray, optional): Potential energy
        filename (str, optional): Filename to save animation (e.g., 'anim.gif')
        fps (int): Frames per second
        title (str): Animation title
        figsize (tuple): Figure size
        
    Returns:
        animation: Matplotlib animation object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Initialize plots
    line_real, = ax1.plot([], [], 'b-', alpha=0.7, label='Re(ψ)')
    line_imag, = ax1.plot([], [], 'r-', alpha=0.7, label='Im(ψ)')
    line_abs, = ax1.plot([], [], 'k-', linewidth=2, label='|ψ|')
    
    # Probability density
    prob_fill = ax2.fill_between(x, 0, alpha=0.6, color='green', label='|ψ|²')
    
    # Set up axes
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(-np.max(np.abs(psi_t)) * 1.1, np.max(np.abs(psi_t)) * 1.1)
    ax1.set_ylabel('Wave Function')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(0, np.max(np.abs(psi_t)**2) * 1.1)
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True, alpha=0.3)
    
    # Add potential
    if V is not None:
        ax2_twin = ax2.twinx()
        V_scaled = V / np.max(np.abs(V)) * np.max(np.abs(psi_t)**2) * 0.5 if np.max(np.abs(V)) > 0 else V
        ax2_twin.plot(x, V_scaled, 'k--', alpha=0.5, linewidth=2, label='V(x)')
        ax2_twin.set_ylabel('Potential (scaled)')
        ax2_twin.legend(loc='upper left')
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14)
    
    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        line_abs.set_data([], [])
        time_text.set_text('')
        return line_real, line_imag, line_abs, time_text
    
    def animate(frame):
        psi = psi_t[frame]
        
        line_real.set_data(x, np.real(psi))
        line_imag.set_data(x, np.imag(psi))
        line_abs.set_data(x, np.abs(psi))
        
        # Update probability density
        prob = np.abs(psi)**2
        # Clear collections properly
        for coll in ax2.collections:
            coll.remove()
        ax2.fill_between(x, prob, alpha=0.6, color='green')
        
        time_text.set_text(f'Time = {times[frame]:.3f}')
        
        return line_real, line_imag, line_abs, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(times),
                        interval=1000/fps, blit=False, repeat=True)
    
    if filename:
        if filename.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer)
            print(f"Animation saved to {filename}")
        elif filename.endswith('.mp4'):
            anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
            print(f"Animation saved to {filename}")
    
    plt.tight_layout()
    return anim


def plot_probability_current(x, times, psi_t, hbar=1.0, m=1.0, figsize=(10, 6)):
    """
    Plot the probability current density j(x,t) = (ℏ/m) Im(ψ* ∂ψ/∂x).
    
    Parameters:
        x (ndarray): Spatial grid
        times (ndarray): Time points
        psi_t (ndarray): Wave function at each time
        hbar (float): Reduced Planck constant
        m (float): Mass
        figsize (tuple): Figure size
        
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    dx = x[1] - x[0]
    
    # Calculate probability current for each time
    j_t = np.zeros((len(times), len(x)))
    
    for i, psi in enumerate(psi_t):
        # Gradient of wave function
        dpsi_dx = np.gradient(psi, dx)
        # Current density
        j_t[i] = (hbar / m) * np.imag(np.conj(psi) * dpsi_dx)
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=figsize)
    
    T, X = np.meshgrid(times, x, indexing='ij')
    
    levels = 20
    contour = ax.contourf(X, T, j_t, levels=levels, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, label='Probability Current j(x,t)')
    
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Time (t)')
    ax.set_title('Probability Current Density')
    
    plt.tight_layout()
    return fig, ax


def plot_spacetime_density(x, times, psi_t, figsize=(10, 6)):
    """
    Create a space-time plot of probability density.
    
    Parameters:
        x (ndarray): Spatial grid
        times (ndarray): Time points
        psi_t (ndarray): Wave function at each time
        figsize (tuple): Figure size
        
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    prob_t = np.abs(psi_t)**2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    T, X = np.meshgrid(times, x, indexing='ij')
    
    contour = ax.contourf(X, T, prob_t, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Probability Density |ψ|²')
    
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Time (t)')
    ax.set_title('Space-Time Evolution of Probability Density')
    
    plt.tight_layout()
    return fig, ax


def plot_momentum_space(k, phi, title="Momentum Space Wave Function", figsize=(10, 6)):
    """
    Plot the momentum space wave function.
    
    Parameters:
        k (ndarray): Momentum grid
        phi (ndarray): Momentum space wave function
        title (str): Plot title
        figsize (tuple): Figure size
        
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    prob_k = np.abs(phi)**2
    
    ax.plot(k, np.real(phi), 'b-', alpha=0.5, label='Re(φ)')
    ax.plot(k, np.imag(phi), 'r-', alpha=0.5, label='Im(φ)')
    ax.fill_between(k, prob_k, alpha=0.3, color='purple', label='|φ|²')
    
    ax.set_xlabel('Momentum (k)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax
