"""
Potential energy functions for the TDSE solver.

This module provides various potential energy functions commonly used
in quantum mechanics simulations.
"""

import numpy as np


def HarmonicOscillator(x, omega=1.0, m=1.0, hbar=1.0, center=0.0):
    """
    Harmonic oscillator potential: V(x) = (1/2) m ω² (x - x₀)²
    
    Parameters:
        x (ndarray): Spatial grid
        omega (float): Angular frequency
        m (float): Mass
        hbar (float): Reduced Planck constant
        center (float): Center position of the oscillator
        
    Returns:
        ndarray: Potential energy at each point
    """
    return 0.5 * m * omega**2 * (x - center)**2


def SquareBarrier(x, height=1.0, width=2.0, center=0.0):
    """
    Square potential barrier.
    
    Parameters:
        x (ndarray): Spatial grid
        height (float): Barrier height
        width (float): Barrier width
        center (float): Center position of the barrier
        
    Returns:
        ndarray: Potential energy at each point
    """
    V = np.zeros_like(x)
    barrier_mask = np.abs(x - center) <= width / 2
    V[barrier_mask] = height
    return V


def SquareWell(x, depth=-1.0, width=2.0, center=0.0):
    """
    Square potential well.
    
    Parameters:
        x (ndarray): Spatial grid
        depth (float): Well depth (negative value)
        width (float): Well width
        center (float): Center position of the well
        
    Returns:
        ndarray: Potential energy at each point
    """
    V = np.zeros_like(x)
    well_mask = np.abs(x - center) <= width / 2
    V[well_mask] = depth
    return V


def DoubleSlit(x, slit_separation=10.0, slit_width=1.0, barrier_height=10.0, center=0.0):
    """
    Double slit potential (two narrow gaps in a barrier).
    
    Parameters:
        x (ndarray): Spatial grid
        slit_separation (float): Distance between slit centers
        slit_width (float): Width of each slit
        barrier_height (float): Height of the barrier
        center (float): Center position between slits
        
    Returns:
        ndarray: Potential energy at each point
    """
    V = np.ones_like(x) * barrier_height
    
    # Create two slits
    slit1_mask = np.abs(x - (center - slit_separation/2)) <= slit_width / 2
    slit2_mask = np.abs(x - (center + slit_separation/2)) <= slit_width / 2
    
    V[slit1_mask] = 0.0
    V[slit2_mask] = 0.0
    
    return V


def GaussianBarrier(x, height=1.0, width=2.0, center=0.0):
    """
    Gaussian-shaped potential barrier.
    
    Parameters:
        x (ndarray): Spatial grid
        height (float): Peak height
        width (float): Width parameter (standard deviation)
        center (float): Center position
        
    Returns:
        ndarray: Potential energy at each point
    """
    return height * np.exp(-((x - center)**2) / (2 * width**2))


def GaussianWell(x, depth=-1.0, width=2.0, center=0.0):
    """
    Gaussian-shaped potential well.
    
    Parameters:
        x (ndarray): Spatial grid
        depth (float): Maximum depth (negative value)
        width (float): Width parameter (standard deviation)
        center (float): Center position
        
    Returns:
        ndarray: Potential energy at each point
    """
    return depth * np.exp(-((x - center)**2) / (2 * width**2))


def DoubleWell(x, separation=5.0, depth=-1.0, width=1.0):
    """
    Double well potential (two Gaussian wells).
    
    Parameters:
        x (ndarray): Spatial grid
        separation (float): Distance between well centers
        depth (float): Well depth (negative value)
        width (float): Width of each well
        
    Returns:
        ndarray: Potential energy at each point
    """
    well1 = GaussianWell(x, depth=depth, width=width, center=-separation/2)
    well2 = GaussianWell(x, depth=depth, width=width, center=separation/2)
    return well1 + well2


def PeriodicPotential(x, amplitude=1.0, period=2*np.pi, phase=0.0):
    """
    Periodic potential: V(x) = A cos(2πx/λ + φ)
    
    Parameters:
        x (ndarray): Spatial grid
        amplitude (float): Potential amplitude
        period (float): Spatial period
        phase (float): Phase shift
        
    Returns:
        ndarray: Potential energy at each point
    """
    return amplitude * np.cos(2 * np.pi * x / period + phase)


def StepPotential(x, height=1.0, position=0.0):
    """
    Step potential (sudden change in potential).
    
    Parameters:
        x (ndarray): Spatial grid
        height (float): Height of the step
        position (float): Position of the step
        
    Returns:
        ndarray: Potential energy at each point
    """
    V = np.zeros_like(x)
    V[x >= position] = height
    return V


def LinearPotential(x, slope=0.1, offset=0.0):
    """
    Linear potential (uniform force): V(x) = ax + b
    
    Parameters:
        x (ndarray): Spatial grid
        slope (float): Slope (force = -slope)
        offset (float): Offset
        
    Returns:
        ndarray: Potential energy at each point
    """
    return slope * x + offset


def CoulombPotential(x, charge=-1.0, epsilon=0.1, center=0.0):
    """
    Coulomb-like potential: V(x) = q / (|x - x₀| + ε)
    
    The epsilon parameter prevents singularity at x = center.
    
    Parameters:
        x (ndarray): Spatial grid
        charge (float): Effective charge
        epsilon (float): Regularization parameter
        center (float): Center position
        
    Returns:
        ndarray: Potential energy at each point
    """
    return charge / (np.abs(x - center) + epsilon)


def CustomPotential(x, func):
    """
    Create a custom potential from a user-defined function.
    
    Parameters:
        x (ndarray): Spatial grid
        func (callable): Function that takes x and returns V(x)
        
    Returns:
        ndarray: Potential energy at each point
    """
    return func(x)


def ZeroPotential(x):
    """
    Zero potential (free particle).
    
    Parameters:
        x (ndarray): Spatial grid
        
    Returns:
        ndarray: Zero array
    """
    return np.zeros_like(x)
