"""
Wave function utilities for creating and manipulating quantum states.

This module provides functions for creating initial wave packets and
normalizing wave functions.
"""

import numpy as np


def GaussianWavePacket(x, x0=0.0, k0=0.0, sigma=1.0, normalize=True):
    """
    Create a Gaussian wave packet.
    
    The wave packet has the form:
    ψ(x) = (1/(2πσ²)^(1/4)) * exp(ik₀(x-x₀)) * exp(-(x-x₀)²/(4σ²))
    
    Parameters:
        x (ndarray): Spatial grid
        x0 (float): Initial center position
        k0 (float): Initial momentum (wave number)
        sigma (float): Width parameter (standard deviation)
        normalize (bool): Whether to normalize the wave function
        
    Returns:
        ndarray: Complex wave function
    """
    # Normalization constant
    if normalize:
        A = (1.0 / (2 * np.pi * sigma**2))**(0.25)
    else:
        A = 1.0
    
    # Gaussian envelope
    envelope = np.exp(-((x - x0)**2) / (4 * sigma**2))
    
    # Plane wave
    plane_wave = np.exp(1j * k0 * x)
    
    psi = A * envelope * plane_wave
    
    if normalize:
        psi = normalize_wavefunction(psi, x)
    
    return psi


def PlaneWave(x, k=1.0, amplitude=1.0, normalize=True):
    """
    Create a plane wave state.
    
    ψ(x) = A * exp(ikx)
    
    Parameters:
        x (ndarray): Spatial grid
        k (float): Wave number (momentum)
        amplitude (float): Amplitude
        normalize (bool): Whether to normalize
        
    Returns:
        ndarray: Complex wave function
    """
    psi = amplitude * np.exp(1j * k * x)
    
    if normalize:
        psi = normalize_wavefunction(psi, x)
    
    return psi


def GroundStateHO(x, omega=1.0, m=1.0, hbar=1.0, center=0.0):
    """
    Ground state of the quantum harmonic oscillator.
    
    ψ₀(x) = (mω/πℏ)^(1/4) * exp(-mω(x-x₀)²/(2ℏ))
    
    Parameters:
        x (ndarray): Spatial grid
        omega (float): Angular frequency
        m (float): Mass
        hbar (float): Reduced Planck constant
        center (float): Center position
        
    Returns:
        ndarray: Real wave function (ground state)
    """
    alpha = m * omega / hbar
    A = (alpha / np.pi)**(0.25)
    
    psi = A * np.exp(-alpha * (x - center)**2 / 2)
    
    return psi


def ExcitedStateHO(x, n, omega=1.0, m=1.0, hbar=1.0, center=0.0):
    """
    Excited state of the quantum harmonic oscillator using Hermite polynomials.
    
    Parameters:
        x (ndarray): Spatial grid
        n (int): Quantum number (0, 1, 2, ...)
        omega (float): Angular frequency
        m (float): Mass
        hbar (float): Reduced Planck constant
        center (float): Center position
        
    Returns:
        ndarray: Real wave function (nth excited state)
    """
    from scipy.special import hermite
    
    alpha = m * omega / hbar
    xi = np.sqrt(alpha) * (x - center)
    
    # Normalization constant
    A = (alpha / np.pi)**(0.25) / np.sqrt(2**n * np.math.factorial(n))
    
    # Hermite polynomial
    Hn = hermite(n)
    
    psi = A * Hn(xi) * np.exp(-xi**2 / 2)
    
    return psi


def SquareWavePacket(x, x0=0.0, width=2.0, k0=0.0, normalize=True):
    """
    Create a square wave packet (uniform amplitude in a region).
    
    Parameters:
        x (ndarray): Spatial grid
        x0 (float): Center position
        width (float): Width of the packet
        k0 (float): Wave number
        normalize (bool): Whether to normalize
        
    Returns:
        ndarray: Complex wave function
    """
    psi = np.zeros_like(x, dtype=complex)
    
    # Square envelope
    mask = np.abs(x - x0) <= width / 2
    psi[mask] = np.exp(1j * k0 * x[mask])
    
    if normalize:
        psi = normalize_wavefunction(psi, x)
    
    return psi


def TriangularWavePacket(x, x0=0.0, width=2.0, k0=0.0, normalize=True):
    """
    Create a triangular wave packet.
    
    Parameters:
        x (ndarray): Spatial grid
        x0 (float): Center position
        width (float): Base width
        k0 (float): Wave number
        normalize (bool): Whether to normalize
        
    Returns:
        ndarray: Complex wave function
    """
    psi = np.zeros_like(x, dtype=complex)
    
    # Triangular envelope
    mask = np.abs(x - x0) <= width / 2
    envelope = 1.0 - 2.0 * np.abs(x - x0) / width
    psi[mask] = envelope[mask] * np.exp(1j * k0 * x[mask])
    
    if normalize:
        psi = normalize_wavefunction(psi, x)
    
    return psi


def Superposition(x, psi1, psi2, c1=1.0, c2=1.0, normalize=True):
    """
    Create a superposition of two wave functions.
    
    ψ = c₁ψ₁ + c₂ψ₂
    
    Parameters:
        x (ndarray): Spatial grid
        psi1 (ndarray): First wave function
        psi2 (ndarray): Second wave function
        c1 (complex): Coefficient for first state
        c2 (complex): Coefficient for second state
        normalize (bool): Whether to normalize
        
    Returns:
        ndarray: Superposed wave function
    """
    psi = c1 * psi1 + c2 * psi2
    
    if normalize:
        psi = normalize_wavefunction(psi, x)
    
    return psi


def normalize_wavefunction(psi, x):
    """
    Normalize a wave function so that ∫|ψ|²dx = 1.
    
    Parameters:
        psi (ndarray): Wave function to normalize
        x (ndarray): Spatial grid
        
    Returns:
        ndarray: Normalized wave function
    """
    dx = x[1] - x[0]
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    
    if norm > 1e-10:  # Avoid division by zero
        return psi / norm
    else:
        return psi


def calculate_overlap(psi1, psi2, x):
    """
    Calculate the overlap between two wave functions.
    
    <ψ₁|ψ₂> = ∫ ψ₁* ψ₂ dx
    
    Parameters:
        psi1 (ndarray): First wave function
        psi2 (ndarray): Second wave function
        x (ndarray): Spatial grid
        
    Returns:
        complex: Overlap integral
    """
    dx = x[1] - x[0]
    return np.sum(np.conj(psi1) * psi2) * dx


def get_momentum_distribution(psi, x):
    """
    Calculate the momentum space wave function via Fourier transform.
    
    Parameters:
        psi (ndarray): Position space wave function
        x (ndarray): Spatial grid
        
    Returns:
        k (ndarray): Momentum space grid
        phi (ndarray): Momentum space wave function
    """
    dx = x[1] - x[0]
    N = len(x)
    
    # Fourier transform (with proper normalization)
    phi = np.fft.fftshift(np.fft.fft(psi)) * dx / np.sqrt(2 * np.pi)
    
    # Momentum grid
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dx) * 2 * np.pi)
    
    return k, phi


def spreading_width(psi, x):
    """
    Calculate the width (standard deviation) of a wave packet.
    
    Δx = √(<x²> - <x>²)
    
    Parameters:
        psi (ndarray): Wave function
        x (ndarray): Spatial grid
        
    Returns:
        float: Width of the wave packet
    """
    dx = x[1] - x[0]
    prob = np.abs(psi)**2
    
    x_mean = np.sum(x * prob) * dx
    x2_mean = np.sum(x**2 * prob) * dx
    
    return np.sqrt(x2_mean - x_mean**2)
