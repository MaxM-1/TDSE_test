"""
Core solver module for the time-dependent Schrödinger equation.

This module implements the split-operator method for solving the TDSE.
"""

import numpy as np
from scipy import fftpack


class TDSESolver:
    """
    Time-Dependent Schrödinger Equation solver using the split-operator method.
    
    The TDSE is solved using the split-operator algorithm, which separates
    the time evolution operator into kinetic and potential parts. The kinetic
    operator is applied in momentum space via FFT for accuracy.
    
    Attributes:
        x (ndarray): Spatial grid points
        dx (float): Spatial grid spacing
        dt (float): Time step
        psi (ndarray): Current wave function (complex)
        V (ndarray): Potential energy array
        k (ndarray): Momentum space grid
        hbar (float): Reduced Planck constant (default: 1.0 in atomic units)
        m (float): Particle mass (default: 1.0 in atomic units)
    """
    
    def __init__(self, x, psi_initial, V, dt=0.01, hbar=1.0, m=1.0):
        """
        Initialize the TDSE solver.
        
        Parameters:
            x (ndarray): Spatial grid points
            psi_initial (ndarray): Initial wave function (complex)
            V (ndarray or callable): Potential energy (array or function of x)
            dt (float): Time step for evolution
            hbar (float): Reduced Planck constant (atomic units)
            m (float): Particle mass (atomic units)
        """
        self.x = np.asarray(x)
        self.N = len(x)
        self.dx = x[1] - x[0]
        self.dt = dt
        self.hbar = hbar
        self.m = m
        
        # Initialize wave function
        self.psi = np.asarray(psi_initial, dtype=complex)
        self._normalize()
        
        # Set up potential
        if callable(V):
            self.V = V(x)
        else:
            self.V = np.asarray(V)
        
        # Set up momentum space grid
        self.k = fftpack.fftfreq(self.N, d=self.dx) * 2 * np.pi
        
        # Precompute evolution operators
        self._setup_operators()
        
        # Storage for observables
        self.time = 0.0
        
    def _setup_operators(self):
        """Precompute the time evolution operators."""
        # Potential evolution operator (half step)
        self.exp_V_half = np.exp(-1j * self.V * self.dt / (2 * self.hbar))
        
        # Kinetic evolution operator (full step)
        KE = (self.hbar**2 * self.k**2) / (2 * self.m)
        self.exp_K = np.exp(-1j * KE * self.dt / self.hbar)
    
    def _normalize(self):
        """Normalize the wave function."""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        self.psi /= norm
    
    def step(self):
        """
        Perform one time step using the split-operator method.
        
        The evolution is: exp(-iVdt/2) * exp(-iKdt) * exp(-iVdt/2) * psi
        """
        # Apply potential operator (half step)
        self.psi *= self.exp_V_half
        
        # Transform to momentum space
        psi_k = fftpack.fft(self.psi)
        
        # Apply kinetic operator (full step)
        psi_k *= self.exp_K
        
        # Transform back to position space
        self.psi = fftpack.ifft(psi_k)
        
        # Apply potential operator (half step)
        self.psi *= self.exp_V_half
        
        # Update time
        self.time += self.dt
    
    def evolve(self, t_max, store_every=1):
        """
        Evolve the wave function up to a maximum time.
        
        Parameters:
            t_max (float): Maximum time to evolve
            store_every (int): Store wave function every N steps
            
        Returns:
            times (ndarray): Array of time points
            psi_t (ndarray): Wave function at each time point (shape: [n_times, N])
        """
        n_steps = int(t_max / self.dt)
        n_store = n_steps // store_every + 1
        
        times = np.zeros(n_store)
        psi_t = np.zeros((n_store, self.N), dtype=complex)
        
        # Store initial state
        times[0] = self.time
        psi_t[0] = self.psi.copy()
        
        # Time evolution
        store_idx = 1
        for step in range(1, n_steps + 1):
            self.step()
            
            if step % store_every == 0 and store_idx < n_store:
                times[store_idx] = self.time
                psi_t[store_idx] = self.psi.copy()
                store_idx += 1
        
        return times[:store_idx], psi_t[:store_idx]
    
    def get_probability_density(self):
        """
        Get the probability density |ψ|².
        
        Returns:
            ndarray: Probability density
        """
        return np.abs(self.psi)**2
    
    def get_expectation_x(self):
        """
        Calculate expectation value of position <x>.
        
        Returns:
            float: Expectation value of position
        """
        prob_density = self.get_probability_density()
        return np.sum(self.x * prob_density) * self.dx
    
    def get_expectation_x2(self):
        """
        Calculate expectation value of position squared <x²>.
        
        Returns:
            float: Expectation value of x²
        """
        prob_density = self.get_probability_density()
        return np.sum(self.x**2 * prob_density) * self.dx
    
    def get_expectation_p(self):
        """
        Calculate expectation value of momentum <p>.
        
        Returns:
            float: Expectation value of momentum
        """
        # Calculate derivative in position space
        psi_dx = np.gradient(self.psi, self.dx)
        integrand = np.conj(self.psi) * (-1j * self.hbar * psi_dx)
        return np.real(np.sum(integrand) * self.dx)
    
    def get_energy(self):
        """
        Calculate total energy expectation value <H>.
        
        Returns:
            float: Total energy (kinetic + potential)
        """
        # Kinetic energy
        psi_k = fftpack.fft(self.psi)
        KE = (self.hbar**2 * self.k**2) / (2 * self.m)
        T = np.sum(KE * np.abs(psi_k)**2) * self.dx / self.N
        
        # Potential energy
        prob_density = self.get_probability_density()
        V_exp = np.sum(self.V * prob_density) * self.dx
        
        return T + V_exp
    
    def get_norm(self):
        """
        Calculate the norm of the wave function (should be 1).
        
        Returns:
            float: Norm of wave function
        """
        return np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
    
    def reset(self, psi_initial):
        """
        Reset the solver with a new initial wave function.
        
        Parameters:
            psi_initial (ndarray): New initial wave function
        """
        self.psi = np.asarray(psi_initial, dtype=complex)
        self._normalize()
        self.time = 0.0
