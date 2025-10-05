"""
Unit tests for the TDSE solver.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tdse_solver import TDSESolver, GaussianWavePacket
from tdse_solver.potentials import ZeroPotential, HarmonicOscillator, SquareBarrier


class TestTDSESolver:
    """Test cases for the TDSE solver."""
    
    def test_initialization(self):
        """Test solver initialization."""
        x = np.linspace(-10, 10, 128)
        psi0 = GaussianWavePacket(x, x0=0, k0=1, sigma=1)
        V = ZeroPotential(x)
        
        solver = TDSESolver(x, psi0, V, dt=0.01)
        
        assert solver.N == 128
        assert solver.dt == 0.01
        assert np.allclose(solver.get_norm(), 1.0)
    
    def test_normalization(self):
        """Test that wave function normalization works."""
        x = np.linspace(-20, 20, 256)
        psi0 = GaussianWavePacket(x, x0=0, k0=0, sigma=2)
        V = ZeroPotential(x)
        
        solver = TDSESolver(x, psi0, V, dt=0.01)
        
        # Check initial normalization
        assert np.allclose(solver.get_norm(), 1.0, atol=1e-6)
        
        # Evolve and check normalization is preserved
        solver.evolve(t_max=1.0)
        assert np.allclose(solver.get_norm(), 1.0, atol=1e-4)
    
    def test_free_particle_propagation(self):
        """Test free particle motion."""
        x = np.linspace(-30, 30, 512)
        dx = x[1] - x[0]
        
        x0 = -10.0
        k0 = 2.0
        sigma = 2.0
        
        psi0 = GaussianWavePacket(x, x0=x0, k0=k0, sigma=sigma)
        V = ZeroPotential(x)
        
        solver = TDSESolver(x, psi0, V, dt=0.01)
        
        # Expected velocity: v = k0 (in atomic units with m=1)
        t_max = 5.0
        solver.evolve(t_max=t_max)
        
        # Check that center of wave packet has moved
        x_final = solver.get_expectation_x()
        x_expected = x0 + k0 * t_max
        
        # Allow some tolerance due to numerical errors and spreading
        assert np.abs(x_final - x_expected) < 0.5
    
    def test_energy_conservation(self):
        """Test that energy is conserved during evolution."""
        x = np.linspace(-20, 20, 256)
        psi0 = GaussianWavePacket(x, x0=0, k0=1, sigma=2)
        V = HarmonicOscillator(x, omega=0.5)
        
        solver = TDSESolver(x, psi0, V, dt=0.01)
        
        E0 = solver.get_energy()
        solver.evolve(t_max=10.0)
        E_final = solver.get_energy()
        
        # Energy should be conserved to high precision
        relative_error = np.abs(E_final - E0) / np.abs(E0)
        assert relative_error < 1e-3
    
    def test_harmonic_oscillator_period(self):
        """Test oscillation period in harmonic potential."""
        x = np.linspace(-20, 20, 256)
        dx = x[1] - x[0]
        
        omega = 0.5
        V = HarmonicOscillator(x, omega=omega)
        
        # Displaced Gaussian (coherent state)
        x0 = 5.0
        psi0 = GaussianWavePacket(x, x0=x0, k0=0, sigma=1.0)
        
        solver = TDSESolver(x, psi0, V, dt=0.05)
        
        # Evolve for one period
        T = 2 * np.pi / omega
        times, psi_t = solver.evolve(t_max=T, store_every=5)
        
        # Calculate position expectation values
        x_exp = []
        for psi in psi_t:
            solver.psi = psi
            x_exp.append(solver.get_expectation_x())
        
        x_exp = np.array(x_exp)
        
        # Check that it returns close to initial position
        # (won't be exact due to discretization)
        assert np.abs(x_exp[0] - x_exp[-1]) < 1.0
    
    def test_momentum_expectation(self):
        """Test momentum expectation value calculation."""
        x = np.linspace(-20, 20, 512)
        
        k0 = 2.0
        psi0 = GaussianWavePacket(x, x0=0, k0=k0, sigma=2)
        V = ZeroPotential(x)
        
        solver = TDSESolver(x, psi0, V, dt=0.01)
        
        p_exp = solver.get_expectation_p()
        
        # Expected momentum is k0 (in atomic units with hbar=1)
        assert np.abs(p_exp - k0) < 0.1
    
    def test_potential_callable(self):
        """Test that callable potentials work."""
        x = np.linspace(-10, 10, 128)
        
        def custom_potential(x):
            return 0.5 * x**2
        
        psi0 = GaussianWavePacket(x, x0=0, k0=0, sigma=1)
        
        solver = TDSESolver(x, psi0, custom_potential, dt=0.01)
        
        assert np.allclose(solver.V, 0.5 * x**2)


class TestConservationLaws:
    """Test conservation laws."""
    
    def test_probability_conservation(self):
        """Test that total probability is conserved."""
        x = np.linspace(-30, 30, 512)
        psi0 = GaussianWavePacket(x, x0=-10, k0=2, sigma=2)
        V = SquareBarrier(x, height=1.5, width=5, center=0)
        
        solver = TDSESolver(x, psi0, V, dt=0.01)
        
        times, psi_t = solver.evolve(t_max=20.0, store_every=10)
        
        # Check normalization at all time steps
        for psi in psi_t:
            solver.psi = psi
            assert np.allclose(solver.get_norm(), 1.0, atol=1e-3)
    
    def test_energy_harmonic_oscillator(self):
        """Test energy conservation in harmonic oscillator."""
        x = np.linspace(-20, 20, 256)
        psi0 = GaussianWavePacket(x, x0=3, k0=0, sigma=1.5)
        V = HarmonicOscillator(x, omega=0.8)
        
        solver = TDSESolver(x, psi0, V, dt=0.02)
        
        E0 = solver.get_energy()
        
        times, psi_t = solver.evolve(t_max=15.0, store_every=5)
        
        energies = []
        for psi in psi_t:
            solver.psi = psi
            energies.append(solver.get_energy())
        
        energies = np.array(energies)
        
        # All energies should be close to initial energy
        max_deviation = np.max(np.abs(energies - E0)) / np.abs(E0)
        assert max_deviation < 0.01  # 1% tolerance


class TestEdgeCases:
    """Test edge cases and potential issues."""
    
    def test_very_narrow_wave_packet(self):
        """Test with a very narrow wave packet."""
        x = np.linspace(-10, 10, 512)
        psi0 = GaussianWavePacket(x, x0=0, k0=0, sigma=0.1)
        V = ZeroPotential(x)
        
        solver = TDSESolver(x, psi0, V, dt=0.001)
        
        assert np.allclose(solver.get_norm(), 1.0)
    
    def test_high_momentum(self):
        """Test with high momentum wave packet."""
        x = np.linspace(-20, 20, 1024)
        psi0 = GaussianWavePacket(x, x0=-10, k0=10, sigma=1)
        V = ZeroPotential(x)
        
        solver = TDSESolver(x, psi0, V, dt=0.001)
        
        # Should still be normalized
        assert np.allclose(solver.get_norm(), 1.0, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
