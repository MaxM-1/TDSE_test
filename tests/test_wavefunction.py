"""
Unit tests for wave function utilities.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tdse_solver.wavefunction import (
    GaussianWavePacket, normalize_wavefunction, PlaneWave,
    calculate_overlap, spreading_width, GroundStateHO
)


class TestWaveFunctions:
    """Test wave function creation."""
    
    def test_gaussian_wave_packet(self):
        """Test Gaussian wave packet creation."""
        x = np.linspace(-10, 10, 256)
        dx = x[1] - x[0]
        
        psi = GaussianWavePacket(x, x0=0, k0=1, sigma=2)
        
        # Check normalization
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        assert np.allclose(norm, 1.0, atol=1e-6)
        
        # Check that peak is near x0
        prob = np.abs(psi)**2
        x_max = x[np.argmax(prob)]
        assert np.abs(x_max - 0.0) < dx * 2
    
    def test_plane_wave(self):
        """Test plane wave creation."""
        x = np.linspace(0, 10, 256)
        k = 2.0
        
        psi = PlaneWave(x, k=k, normalize=False)
        
        # Check phase evolution
        phase = np.angle(psi)
        phase_diff = np.diff(phase)
        
        # Phase should increase linearly with x
        expected_diff = k * (x[1] - x[0])
        assert np.allclose(phase_diff, expected_diff, atol=0.01)
    
    def test_ground_state_ho(self):
        """Test harmonic oscillator ground state."""
        x = np.linspace(-10, 10, 256)
        dx = x[1] - x[0]
        
        omega = 1.0
        psi = GroundStateHO(x, omega=omega)
        
        # Check normalization
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        assert np.allclose(norm, 1.0, atol=1e-6)
        
        # Check that it's centered at origin
        prob = np.abs(psi)**2
        x_mean = np.sum(x * prob) * dx
        assert np.abs(x_mean) < 0.01
        
        # Ground state should be real and positive at center
        assert np.real(psi[len(x)//2]) > 0
        assert np.abs(np.imag(psi[len(x)//2])) < 1e-10


class TestWaveFunctionUtilities:
    """Test utility functions."""
    
    def test_normalize_wavefunction(self):
        """Test wave function normalization."""
        x = np.linspace(-10, 10, 256)
        dx = x[1] - x[0]
        
        # Create unnormalized wave function
        psi = np.exp(-x**2)
        
        psi_norm = normalize_wavefunction(psi, x)
        
        # Check normalization
        norm = np.sqrt(np.sum(np.abs(psi_norm)**2) * dx)
        assert np.allclose(norm, 1.0, atol=1e-6)
    
    def test_calculate_overlap(self):
        """Test overlap calculation."""
        x = np.linspace(-10, 10, 256)
        
        psi1 = GaussianWavePacket(x, x0=0, k0=0, sigma=1)
        psi2 = GaussianWavePacket(x, x0=0, k0=0, sigma=1)
        
        # Overlap with itself should be 1
        overlap = calculate_overlap(psi1, psi2, x)
        assert np.allclose(np.abs(overlap), 1.0, atol=1e-6)
        
        # Orthogonal states should have zero overlap
        psi3 = GaussianWavePacket(x, x0=10, k0=0, sigma=1)
        overlap2 = calculate_overlap(psi1, psi3, x)
        assert np.abs(overlap2) < 0.1
    
    def test_spreading_width(self):
        """Test wave packet width calculation."""
        x = np.linspace(-20, 20, 512)
        
        sigma = 2.0
        psi = GaussianWavePacket(x, x0=0, k0=0, sigma=sigma)
        
        width = spreading_width(psi, x)
        
        # Should be close to input sigma
        assert np.abs(width - sigma) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
