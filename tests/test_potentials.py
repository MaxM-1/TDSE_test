"""
Unit tests for potential energy functions.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tdse_solver.potentials import (
    HarmonicOscillator, SquareBarrier, SquareWell,
    GaussianBarrier, ZeroPotential, StepPotential
)


class TestPotentials:
    """Test potential energy functions."""
    
    def test_harmonic_oscillator(self):
        """Test harmonic oscillator potential."""
        x = np.linspace(-10, 10, 100)
        omega = 1.0
        V = HarmonicOscillator(x, omega=omega)
        
        # Should be minimum at center
        assert V[len(x)//2] == np.min(V)
        
        # Should be symmetric
        assert np.allclose(V[:len(x)//2], V[len(x)//2:][::-1], atol=1e-10)
    
    def test_square_barrier(self):
        """Test square barrier potential."""
        x = np.linspace(-10, 10, 201)
        height = 2.0
        width = 4.0
        
        V = SquareBarrier(x, height=height, width=width, center=0)
        
        # Check barrier height in center region
        center_mask = np.abs(x) <= width / 2
        assert np.all(V[center_mask] == height)
        
        # Check zero outside barrier
        outside_mask = np.abs(x) > width / 2
        assert np.all(V[outside_mask] == 0)
    
    def test_square_well(self):
        """Test square well potential."""
        x = np.linspace(-10, 10, 201)
        depth = -3.0
        width = 6.0
        
        V = SquareWell(x, depth=depth, width=width, center=0)
        
        # Check well depth in center region
        center_mask = np.abs(x) <= width / 2
        assert np.all(V[center_mask] == depth)
        
        # Check zero outside well
        outside_mask = np.abs(x) > width / 2
        assert np.all(V[outside_mask] == 0)
    
    def test_gaussian_barrier(self):
        """Test Gaussian barrier potential."""
        x = np.linspace(-10, 10, 100)
        height = 2.0
        
        V = GaussianBarrier(x, height=height, width=1.0, center=0)
        
        # Should be maximum at center
        assert V[len(x)//2] == np.max(V)
        
        # Should be symmetric
        assert np.allclose(V[:len(x)//2], V[len(x)//2:][::-1], atol=1e-10)
        
        # Maximum should be close to height
        assert np.allclose(np.max(V), height, atol=1e-6)
    
    def test_zero_potential(self):
        """Test zero potential."""
        x = np.linspace(-10, 10, 100)
        V = ZeroPotential(x)
        
        assert np.all(V == 0)
    
    def test_step_potential(self):
        """Test step potential."""
        x = np.linspace(-10, 10, 201)
        height = 1.5
        position = 0.0
        
        V = StepPotential(x, height=height, position=position)
        
        # Check values on left and right
        assert np.all(V[x < position] == 0)
        assert np.all(V[x >= position] == height)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
