"""
Time-Dependent Schrödinger Equation Solver Package

A comprehensive package for solving the 1D time-dependent Schrödinger equation
using the split-operator method.
"""

from .solver import TDSESolver
from .wavefunction import GaussianWavePacket, normalize_wavefunction

__version__ = "0.1.0"
__all__ = ["TDSESolver", "GaussianWavePacket", "normalize_wavefunction"]
