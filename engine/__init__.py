"""
Engine package for the thinking transplant experiment.

This package contains the core experimental logic that orchestrates
the entire process while remaining agnostic to specific implementations.
"""

from .experiment_runner import ExperimentRunner

__all__ = [
    "ExperimentRunner"
]
