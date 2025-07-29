"""
Configuration package for the thinking transplant experiment.

This package contains all experimental configurations, preserving
100% of your friend's experimental logic in a maintainable format.
"""

from .experiments import (
    PHASE1_CONFIG,
    PHASE2_CONFIG,
    MATH_PROBLEMS,
    PROVIDER_CONFIGS,
    CONDITION_PROMPTS,
    get_provider_config,
    get_prompt_template
)

__all__ = [
    "PHASE1_CONFIG",
    "PHASE2_CONFIG", 
    "MATH_PROBLEMS",
    "PROVIDER_CONFIGS",
    "CONDITION_PROMPTS",
    "get_provider_config",
    "get_prompt_template"
]
