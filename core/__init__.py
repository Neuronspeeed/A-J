"""
Core package for the thinking transplant experiment.

This package contains the fundamental building blocks:
- Data models (Pydantic models for type safety)
- LLM providers (Interface-based design for different APIs)
- Persistence (Reading and writing experiment results)
"""

from .data_models import (
    MathProblem,
    TrialResult, 
    ExperimentConfig,
    ExperimentResults,
    ConditionType,
    ExperimentPhase,
    ProviderConfig
)

from .llm_providers import (
    LLMProvider,
    LLMProviderError,
    OpenAIProvider,
    AnthropicProvider,
    MockLLMProvider,
    create_provider
)

from .persistence import (
    ResultWriter,
    ResultReader,
    CsvResultWriter,
    CsvResultReader,
    JsonResultWriter,
    find_latest_results_file
)

from .utils import (
    compare_decimal_strings,
    extract_xml_answers,
    extract_random_numbers,
    extract_numerical_answer,
    validate_api_key,
    format_duration
)

from .data_manager import (
    DataManager,
    DataPaths
)

__all__ = [
    # Data models
    "MathProblem",
    "TrialResult", 
    "ExperimentConfig",
    "ExperimentResults",
    "ConditionType",
    "ExperimentPhase",
    "ProviderConfig",
    
    # LLM providers
    "LLMProvider",
    "LLMProviderError",
    "OpenAIProvider",
    "AnthropicProvider", 
    "MockLLMProvider",
    "create_provider",
    
    # Persistence
    "ResultWriter",
    "ResultReader",
    "CsvResultWriter",
    "CsvResultReader",
    "JsonResultWriter",
    "find_latest_results_file",

    # Utilities
    "compare_decimal_strings",
    "extract_xml_answers",
    "extract_random_numbers",
    "extract_numerical_answer",
    "validate_api_key",
    "format_duration",

    # Data Management
    "DataManager",
    "DataPaths"
]
