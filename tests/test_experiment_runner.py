"""
Unit tests for the ExperimentRunner.

We can test the core experimental logic without making any real API calls
or writing to disk.
"""

import pytest
import asyncio
from unittest.mock import Mock
from datetime import datetime

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from engine.experiment_runner import ExperimentRunner
from core.data_models import (
    MathProblem, ExperimentConfig, ConditionType, ExperimentPhase
)
from core.llm_providers import MockLLMProvider
from core.utils import (
    compare_decimal_strings, extract_xml_answers, extract_random_numbers,
    extract_numerical_answer
)


class MockResultWriter:
    """Mock result writer for testing."""
    
    def __init__(self):
        self.saved_trials = []
    
    def save_trial(self, trial):
        self.saved_trials.append(trial)
    
    def save_experiment(self, results):
        pass
    
    def finalize(self):
        return "mock_results.csv"


@pytest.fixture
def sample_problem():
    """Sample math problem for testing."""
    return MathProblem(
        id="test_problem",
        question="What is 2 + 2?",
        expected_answer="4"
    )


@pytest.fixture
def sample_config(sample_problem):
    """Sample experiment configuration."""
    return ExperimentConfig(
        name="Test Experiment",
        phase=ExperimentPhase.PHASE_1,
        description="Test configuration",
        conditions=[ConditionType.BASELINE],
        math_problems=[sample_problem],
        model_names=["test-model"],
        iterations_per_condition=1,  # Required field - no defaults
        max_retries=3,               # Required field - no defaults
        timeout_seconds=30.0,        # Required field - no defaults
        output_filename_template="test_{timestamp}.csv"
    )


@pytest.fixture
def mock_provider():
    """Mock LLM provider that returns predictable responses."""
    return MockLLMProvider(
        mock_response="<answer1>Test first answer</answer1><answer2>4</answer2>"
    )


@pytest.fixture
def mock_writer():
    """Mock result writer."""
    return MockResultWriter()


@pytest.fixture
def runner(mock_provider, mock_writer):
    """ExperimentRunner with mocked dependencies."""
    return ExperimentRunner(provider=mock_provider, writer=mock_writer)


@pytest.mark.asyncio
async def test_single_trial_success(runner, sample_problem):
    """Test that a single trial runs successfully and calculates accuracy."""
    trial = await runner._run_single_trial(
        problem=sample_problem,
        model_name="test-model",
        condition=ConditionType.BASELINE,
        iteration=0,
        config=ExperimentConfig(
            name="Test",
            phase=ExperimentPhase.PHASE_1,
            description="Test",
            conditions=[ConditionType.BASELINE],
            math_problems=[sample_problem],
            model_names=["test-model"],
            iterations_per_condition=1,  # Required field
            max_retries=3,               # Required field
            timeout_seconds=30.0,        # Required field
            output_filename_template="test.csv"
        )
    )
    
    # Verify trial data
    assert trial.model_name == "test-model"
    assert trial.condition == ConditionType.BASELINE
    assert trial.problem == sample_problem
    assert trial.error is None
    assert trial.math_answer == "4"
    assert trial.digits_correct == 1  # "4" matches "4" for 1 digit


def test_xml_parsing():
    """Test that XML answer parsing works correctly."""
    response = "<answer1>Random numbers: 123, 456, 789</answer1><answer2>42.5</answer2>"

    first_answer, math_answer = extract_xml_answers(response)

    assert first_answer == "Random numbers: 123, 456, 789"
    assert math_answer == "42.5"


def test_random_number_extraction():
    """Test extraction of random numbers from AI responses."""
    text = "Here are some random numbers: 1234567890, 9876543210, 5555555555"

    numbers = extract_random_numbers(text)

    assert numbers == [1234567890, 9876543210, 5555555555]


def test_accuracy_calculation():
    """Test the digits_correct accuracy metric."""
    # Perfect match
    assert compare_decimal_strings("123.456", "123.456") == 6

    # Partial match
    assert compare_decimal_strings("123.456", "123.999") == 3

    # No match
    assert compare_decimal_strings("123.456", "999.456") == 0

    # Different lengths
    assert compare_decimal_strings("123.456789", "123.45") == 5


@pytest.mark.asyncio
async def test_experiment_run_complete(runner, sample_config, mock_writer):
    """Test that a complete experiment runs and saves results."""
    results = await runner.run_experiment(sample_config)
    
    # Verify experiment completed
    assert results.total_trials == 1
    assert results.successful_trials == 1
    assert results.failed_trials == 0
    
    # Verify results were saved
    assert len(mock_writer.saved_trials) == 1
    
    # Verify accuracy calculation
    accuracy_by_condition = results.get_accuracy_by_condition()
    assert "baseline" in accuracy_by_condition


def test_prompt_building_baseline(runner, sample_problem):
    """Test prompt building for baseline condition."""
    system_msg, user_msg = runner._build_prompts(
        problem=sample_problem,
        condition=ConditionType.BASELINE,
        iteration=0
    )
    
    # Should contain the math question directly
    assert sample_problem.question in user_msg
    # Should not contain "Question 1:" for baseline
    assert "Question 1:" not in user_msg


def test_prompt_building_with_filler(runner, sample_problem):
    """Test prompt building for conditions with filler questions."""
    system_msg, user_msg = runner._build_prompts(
        problem=sample_problem,
        condition=ConditionType.MEMORIZED,
        iteration=0
    )

    # Should contain both questions
    assert "Question 1:" in user_msg
    assert "Question 2:" in user_msg
    assert "Sing Happy Birthday" in user_msg
    assert sample_problem.question in user_msg


def test_transplant_condition_without_numbers_fails(runner, sample_problem):
    """Test that transplant condition fails properly when no numbers are available."""
    # Should raise ValueError when no harvested numbers available
    with pytest.raises(ValueError, match="No harvested numbers available"):
        runner._build_prompts(
            problem=sample_problem,
            condition=ConditionType.WITH_TRANSPLANTED_NUMBERS,
            iteration=0
        )


@pytest.mark.asyncio
async def test_provider_error_handling(mock_writer, sample_config):
    """Test that provider errors are handled gracefully."""
    # Create a provider that always fails
    failing_provider = MockLLMProvider()
    
    # Make it raise an exception
    async def failing_completion(*args, **kwargs):
        raise Exception("API Error")
    
    failing_provider.get_completion = failing_completion
    
    runner = ExperimentRunner(provider=failing_provider, writer=mock_writer)
    
    results = await runner.run_experiment(sample_config)
    
    # Should have recorded the failure
    assert results.total_trials == 1
    assert results.successful_trials == 0
    assert results.failed_trials == 1
    
    # Error should be recorded in the trial
    trial = mock_writer.saved_trials[0]
    assert trial.error is not None
    assert "API Error" in trial.error


def test_numerical_answer_extraction():
    """Test extraction of numerical answers from various response formats."""
    # Test different answer formats
    test_cases = [
        ("The answer is 42.5", "42.5"),
        ("Final answer: 123.456", "123.456"),
        ("It equals 99.9 dollars", "99.9"),
        ("The result is approximately 7.8 hours", "7.8"),
        ("Random text with 15.25 at the end", "15.25"),
    ]

    for response, expected in test_cases:
        result = extract_numerical_answer(response)
        assert result == expected, f"Failed for: {response}"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
