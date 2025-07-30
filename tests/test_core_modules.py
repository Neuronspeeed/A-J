"""
Unit tests for core modules.

Tests the new core/ refactor with proper mocking and isolation.
"""

import pytest
from unittest.mock import Mock, patch
from core.utils import compare_decimal_strings, extract_numerical_answer
from core.llm_providers import CircuitBreaker, MockLLMProvider
from core.data_models import ProviderConfig


class TestCompareDecimalStrings:
    """Test the core accuracy measurement function."""
    
    def test_exact_match(self):
        assert compare_decimal_strings("123.456", "123.456") == 6
    
    def test_partial_match(self):
        assert compare_decimal_strings("123.456", "123.999") == 3
    
    def test_no_match(self):
        assert compare_decimal_strings("123.456", "999.456") == 0
    
    def test_different_lengths(self):
        assert compare_decimal_strings("123.456", "123") == 3
        assert compare_decimal_strings("123", "123.456") == 3
    
    def test_empty_inputs(self):
        assert compare_decimal_strings("", "123") == 0
        assert compare_decimal_strings("123", "") == 0
        assert compare_decimal_strings("", "") == 0
    
    def test_none_inputs(self):
        assert compare_decimal_strings(None, "123") == 0
        assert compare_decimal_strings("123", None) == 0


class TestExtractNumericalAnswer:
    """Test numerical answer extraction patterns."""
    
    def test_therefore_pattern(self):
        text = "Therefore, $5,542.86 was invested at 6.7%."
        assert extract_numerical_answer(text) == "5542.86"
    
    def test_hours_pattern(self):
        text = "The faster train will overtake 8.78 hours after it departs."
        assert extract_numerical_answer(text) == "8.78"
    
    def test_no_pattern_match(self):
        text = "This text has no extractable answer."
        assert extract_numerical_answer(text) is None
    
    def test_comma_removal(self):
        text = "Therefore, $1,234,567.89 was the result."
        assert extract_numerical_answer(text) == "1234567.89"


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.can_execute() is True
        assert cb.state == "CLOSED"
    
    def test_failure_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record failures below threshold
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is True
        assert cb.state == "CLOSED"
        
        # Hit threshold - should open
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() is False
    
    def test_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"


class TestMockLLMProvider:
    """Test mock provider for testing."""
    
    @pytest.mark.asyncio
    async def test_mock_provider_basic(self):
        provider = MockLLMProvider("Test response")
        response = await provider.get_completion("System", "User", 30.0)

        assert response == "Test response"
        assert provider.call_count == 1
        assert provider.last_system_message == "System"
        assert provider.last_user_message == "User"
    
    @pytest.mark.asyncio
    async def test_mock_provider_multiple_calls(self):
        provider = MockLLMProvider("Mock response")

        await provider.get_completion("Sys1", "User1", 30.0)
        await provider.get_completion("Sys2", "User2", 30.0)

        assert provider.call_count == 2
        assert provider.last_system_message == "Sys2"
        assert provider.last_user_message == "User2"


class TestProviderConfig:
    """Test provider configuration validation."""
    
    def test_valid_config(self):
        config = ProviderConfig(
            provider_type="openai",
            model_name="gpt-4o",
            api_key_env_var="OPENAI_API_KEY",
            max_tokens=1000,
            temperature=0.7
        )
        
        assert config.provider_type == "openai"
        assert config.model_name == "gpt-4o"
        assert config.max_tokens == 1000
    
    def test_anthropic_config(self):
        config = ProviderConfig(
            provider_type="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            api_key_env_var="ANTHROPIC_API_KEY",
            max_tokens=1000,
            temperature=0.7
        )
        
        assert config.provider_type == "anthropic"
        assert config.api_key_env_var == "ANTHROPIC_API_KEY"
