"""
LLM Provider interfaces and implementations.

This module implements the Interface Segregation and Dependency Inversion principles
by defining a clear Protocol for LLM interactions and concrete implementations.

Key benefits:
- Easy to add new providers (Open/Closed Principle)
- Core engine doesn't depend on specific LLM APIs (Dependency Inversion)
- Each provider handles only its own logic (Single Responsibility)
- Testable through interface mocking
"""

import asyncio
import functools
import os
from typing import Protocol, Optional
from abc import ABC, abstractmethod

from openai import OpenAI
from anthropic import Anthropic

from .data_models import ProviderConfig


class LLMProvider(Protocol):
    """
    Protocol defining the interface for any LLM provider.
    
    This ensures all providers have a consistent contract, enabling
    dependency injection and easy testing.
    """
    
    async def get_completion(
        self, 
        system_message: str, 
        user_message: str,
        timeout_seconds: float = 30.0
    ) -> str:
        """
        Get a completion from the LLM.
        
        Args:
            system_message: System prompt for the model
            user_message: User prompt for the model
            timeout_seconds: Maximum time to wait for response
            
        Returns:
            The model's response as a string
            
        Raises:
            LLMProviderError: If the request fails
            asyncio.TimeoutError: If the request times out
        """
        ...


class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""
    
    def __init__(self, message: str, provider: str, model: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.model = model
        self.original_error = original_error
        super().__init__(f"{provider} ({model}): {message}")


def async_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for exponential backoff retry logic.
    
    Implements fail-fast error handling with graceful degradation.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        return wrapper
    return decorator


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implements common functionality like error handling and configuration.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    async def _make_request(self, system_message: str, user_message: str) -> str:
        """Make the actual API request."""
        pass
    
    @async_exponential_backoff(max_retries=3, base_delay=1.0)
    async def get_completion(
        self, 
        system_message: str, 
        user_message: str,
        timeout_seconds: float = 30.0
    ) -> str:
        """
        Get completion with retry logic and error handling.
        """
        try:
            # Apply timeout
            return await asyncio.wait_for(
                self._make_request(system_message, user_message),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise LLMProviderError(
                f"Request timed out after {timeout_seconds} seconds",
                self.config.provider_type,
                self.config.model_name
            )
        except Exception as e:
            raise LLMProviderError(
                f"Request failed: {str(e)}",
                self.config.provider_type,
                self.config.model_name,
                original_error=e
            )


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation.
    
    Handles OpenAI-specific API calls and error handling.
    """
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client with API key from environment or .env file."""
        api_key = os.getenv(self.config.api_key_env_var)

        # Try to read from .env file if not in environment
        if not api_key:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(f'{self.config.api_key_env_var}='):
                            api_key = line.split('=', 1)[1].strip()
                            break
            except FileNotFoundError:
                pass

        if not api_key:
            raise LLMProviderError(
                f"API key not found in environment variable: {self.config.api_key_env_var}",
                self.config.provider_type,
                self.config.model_name
            )

        self._client = OpenAI(api_key=api_key)
    
    async def _make_request(self, system_message: str, user_message: str) -> str:
        """Make OpenAI API request."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMProviderError(
                f"OpenAI API error: {str(e)}",
                self.config.provider_type,
                self.config.model_name,
                original_error=e
            )


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic provider implementation.
    
    Handles Anthropic-specific API calls and error handling.
    """
    
    def _initialize_client(self) -> None:
        """Initialize Anthropic client with API key from environment or .env file."""
        api_key = os.getenv(self.config.api_key_env_var)

        # Try to read from .env file if not in environment
        if not api_key:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(f'{self.config.api_key_env_var}='):
                            api_key = line.split('=', 1)[1].strip()
                            break
                        # Also try CLAUDE_API_KEY as alternative
                        if self.config.api_key_env_var == "ANTHROPIC_API_KEY" and line.startswith('CLAUDE_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break
            except FileNotFoundError:
                pass

        if not api_key:
            raise LLMProviderError(
                f"API key not found in environment variable: {self.config.api_key_env_var}",
                self.config.provider_type,
                self.config.model_name
            )

        self._client = Anthropic(api_key=api_key)
    
    async def _make_request(self, system_message: str, user_message: str) -> str:
        """Make Anthropic API request."""
        try:
            response = await asyncio.to_thread(
                self._client.messages.create,
                model=self.config.model_name,
                system=system_message,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.content[0].text
        except Exception as e:
            raise LLMProviderError(
                f"Anthropic API error: {str(e)}",
                self.config.provider_type,
                self.config.model_name,
                original_error=e
            )


class MockLLMProvider:
    """
    Mock provider for testing.
    
    Returns predictable responses for unit tests.
    """
    
    def __init__(self, mock_response: str = "Mock response"):
        self.mock_response = mock_response
        self.call_count = 0
        self.last_system_message = None
        self.last_user_message = None
    
    async def get_completion(
        self, 
        system_message: str, 
        user_message: str,
        timeout_seconds: float = 30.0
    ) -> str:
        """Return mock response and track calls for testing."""
        self.call_count += 1
        self.last_system_message = system_message
        self.last_user_message = user_message
        
        # Simulate some async work
        await asyncio.sleep(0.01)
        
        return self.mock_response


def create_provider(config: ProviderConfig) -> LLMProvider:
    """
    Factory function to create providers based on configuration.
    
    This implements the Factory pattern and supports dependency injection.
    """
    if config.provider_type == "openai":
        return OpenAIProvider(config)
    elif config.provider_type == "anthropic":
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {config.provider_type}")
