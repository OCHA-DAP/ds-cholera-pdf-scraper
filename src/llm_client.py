"""
LLM Client abstraction for OpenAI and OpenRouter support.
Provides a unified interface for different model providers.
"""

import json
import os
from typing import Any, Dict, Tuple

import requests
from openai import OpenAI

# Try relative import first, fall back to absolute
try:
    from .config import Config
except ImportError:
    from config import Config


class LLMClient:
    """
    Unified LLM client that supports both OpenAI direct and OpenRouter.
    """

    def __init__(self, custom_config: Dict[str, Any] = None):
        """
        Initialize LLM client with configuration.

        Args:
            custom_config: Optional custom configuration override
        """
        # Get configuration
        if custom_config:
            self.config = custom_config
        else:
            # Use intelligent configuration for default model
            self.config = Config.get_llm_client_config_for_model()

        # Store model info for logging (needed before client initialization)
        self.model_name = self.config["model"]
        self.provider = self.config["provider"]

        # Initialize OpenAI client based on provider
        if self.config["provider"] == "openrouter":
            self.client = OpenAI(
                base_url=self.config["base_url"],
                api_key=self.config["api_key"],
                max_retries=3,  # Retry up to 3 times for transient network errors
            )
        else:
            # Check if this is a reasoning model that needs longer timeout
            is_reasoning_model = (
                "gpt-5" in self.model_name.lower() or "grok" in self.model_name.lower()
            )
            timeout_seconds = 1500 if is_reasoning_model else 60  # 10 min vs 1 min

            self.client = OpenAI(
                api_key=self.config["api_key"],
                timeout=timeout_seconds,
                max_retries=3,  # Retry up to 3 times for transient network errors
            )

    def create_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a chat completion using the configured provider.

        Args:
            system_prompt: System message content
            user_prompt: User message content
            max_tokens: Maximum tokens to generate
            temperature: Temperature override
            **kwargs: Additional model parameters

        Returns:
            Tuple of (response_content, metadata)
        """
        # Use config temperature if not overridden
        if temperature is None:
            temperature = self.config["temperature"]

        # GPT-5 only supports default temperature (1), ignore custom temperature settings
        if "gpt-5" in self.model_name.lower():
            temperature = None  # Use default temperature for GPT-5

        # Grok-4 is also a reasoning model and may need special handling
        if "grok" in self.model_name.lower():
            # Use much higher token limits for reasoning models (like GPT-5)
            max_tokens = max(
                max_tokens, 16384
            )  # Ensure minimum 16K tokens for reasoning

        # Check if this is GPT-5 - temporarily use Chat Completions until org verification
        # GPT-5 works best with Responses API, but needs organizational verification
        if "gpt-5" in self.model_name.lower():
            # Use Chat Completions with GPT-5 specific parameters
            return self._create_gpt5_chat_completion(
                system_prompt, user_prompt, max_tokens, **kwargs
            )
        else:
            return self._create_chat_completion(
                system_prompt, user_prompt, max_tokens, temperature, **kwargs
            )

    def _create_gpt5_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a response using GPT-5's Responses API.
        """
        # Combine system and user prompts for GPT-5
        combined_input = f"{system_prompt}\n\n{user_prompt}"

        # Prepare GPT-5 specific parameters
        response_params = {
            "model": self.model_name,
            "input": combined_input,
            "max_completion_tokens": max_tokens,
            "reasoning": {
                "effort": "minimal"  # Use minimal reasoning for faster responses
            },
            "text": {
                "verbosity": "low"  # Use low verbosity for faster, more concise responses
            },
            **kwargs,
        }

        # Make the API call using responses endpoint
        response = self.client.responses.create(**response_params)

        # Extract response content
        response_content = response.output_text

        # Prepare metadata for logging
        metadata = {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_parameters": {
                "max_completion_tokens": max_tokens,
                "reasoning_effort": "medium",
                "text_verbosity": "medium",
                **kwargs,
            },
            "usage": (
                response.usage.model_dump()
                if hasattr(response, "usage") and response.usage
                else None
            ),
        }

        return response_content, metadata

    def _create_gpt5_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a chat completion for GPT-5 using Chat Completions API with correct parameters.
        """
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prepare GPT-5 specific parameters for Chat Completions (optimized for speed)
        model_params = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max(
                max_tokens, 16384
            ),  # Ensure sufficient tokens for GPT-5 reasoning
            # No temperature for GPT-5 - uses default for faster processing
            **kwargs,
        }

        # Add extra headers for OpenRouter (if using OpenRouter)
        extra_kwargs = {}
        if self.provider == "openrouter":
            extra_kwargs["extra_headers"] = self.config["extra_headers"]

        # Make the API call
        response = self.client.chat.completions.create(**model_params, **extra_kwargs)

        # Extract response content
        response_content = response.choices[0].message.content

        # Prepare metadata for logging
        metadata = {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_parameters": {
                "max_completion_tokens": max(max_tokens, 16384),
                **kwargs,
            },
            "usage": response.usage.model_dump() if response.usage else None,
        }

        return response_content, metadata

    def _create_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a chat completion using traditional Chat Completions API.
        """
        # Check if this is a Llama model - use direct HTTP to avoid OpenAI client parsing issues
        if "llama" in self.model_name.lower():
            return self._create_direct_http_completion(
                system_prompt, user_prompt, max_tokens, temperature, **kwargs
            )

        # Use OpenAI client for other models
        return self._create_openai_client_completion(
            system_prompt, user_prompt, max_tokens, temperature, **kwargs
        )

    def _create_direct_http_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create completion using direct HTTP requests to bypass OpenAI client JSON parsing issues.
        Used specifically for Llama models which return large responses that break OpenAI client.
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for direct HTTP requests"
            )

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/OCHA-DAP/ds-cholera-pdf-scraper",
            "X-Title": "OCHA Cholera PDF Scraper",
        }

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }

        # Add temperature only if specified
        if temperature is not None:
            data["temperature"] = temperature
        try:
            response = requests.post(url, headers=headers, json=data, timeout=180)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Request to OpenRouter timed out after 180 seconds. Please try again later or check your network connection."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request to OpenRouter failed: {str(e)}")

        response_json = response.json()
        response_content = response_json["choices"][0]["message"]["content"]

        # Prepare metadata for logging (matching OpenAI client format)
        model_parameters = {
            "max_tokens": max_tokens,
            **kwargs,
        }
        if temperature is not None:
            model_parameters["temperature"] = temperature

        metadata = {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_parameters": model_parameters,
            "usage": response_json.get("usage", None),
        }

        return response_content, metadata

    def _create_openai_client_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a chat completion using OpenAI client library for compatible models.
        """
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prepare model parameters
        model_params = {
            "model": self.model_name,
            "messages": messages,
            **kwargs,
        }

        # Add temperature only if specified
        if temperature is not None:
            model_params["temperature"] = temperature

        # Handle different token limit parameters for different models
        model_params["max_tokens"] = max_tokens

        # Add extra headers for OpenRouter
        extra_kwargs = {}
        if self.provider == "openrouter":
            extra_kwargs["extra_headers"] = self.config["extra_headers"]

        # Make the API call
        response = self.client.chat.completions.create(**model_params, **extra_kwargs)

        # Extract response content
        response_content = response.choices[0].message.content

        # Prepare metadata for logging
        model_parameters = {
            "max_tokens": max_tokens,
            **kwargs,
        }
        if temperature is not None:
            model_parameters["temperature"] = temperature

        metadata = {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_parameters": model_parameters,
            "usage": response.usage.model_dump() if response.usage else None,
        }

        return response_content, metadata

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns:
            Dictionary with model info
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.config["temperature"],
            "base_url": self.config.get("base_url"),
        }

    @staticmethod
    def create_client_for_model(model_name: str, provider: str = None) -> "LLMClient":
        """
        Create a client for a specific model with intelligent provider selection.
        OpenAI models use organizational OpenAI API, others use OpenRouter.

        Args:
            model_name: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            provider: Provider to use (None=auto-detect, "openrouter", "openai")

        Returns:
            Configured LLMClient instance
        """
        if provider is None:
            # Use intelligent configuration that auto-selects provider
            custom_config = Config.get_llm_client_config_for_model(model_name)
        elif provider == "openrouter":
            custom_config = {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": Config.OPENROUTER_API_KEY,
                "model": model_name,
                "temperature": Config.MODEL_TEMPERATURE,
                "extra_headers": {
                    "HTTP-Referer": Config.OPENROUTER_SITE_URL,
                    "X-Title": Config.OPENROUTER_SITE_NAME,
                },
            }
        else:  # provider == "openai"
            custom_config = {
                "provider": "openai",
                "api_key": Config.EFFECTIVE_OPENAI_KEY,
                "model": model_name,
                "temperature": Config.MODEL_TEMPERATURE,
            }

        return LLMClient(custom_config)


# Common model configurations for easy access
POPULAR_MODELS = {
    # OpenAI via OpenRouter
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-5": "openai/gpt-5",
    "gpt-5-mini": "openai/gpt-5-mini",
    # Anthropic Claude
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-4-sonnet": "anthropic/claude-sonnet-4",
    "claude-4-opus": "anthropic/claude-opus-4",
    "claude-4.1-opus": "anthropic/claude-opus-4.1",
    # Google Gemini (updated with correct IDs)
    "gemini-pro": "google/gemini-pro-1.5",
    "gemini-flash": "google/gemini-flash-1.5",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    # Meta Llama
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    # Mistral
    "mistral-large": "mistralai/mistral-large",
    "mistral-medium": "mistralai/mistral-medium-3.1",
    # Specialized models
    "deepseek-chat": "deepseek/deepseek-chat",
    "qwen-turbo": "qwen/qwen-turbo",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    # X.AI Grok
    "grok-4": "x-ai/grok-4",
}


def get_model_identifier(model_nickname: str) -> str:
    """
    Get full model identifier from nickname.

    Args:
        model_nickname: Short name like 'claude-3.5-sonnet'

    Returns:
        Full model identifier for OpenRouter
    """
    return POPULAR_MODELS.get(model_nickname, model_nickname)


# Example usage
if __name__ == "__main__":
    # Test basic client
    client = LLMClient()
    print(f"Client configured for: {client.get_model_info()}")

    # Test specific model client
    claude_client = LLMClient.create_client_for_model("anthropic/claude-3.5-sonnet")
    print(f"Claude client: {claude_client.get_model_info()}")

    # Show available models
    print("\nPopular models available:")
    for nickname, full_id in POPULAR_MODELS.items():
        print(f"  {nickname}: {full_id}")
