"""
LLM Client abstraction for OpenAI and OpenRouter support.
Provides a unified interface for different model providers.
"""

import os
from typing import Any, Dict, Tuple

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

        # Initialize OpenAI client based on provider
        if self.config["provider"] == "openrouter":
            self.client = OpenAI(
                base_url=self.config["base_url"],
                api_key=self.config["api_key"],
            )
        else:
            self.client = OpenAI(
                api_key=self.config["api_key"],
            )

        # Store model info for logging
        self.model_name = self.config["model"]
        self.provider = self.config["provider"]

    def create_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 16384,
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

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prepare model parameters
        model_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        # Add extra headers for OpenRouter
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
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
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
