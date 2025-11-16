"""
Validation utilities for nanochat.
Provides functions to validate inputs and configurations.
"""

from typing import Any, Optional
import torch


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_positive_int(value: Any, name: str) -> int:
    """Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)

    Returns:
        The validated integer value

    Raises:
        ValidationError: If value is not a positive integer
    """
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value)}")
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative_float(value: Any, name: str) -> float:
    """Validate that a value is a non-negative float.

    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)

    Returns:
        The validated float value

    Raises:
        ValidationError: If value is not a non-negative float
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")
    return float(value)


def validate_temperature(temperature: float) -> float:
    """Validate temperature parameter for sampling.

    Args:
        temperature: Temperature value to validate

    Returns:
        The validated temperature value

    Raises:
        ValidationError: If temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError(f"temperature must be a number, got {type(temperature)}")
    if temperature < 0:
        raise ValidationError(f"temperature must be non-negative, got {temperature}")
    return float(temperature)


def validate_tokens(tokens: Any, max_length: Optional[int] = None) -> list:
    """Validate a list of token IDs.

    Args:
        tokens: Token list to validate
        max_length: Optional maximum length constraint

    Returns:
        The validated token list

    Raises:
        ValidationError: If tokens are invalid
    """
    if not isinstance(tokens, list):
        raise ValidationError(f"tokens must be a list, got {type(tokens)}")
    if len(tokens) == 0:
        raise ValidationError("tokens list cannot be empty")
    if not all(isinstance(t, int) for t in tokens):
        raise ValidationError("all tokens must be integers")
    if not all(t >= 0 for t in tokens):
        raise ValidationError("all tokens must be non-negative")
    if max_length is not None and len(tokens) > max_length:
        raise ValidationError(f"tokens length {len(tokens)} exceeds maximum {max_length}")
    return tokens


def validate_tensor_shape(tensor: torch.Tensor, expected_shape: tuple,
                         name: str, allow_batch: bool = False) -> torch.Tensor:
    """Validate that a tensor has the expected shape.

    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (use None for any dimension)
        name: Name of the tensor (for error messages)
        allow_batch: If True, allows an extra batch dimension at the start

    Returns:
        The validated tensor

    Raises:
        ValidationError: If shape doesn't match
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")

    actual_shape = tensor.shape
    if allow_batch and len(actual_shape) == len(expected_shape) + 1:
        # Skip batch dimension
        actual_shape = actual_shape[1:]

    if len(actual_shape) != len(expected_shape):
        raise ValidationError(
            f"{name} has wrong number of dimensions: "
            f"expected {len(expected_shape)}, got {len(actual_shape)}"
        )

    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValidationError(
                f"{name} has wrong shape at dimension {i}: "
                f"expected {expected}, got {actual}"
            )

    return tensor


def validate_config(config: Any, required_attrs: list) -> Any:
    """Validate that a config object has all required attributes.

    Args:
        config: Configuration object to validate
        required_attrs: List of required attribute names

    Returns:
        The validated config object

    Raises:
        ValidationError: If config is missing required attributes
    """
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValidationError(f"config is missing required attribute: {attr}")
    return config


def validate_vocab_size(vocab_size: int, min_size: int = 256) -> int:
    """Validate vocabulary size.

    Args:
        vocab_size: Vocabulary size to validate
        min_size: Minimum acceptable vocabulary size

    Returns:
        The validated vocabulary size

    Raises:
        ValidationError: If vocab_size is invalid
    """
    if not isinstance(vocab_size, int):
        raise ValidationError(f"vocab_size must be an integer, got {type(vocab_size)}")
    if vocab_size < min_size:
        raise ValidationError(f"vocab_size must be at least {min_size}, got {vocab_size}")
    return vocab_size


def validate_model_config(config: Any) -> Any:
    """Validate a GPT model configuration.

    Args:
        config: GPTConfig instance to validate

    Returns:
        The validated config

    Raises:
        ValidationError: If config is invalid
    """
    # Check required attributes
    required = ['sequence_len', 'vocab_size', 'n_layer', 'n_head', 'n_kv_head', 'n_embd']
    validate_config(config, required)

    # Validate individual fields
    validate_positive_int(config.sequence_len, 'sequence_len')
    validate_vocab_size(config.vocab_size)
    validate_positive_int(config.n_layer, 'n_layer')
    validate_positive_int(config.n_head, 'n_head')
    validate_positive_int(config.n_kv_head, 'n_kv_head')
    validate_positive_int(config.n_embd, 'n_embd')

    # Validate relationships between fields
    if config.n_embd % config.n_head != 0:
        raise ValidationError(
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

    if config.n_head % config.n_kv_head != 0:
        raise ValidationError(
            f"n_head ({config.n_head}) must be divisible by n_kv_head ({config.n_kv_head})"
        )

    if config.n_kv_head > config.n_head:
        raise ValidationError(
            f"n_kv_head ({config.n_kv_head}) cannot be greater than n_head ({config.n_head})"
        )

    return config
