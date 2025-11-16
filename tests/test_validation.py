"""
Tests for validation utilities.
"""

import pytest
import torch
from nanochat.validation import (
    ValidationError, validate_positive_int, validate_non_negative_float,
    validate_temperature, validate_tokens, validate_tensor_shape,
    validate_config, validate_vocab_size, validate_model_config
)
from nanochat.gpt import GPTConfig


class TestValidatePositiveInt:
    """Test validate_positive_int function."""

    def test_valid_positive_int(self):
        """Test with valid positive integer."""
        assert validate_positive_int(5, "test") == 5
        assert validate_positive_int(100, "test") == 100

    def test_invalid_zero(self):
        """Test that zero is rejected."""
        with pytest.raises(ValidationError):
            validate_positive_int(0, "test")

    def test_invalid_negative(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            validate_positive_int(-5, "test")

    def test_invalid_type(self):
        """Test that non-integers are rejected."""
        with pytest.raises(ValidationError):
            validate_positive_int(5.5, "test")
        with pytest.raises(ValidationError):
            validate_positive_int("5", "test")


class TestValidateNonNegativeFloat:
    """Test validate_non_negative_float function."""

    def test_valid_positive_float(self):
        """Test with valid positive float."""
        assert validate_non_negative_float(5.5, "test") == 5.5
        assert validate_non_negative_float(0.0, "test") == 0.0

    def test_valid_int(self):
        """Test that integers are converted to float."""
        assert validate_non_negative_float(5, "test") == 5.0

    def test_invalid_negative(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            validate_non_negative_float(-1.5, "test")

    def test_invalid_type(self):
        """Test that non-numeric types are rejected."""
        with pytest.raises(ValidationError):
            validate_non_negative_float("5.5", "test")


class TestValidateTemperature:
    """Test validate_temperature function."""

    def test_valid_temperatures(self):
        """Test valid temperature values."""
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.5) == 2.5

    def test_invalid_negative(self):
        """Test that negative temperature is rejected."""
        with pytest.raises(ValidationError):
            validate_temperature(-0.5)

    def test_invalid_type(self):
        """Test that non-numeric types are rejected."""
        with pytest.raises(ValidationError):
            validate_temperature("1.0")


class TestValidateTokens:
    """Test validate_tokens function."""

    def test_valid_tokens(self):
        """Test with valid token list."""
        tokens = [1, 2, 3, 4, 5]
        assert validate_tokens(tokens) == tokens

    def test_empty_list(self):
        """Test that empty list is rejected."""
        with pytest.raises(ValidationError):
            validate_tokens([])

    def test_non_list(self):
        """Test that non-list is rejected."""
        with pytest.raises(ValidationError):
            validate_tokens("not a list")
        with pytest.raises(ValidationError):
            validate_tokens(123)

    def test_non_integer_tokens(self):
        """Test that non-integer tokens are rejected."""
        with pytest.raises(ValidationError):
            validate_tokens([1, 2, 3.5, 4])
        with pytest.raises(ValidationError):
            validate_tokens([1, 2, "3", 4])

    def test_negative_tokens(self):
        """Test that negative tokens are rejected."""
        with pytest.raises(ValidationError):
            validate_tokens([1, 2, -1, 4])

    def test_max_length(self):
        """Test maximum length constraint."""
        tokens = [1, 2, 3, 4, 5]
        assert validate_tokens(tokens, max_length=10) == tokens
        with pytest.raises(ValidationError):
            validate_tokens(tokens, max_length=3)


class TestValidateTensorShape:
    """Test validate_tensor_shape function."""

    def test_valid_shape(self):
        """Test with matching shape."""
        tensor = torch.randn(2, 3, 4)
        result = validate_tensor_shape(tensor, (2, 3, 4), "test")
        assert torch.equal(result, tensor)

    def test_wildcard_dimensions(self):
        """Test with None wildcards in expected shape."""
        tensor = torch.randn(2, 3, 4)
        result = validate_tensor_shape(tensor, (None, 3, None), "test")
        assert torch.equal(result, tensor)

    def test_invalid_shape(self):
        """Test with mismatched shape."""
        tensor = torch.randn(2, 3, 4)
        with pytest.raises(ValidationError):
            validate_tensor_shape(tensor, (2, 4, 4), "test")

    def test_wrong_ndim(self):
        """Test with wrong number of dimensions."""
        tensor = torch.randn(2, 3)
        with pytest.raises(ValidationError):
            validate_tensor_shape(tensor, (2, 3, 4), "test")

    def test_allow_batch(self):
        """Test with allow_batch option."""
        tensor = torch.randn(8, 2, 3, 4)
        result = validate_tensor_shape(tensor, (2, 3, 4), "test", allow_batch=True)
        assert torch.equal(result, tensor)

    def test_non_tensor(self):
        """Test that non-tensor is rejected."""
        with pytest.raises(ValidationError):
            validate_tensor_shape([1, 2, 3], (3,), "test")


class TestValidateConfig:
    """Test validate_config function."""

    def test_valid_config(self):
        """Test with valid config object."""
        class Config:
            attr1 = 1
            attr2 = 2
            attr3 = 3

        config = Config()
        result = validate_config(config, ['attr1', 'attr2', 'attr3'])
        assert result is config

    def test_missing_attribute(self):
        """Test with missing required attribute."""
        class Config:
            attr1 = 1
            attr2 = 2

        config = Config()
        with pytest.raises(ValidationError):
            validate_config(config, ['attr1', 'attr2', 'attr3'])


class TestValidateVocabSize:
    """Test validate_vocab_size function."""

    def test_valid_vocab_size(self):
        """Test with valid vocabulary size."""
        assert validate_vocab_size(1000) == 1000
        assert validate_vocab_size(50000) == 50000

    def test_minimum_size(self):
        """Test minimum vocabulary size constraint."""
        with pytest.raises(ValidationError):
            validate_vocab_size(100)  # Less than default 256
        assert validate_vocab_size(100, min_size=50) == 100

    def test_invalid_type(self):
        """Test that non-integer is rejected."""
        with pytest.raises(ValidationError):
            validate_vocab_size(1000.5)


class TestValidateModelConfig:
    """Test validate_model_config function."""

    def test_valid_config(self):
        """Test with valid GPT config."""
        config = GPTConfig(
            sequence_len=1024,
            vocab_size=50000,
            n_layer=12,
            n_head=6,
            n_kv_head=2,
            n_embd=768
        )
        result = validate_model_config(config)
        assert result is config

    def test_invalid_n_embd_n_head_relationship(self):
        """Test that n_embd must be divisible by n_head."""
        config = GPTConfig(n_embd=768, n_head=7)
        with pytest.raises(ValidationError, match="divisible by n_head"):
            validate_model_config(config)

    def test_invalid_n_head_n_kv_head_relationship(self):
        """Test that n_head must be divisible by n_kv_head."""
        config = GPTConfig(n_head=6, n_kv_head=4)
        with pytest.raises(ValidationError, match="divisible by n_kv_head"):
            validate_model_config(config)

    def test_n_kv_head_greater_than_n_head(self):
        """Test that n_kv_head cannot exceed n_head."""
        config = GPTConfig(n_head=4, n_kv_head=6)
        # This will fail divisibility check first (4 % 6 != 0)
        with pytest.raises(ValidationError, match="divisible by n_kv_head"):
            validate_model_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
