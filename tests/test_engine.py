"""
Comprehensive tests for the inference Engine.
Tests KV cache, token sampling, generation, and tool use.
"""

import pytest
import torch
from nanochat.engine import (
    KVCache, Engine, sample_next_token,
    use_calculator, eval_with_timeout, RowState
)
from nanochat.gpt import GPT, GPTConfig


class TestCalculatorTool:
    """Test the calculator tool functionality."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert use_calculator("1 + 1") == 2
        assert use_calculator("10 - 5") == 5
        assert use_calculator("3 * 4") == 12
        assert use_calculator("15 / 3") == 5

    def test_complex_expressions(self):
        """Test more complex mathematical expressions."""
        assert use_calculator("(10 + 5) * 2") == 30
        assert use_calculator("100 / (5 + 5)") == 10
        assert use_calculator("2 + 3 * 4") == 14

    def test_decimal_numbers(self):
        """Test decimal arithmetic."""
        result = use_calculator("1.5 + 2.5")
        assert abs(result - 4.0) < 1e-6
        result = use_calculator("10.5 / 2")
        assert abs(result - 5.25) < 1e-6

    def test_comma_removal(self):
        """Test that commas are removed from numbers."""
        assert use_calculator("1,000 + 1,000") == 2000
        assert use_calculator("10,000 / 2") == 5000

    def test_invalid_characters(self):
        """Test that invalid characters return None."""
        assert use_calculator("1 + a") is None
        assert use_calculator("print(1)") is None
        assert use_calculator("import os") is None
        assert use_calculator("__import__") is None

    def test_power_operator_blocked(self):
        """Test that power operator is blocked for security."""
        assert use_calculator("2 ** 1000") is None
        assert use_calculator("10**10") is None

    def test_timeout(self):
        """Test that timeout works for expensive operations."""
        # This would be expensive but is blocked by ** check
        assert use_calculator("2 ** 100000") is None

    def test_division_by_zero(self):
        """Test division by zero returns None."""
        result = use_calculator("1 / 0")
        assert result is None

    def test_eval_with_timeout_success(self):
        """Test eval_with_timeout with valid expression."""
        result = eval_with_timeout("2 + 2", max_time=1)
        assert result == 4

    def test_eval_with_timeout_failure(self):
        """Test eval_with_timeout with invalid expression."""
        result = eval_with_timeout("invalid syntax!", max_time=1)
        assert result is None


class TestKVCache:
    """Test the KV cache implementation."""

    def test_initialization(self):
        """Test KV cache initialization."""
        cache = KVCache(
            batch_size=2,
            num_heads=4,
            seq_len=100,
            head_dim=64,
            num_layers=6
        )
        assert cache.kv_cache is None  # Lazy init
        assert cache.pos == 0
        expected_shape = (6, 2, 2, 4, 100, 64)
        assert cache.kv_shape == expected_shape

    def test_reset(self):
        """Test cache reset."""
        cache = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=100,
            head_dim=64,
            num_layers=6
        )
        cache.pos = 50
        cache.reset()
        assert cache.pos == 0

    def test_insert_kv_lazy_init(self):
        """Test that cache is lazily initialized on first insert."""
        cache = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=100,
            head_dim=64,
            num_layers=6
        )
        assert cache.kv_cache is None

        # Create dummy k, v tensors
        k = torch.randn(1, 4, 10, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 10, 64, dtype=torch.bfloat16)

        # Insert into layer 0
        k_view, v_view = cache.insert_kv(0, k, v)

        # Cache should now be initialized
        assert cache.kv_cache is not None
        assert cache.kv_cache.dtype == torch.bfloat16

    def test_insert_kv_returns_views(self):
        """Test that insert_kv returns correct views."""
        cache = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=100,
            head_dim=64,
            num_layers=2
        )

        # First insert
        k1 = torch.randn(1, 4, 5, 64, dtype=torch.bfloat16)
        v1 = torch.randn(1, 4, 5, 64, dtype=torch.bfloat16)
        k_view, v_view = cache.insert_kv(0, k1, v1)
        assert k_view.shape == (1, 4, 5, 64)

        # Second layer should increment position
        k2 = torch.randn(1, 4, 5, 64, dtype=torch.bfloat16)
        v2 = torch.randn(1, 4, 5, 64, dtype=torch.bfloat16)
        k_view, v_view = cache.insert_kv(1, k2, v2)
        assert cache.pos == 5  # Position incremented after last layer

    def test_insert_kv_accumulates(self):
        """Test that cache accumulates across multiple inserts."""
        cache = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=100,
            head_dim=64,
            num_layers=1
        )

        # First insert: 5 tokens
        k1 = torch.randn(1, 4, 5, 64, dtype=torch.bfloat16)
        v1 = torch.randn(1, 4, 5, 64, dtype=torch.bfloat16)
        k_view1, v_view1 = cache.insert_kv(0, k1, v1)
        assert k_view1.shape == (1, 4, 5, 64)
        assert cache.pos == 5

        # Second insert: 3 more tokens
        k2 = torch.randn(1, 4, 3, 64, dtype=torch.bfloat16)
        v2 = torch.randn(1, 4, 3, 64, dtype=torch.bfloat16)
        k_view2, v_view2 = cache.insert_kv(0, k2, v2)
        assert k_view2.shape == (1, 4, 8, 64)  # Total 5 + 3 = 8
        assert cache.pos == 8

    def test_dynamic_growth(self):
        """Test that cache grows dynamically when needed."""
        cache = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=10,  # Small initial size
            head_dim=64,
            num_layers=1
        )

        # Insert more tokens than initial size
        k = torch.randn(1, 4, 20, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 20, 64, dtype=torch.bfloat16)
        k_view, v_view = cache.insert_kv(0, k, v)

        # Cache should have grown
        assert cache.kv_cache.size(4) >= 20
        assert k_view.shape == (1, 4, 20, 64)

    def test_prefill_from_another_cache(self):
        """Test prefilling from another cache."""
        # Create source cache with data
        source = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=100,
            head_dim=64,
            num_layers=2
        )
        k = torch.randn(1, 4, 10, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 10, 64, dtype=torch.bfloat16)
        source.insert_kv(0, k, v)
        source.insert_kv(1, k, v)

        # Create target cache and prefill
        target = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=200,
            head_dim=64,
            num_layers=2
        )
        target.prefill(source)

        assert target.pos == source.pos
        assert target.kv_cache is not None

    def test_prefill_batch_expansion(self):
        """Test prefilling with batch expansion."""
        # Source with batch_size=1
        source = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=100,
            head_dim=64,
            num_layers=2
        )
        k = torch.randn(1, 4, 10, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 10, 64, dtype=torch.bfloat16)
        source.insert_kv(0, k, v)
        source.insert_kv(1, k, v)

        # Target with batch_size=4
        target = KVCache(
            batch_size=4,
            num_heads=4,
            seq_len=200,
            head_dim=64,
            num_layers=2
        )
        target.prefill(source)

        assert target.pos == source.pos
        assert target.kv_cache.shape[2] == 4  # Batch dimension


class TestSampleNextToken:
    """Test the token sampling function."""

    def test_greedy_sampling_temperature_zero(self):
        """Test greedy sampling with temperature=0."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 2.5, 1.5]])
        rng = torch.Generator()
        rng.manual_seed(42)

        result = sample_next_token(logits, rng, temperature=0.0)
        assert result.shape == (1, 1)
        assert result[0, 0] == 2  # Index of max value (3.0)

    def test_temperature_sampling(self):
        """Test sampling with temperature > 0."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 2.5, 1.5]])
        rng = torch.Generator()
        rng.manual_seed(42)

        result = sample_next_token(logits, rng, temperature=1.0)
        assert result.shape == (1, 1)
        assert 0 <= result[0, 0] < 5

    def test_top_k_sampling(self):
        """Test top-k sampling."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 2.5, 1.5]])
        rng = torch.Generator()
        rng.manual_seed(42)

        # Sample from top 2
        result = sample_next_token(logits, rng, temperature=1.0, top_k=2)
        assert result.shape == (1, 1)
        # Should be one of top 2 indices: 2 (value 3.0) or 3 (value 2.5)
        assert result[0, 0] in [2, 3]

    def test_batch_sampling(self):
        """Test sampling with batch size > 1."""
        B = 4
        vocab_size = 1000
        logits = torch.randn(B, vocab_size)
        rng = torch.Generator()
        rng.manual_seed(42)

        result = sample_next_token(logits, rng, temperature=1.0)
        assert result.shape == (B, 1)
        assert all(0 <= result[i, 0] < vocab_size for i in range(B))

    def test_reproducibility_with_seed(self):
        """Test that sampling is reproducible with same seed."""
        logits = torch.randn(1, 1000)

        rng1 = torch.Generator()
        rng1.manual_seed(42)
        result1 = sample_next_token(logits, rng1, temperature=1.0)

        rng2 = torch.Generator()
        rng2.manual_seed(42)
        result2 = sample_next_token(logits, rng2, temperature=1.0)

        assert torch.equal(result1, result2)


class TestRowState:
    """Test the RowState class."""

    def test_initialization_empty(self):
        """Test empty initialization."""
        state = RowState()
        assert state.current_tokens == []
        assert len(state.forced_tokens) == 0
        assert state.in_python_block is False
        assert state.python_expr_tokens == []
        assert state.completed is False

    def test_initialization_with_tokens(self):
        """Test initialization with tokens."""
        tokens = [1, 2, 3, 4]
        state = RowState(current_tokens=tokens)
        assert state.current_tokens == [1, 2, 3, 4]

    def test_forced_tokens_queue(self):
        """Test forced tokens deque operations."""
        state = RowState()
        state.forced_tokens.append(100)
        state.forced_tokens.append(200)

        assert len(state.forced_tokens) == 2
        assert state.forced_tokens.popleft() == 100
        assert state.forced_tokens.popleft() == 200
        assert len(state.forced_tokens) == 0


class TestEngine:
    """Test the Engine class for inference."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = GPTConfig(
            sequence_len=256,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=128
        )
        model = GPT(config)
        model.init_weights()
        model.eval()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        class MockTokenizer:
            def encode_special(self, token):
                special_tokens = {
                    "<|python_start|>": 1001,
                    "<|python_end|>": 1002,
                    "<|output_start|>": 1003,
                    "<|output_end|>": 1004,
                    "<|assistant_end|>": 1005,
                }
                return special_tokens.get(token, 0)

            def get_bos_token_id(self):
                return 1

            def encode(self, text):
                # Simple mock: convert each character to ASCII value
                return [ord(c) % 1000 for c in text]

            def decode(self, tokens):
                # Simple mock: convert back
                return ''.join(chr(t) if t < 128 else 'X' for t in tokens)

        return MockTokenizer()

    def test_engine_initialization(self, small_model, mock_tokenizer):
        """Test engine initialization."""
        engine = Engine(small_model, mock_tokenizer)
        assert engine.model is small_model
        assert engine.tokenizer is mock_tokenizer

    def test_generate_basic(self, small_model, mock_tokenizer):
        """Test basic generation."""
        engine = Engine(small_model, mock_tokenizer)
        tokens = [1, 2, 3, 4, 5]

        generated = []
        for token_column, token_masks in engine.generate(
            tokens,
            num_samples=1,
            max_tokens=10,
            temperature=1.0,
            seed=42
        ):
            generated.append(token_column[0])
            if len(generated) >= 10:
                break

        assert len(generated) == 10
        assert all(isinstance(t, int) for t in generated)

    def test_generate_batch(self, small_model, mock_tokenizer):
        """Test batch generation."""
        engine = Engine(small_model, mock_tokenizer)
        tokens = [1, 2, 3, 4, 5]

        results, masks = engine.generate_batch(
            tokens,
            num_samples=3,
            max_tokens=10,
            temperature=1.0,
            seed=42
        )

        assert len(results) == 3
        assert len(masks) == 3
        assert all(len(r) > len(tokens) for r in results)

    def test_generate_deterministic(self, small_model, mock_tokenizer):
        """Test that generation is deterministic with same seed."""
        engine = Engine(small_model, mock_tokenizer)
        tokens = [1, 2, 3, 4, 5]

        results1, _ = engine.generate_batch(
            tokens,
            num_samples=2,
            max_tokens=10,
            temperature=1.0,
            seed=42
        )

        results2, _ = engine.generate_batch(
            tokens,
            num_samples=2,
            max_tokens=10,
            temperature=1.0,
            seed=42
        )

        assert results1 == results2

    def test_generate_multiple_samples(self, small_model, mock_tokenizer):
        """Test generating multiple samples in parallel."""
        engine = Engine(small_model, mock_tokenizer)
        tokens = [1, 2, 3]

        num_samples = 4
        generated_tokens = [[] for _ in range(num_samples)]

        for token_column, token_masks in engine.generate(
            tokens,
            num_samples=num_samples,
            max_tokens=5,
            temperature=1.0,
            seed=42
        ):
            assert len(token_column) == num_samples
            assert len(token_masks) == num_samples
            for i, token in enumerate(token_column):
                generated_tokens[i].append(token)

        # All samples should have generated tokens
        assert all(len(g) > 0 for g in generated_tokens)

    def test_max_tokens_limit(self, small_model, mock_tokenizer):
        """Test that generation respects max_tokens limit."""
        engine = Engine(small_model, mock_tokenizer)
        tokens = [1, 2, 3]
        max_tokens = 15

        generated_count = 0
        for token_column, token_masks in engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=1.0,
            seed=42
        ):
            generated_count += 1

        assert generated_count <= max_tokens

    def test_greedy_generation(self, small_model, mock_tokenizer):
        """Test greedy generation with temperature=0."""
        engine = Engine(small_model, mock_tokenizer)
        tokens = [1, 2, 3, 4, 5]

        results1, _ = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=10,
            temperature=0.0
        )

        results2, _ = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=10,
            temperature=0.0
        )

        # Temperature=0 should be deterministic
        assert results1 == results2

    def test_token_masks(self, small_model, mock_tokenizer):
        """Test that token masks are returned correctly."""
        engine = Engine(small_model, mock_tokenizer)
        tokens = [1, 2, 3]

        results, masks = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=5,
            temperature=1.0,
            seed=42
        )

        # Masks should have same length as results
        assert len(masks[0]) == len(results[0])
        # Most tokens should be sampled (mask=1), not forced (mask=0)
        # First tokens match input, so should have mask=0
        assert sum(masks[0][:len(tokens)]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
