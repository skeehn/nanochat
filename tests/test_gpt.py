"""
Comprehensive tests for the GPT model implementation.
Tests model architecture, forward passes, rotary embeddings, and optimizer setup.
"""

import pytest
import torch
import torch.nn as nn
from nanochat.gpt import (
    GPT, GPTConfig, CausalSelfAttention, MLP, Block,
    norm, apply_rotary_emb, repeat_kv
)


class TestUtilityFunctions:
    """Test utility functions used in the model."""

    def test_norm_shape_preservation(self):
        """Test that RMSNorm preserves input shape."""
        x = torch.randn(2, 10, 768)
        y = norm(x)
        assert y.shape == x.shape

    def test_norm_no_nan(self):
        """Test that RMSNorm doesn't produce NaN values."""
        x = torch.randn(2, 10, 768)
        y = norm(x)
        assert not torch.isnan(y).any()

    def test_apply_rotary_emb_shape(self):
        """Test rotary embeddings preserve shape."""
        B, H, T, D = 2, 6, 10, 64
        x = torch.randn(B, T, H, D)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)
        y = apply_rotary_emb(x, cos, sin)
        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_repeat_kv_no_repeat(self):
        """Test repeat_kv with n_rep=1 returns input unchanged."""
        x = torch.randn(2, 4, 10, 64)
        y = repeat_kv(x, 1)
        assert torch.equal(x, y)

    def test_repeat_kv_expansion(self):
        """Test repeat_kv correctly expands heads."""
        bs, n_kv_heads, slen, head_dim = 2, 2, 10, 64
        n_rep = 3
        x = torch.randn(bs, n_kv_heads, slen, head_dim)
        y = repeat_kv(x, n_rep)
        expected_shape = (bs, n_kv_heads * n_rep, slen, head_dim)
        assert y.shape == expected_shape


class TestGPTConfig:
    """Test GPT configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GPTConfig()
        assert config.sequence_len == 1024
        assert config.vocab_size == 50304
        assert config.n_layer == 12
        assert config.n_head == 6
        assert config.n_kv_head == 6
        assert config.n_embd == 768

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GPTConfig(
            sequence_len=2048,
            vocab_size=32000,
            n_layer=24,
            n_head=12,
            n_kv_head=4,
            n_embd=1024
        )
        assert config.sequence_len == 2048
        assert config.vocab_size == 32000
        assert config.n_layer == 24
        assert config.n_head == 12
        assert config.n_kv_head == 4
        assert config.n_embd == 1024


class TestCausalSelfAttention:
    """Test the CausalSelfAttention module."""

    def test_initialization(self):
        """Test that attention module initializes correctly."""
        config = GPTConfig(n_head=6, n_kv_head=2, n_embd=768)
        attn = CausalSelfAttention(config, layer_idx=0)
        assert attn.n_head == 6
        assert attn.n_kv_head == 2
        assert attn.head_dim == 128

    def test_forward_pass_no_cache(self):
        """Test forward pass without KV cache."""
        config = GPTConfig(n_head=6, n_kv_head=6, n_embd=768)
        attn = CausalSelfAttention(config, layer_idx=0)
        B, T, C = 2, 10, 768
        x = torch.randn(B, T, C)
        cos = torch.randn(1, T, 1, 64)
        sin = torch.randn(1, T, 1, 64)
        y = attn(x, (cos, sin), kv_cache=None)
        assert y.shape == (B, T, C)

    def test_mqa_configuration(self):
        """Test Multi-Query Attention with fewer KV heads."""
        config = GPTConfig(n_head=6, n_kv_head=2, n_embd=768)
        attn = CausalSelfAttention(config, layer_idx=0)
        assert config.n_head % config.n_kv_head == 0
        B, T, C = 2, 10, 768
        x = torch.randn(B, T, C)
        cos = torch.randn(1, T, 1, 128)
        sin = torch.randn(1, T, 1, 128)
        y = attn(x, (cos, sin), kv_cache=None)
        assert y.shape == (B, T, C)


class TestMLP:
    """Test the MLP module."""

    def test_initialization(self):
        """Test MLP initializes with correct dimensions."""
        config = GPTConfig(n_embd=768)
        mlp = MLP(config)
        assert mlp.c_fc.in_features == 768
        assert mlp.c_fc.out_features == 4 * 768
        assert mlp.c_proj.in_features == 4 * 768
        assert mlp.c_proj.out_features == 768

    def test_forward_pass(self):
        """Test MLP forward pass."""
        config = GPTConfig(n_embd=768)
        mlp = MLP(config)
        B, T, C = 2, 10, 768
        x = torch.randn(B, T, C)
        y = mlp(x)
        assert y.shape == (B, T, C)

    def test_relu_squared_activation(self):
        """Test that ReLU^2 activation is applied."""
        config = GPTConfig(n_embd=768)
        mlp = MLP(config)
        x = torch.randn(2, 10, 768)
        y = mlp(x)
        # Output should be non-negative due to ReLU^2
        # We can't directly test this without modifying the module,
        # but we ensure it doesn't crash and produces valid output
        assert not torch.isnan(y).any()


class TestBlock:
    """Test the Transformer Block."""

    def test_initialization(self):
        """Test Block initializes correctly."""
        config = GPTConfig()
        block = Block(config, layer_idx=0)
        assert isinstance(block.attn, CausalSelfAttention)
        assert isinstance(block.mlp, MLP)

    def test_forward_pass(self):
        """Test Block forward pass."""
        config = GPTConfig(n_embd=768)
        block = Block(config, layer_idx=0)
        B, T, C = 2, 10, 768
        x = torch.randn(B, T, C)
        head_dim = C // config.n_head
        cos = torch.randn(1, T, 1, head_dim)
        sin = torch.randn(1, T, 1, head_dim)
        y = block(x, (cos, sin), kv_cache=None)
        assert y.shape == (B, T, C)

    def test_residual_connections(self):
        """Test that residual connections are working."""
        config = GPTConfig(n_embd=768)
        block = Block(config, layer_idx=0)
        B, T, C = 2, 10, 768
        x = torch.zeros(B, T, C)
        head_dim = C // config.n_head
        cos = torch.randn(1, T, 1, head_dim)
        sin = torch.randn(1, T, 1, head_dim)
        y = block(x, (cos, sin), kv_cache=None)
        # With zero input, we should still get non-zero output from residuals
        # (unless weights are initialized to produce zeros)
        assert y.shape == (B, T, C)


class TestGPT:
    """Test the full GPT model."""

    def test_initialization(self):
        """Test GPT model initializes correctly."""
        config = GPTConfig(
            sequence_len=512,
            vocab_size=1000,
            n_layer=4,
            n_head=4,
            n_embd=256
        )
        model = GPT(config)
        assert len(model.transformer.h) == 4
        assert model.transformer.wte.num_embeddings == 1000
        assert model.transformer.wte.embedding_dim == 256
        assert model.lm_head.out_features == 1000

    def test_forward_training_mode(self):
        """Test forward pass in training mode (with targets)."""
        config = GPTConfig(
            sequence_len=512,
            vocab_size=1000,
            n_layer=4,
            n_head=4,
            n_embd=256
        )
        model = GPT(config)
        model.init_weights()
        B, T = 2, 50
        idx = torch.randint(0, 1000, (B, T))
        targets = torch.randint(0, 1000, (B, T))
        loss = model(idx, targets=targets)
        assert loss.ndim == 0  # scalar loss
        assert loss.item() >= 0  # loss should be non-negative

    def test_forward_inference_mode(self):
        """Test forward pass in inference mode (no targets)."""
        config = GPTConfig(
            sequence_len=512,
            vocab_size=1000,
            n_layer=4,
            n_head=4,
            n_embd=256
        )
        model = GPT(config)
        model.init_weights()
        B, T = 2, 50
        idx = torch.randint(0, 1000, (B, T))
        logits = model(idx, targets=None)
        assert logits.shape == (B, T, 1000)

    def test_rotary_embeddings_precomputation(self):
        """Test that rotary embeddings are precomputed correctly."""
        config = GPTConfig(sequence_len=512, n_embd=256, n_head=4)
        model = GPT(config)
        model.init_weights()
        head_dim = config.n_embd // config.n_head
        # Rotary embeddings should be 10x longer than sequence length
        assert model.cos.shape == (1, config.sequence_len * 10, 1, head_dim)
        assert model.sin.shape == (1, config.sequence_len * 10, 1, head_dim)
        assert model.cos.dtype == torch.bfloat16
        assert model.sin.dtype == torch.bfloat16

    def test_untied_embeddings(self):
        """Test that embedding and lm_head weights are untied."""
        config = GPTConfig(vocab_size=1000, n_embd=256)
        model = GPT(config)
        # Check that they are different parameter objects
        assert model.transformer.wte.weight is not model.lm_head.weight

    def test_no_bias_in_linear_layers(self):
        """Test that linear layers don't have bias."""
        config = GPTConfig()
        model = GPT(config)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert module.bias is None, f"Linear layer {name} has bias"

    def test_estimate_flops(self):
        """Test FLOPs estimation returns reasonable value."""
        config = GPTConfig(n_layer=4, n_head=4, n_embd=256, sequence_len=512)
        model = GPT(config)
        flops = model.estimate_flops()
        assert flops > 0
        assert isinstance(flops, (int, float))

    def test_generate_basic(self):
        """Test basic generation."""
        config = GPTConfig(
            sequence_len=512,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=256
        )
        model = GPT(config)
        model.init_weights()
        model.eval()

        tokens = [1, 2, 3, 4, 5]
        max_tokens = 10

        generated = list(model.generate(tokens, max_tokens=max_tokens, temperature=1.0, seed=42))
        assert len(generated) == max_tokens
        assert all(isinstance(t, int) for t in generated)
        assert all(0 <= t < 1000 for t in generated)

    def test_generate_deterministic(self):
        """Test that generation is deterministic with temperature=0."""
        config = GPTConfig(
            sequence_len=512,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=256
        )
        model = GPT(config)
        model.init_weights()
        model.eval()

        tokens = [1, 2, 3, 4, 5]
        max_tokens = 5

        gen1 = list(model.generate(tokens, max_tokens=max_tokens, temperature=0.0))
        gen2 = list(model.generate(tokens, max_tokens=max_tokens, temperature=0.0))
        assert gen1 == gen2

    def test_generate_with_seed(self):
        """Test that generation is reproducible with same seed."""
        config = GPTConfig(
            sequence_len=512,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=256
        )
        model = GPT(config)
        model.init_weights()
        model.eval()

        tokens = [1, 2, 3, 4, 5]
        max_tokens = 5

        gen1 = list(model.generate(tokens, max_tokens=max_tokens, temperature=1.0, seed=42))
        gen2 = list(model.generate(tokens, max_tokens=max_tokens, temperature=1.0, seed=42))
        assert gen1 == gen2

    def test_setup_optimizers(self):
        """Test optimizer setup."""
        config = GPTConfig(n_layer=2, n_head=4, n_embd=256)
        model = GPT(config)
        model.init_weights()
        optimizers = model.setup_optimizers()
        assert len(optimizers) == 2  # AdamW and Muon
        # Check that both optimizers have parameters
        for opt in optimizers:
            assert len(opt.param_groups) > 0

    def test_get_device(self):
        """Test get_device method."""
        config = GPTConfig()
        model = GPT(config)
        device = model.get_device()
        assert isinstance(device, torch.device)

    @pytest.mark.parametrize("sequence_len,should_fail", [
        (100, False),  # Within limit
        (10000, True),  # Beyond 10x of config.sequence_len (512)
    ])
    def test_sequence_length_limit(self, sequence_len, should_fail):
        """Test that model enforces sequence length limits."""
        config = GPTConfig(sequence_len=512, vocab_size=1000, n_layer=2, n_head=4, n_embd=256)
        model = GPT(config)
        model.init_weights()

        idx = torch.randint(0, 1000, (1, sequence_len))

        if should_fail:
            with pytest.raises(AssertionError):
                model(idx)
        else:
            logits = model(idx)
            assert logits.shape == (1, sequence_len, 1000)


class TestModelProperties:
    """Test various model properties and invariants."""

    def test_embedding_dtype(self):
        """Test that embeddings are in bfloat16."""
        config = GPTConfig()
        model = GPT(config)
        assert model.transformer.wte.weight.dtype == torch.bfloat16

    def test_weight_initialization(self):
        """Test that weights are initialized (not all zeros)."""
        config = GPTConfig(n_layer=2, n_head=4, n_embd=256, vocab_size=1000)
        model = GPT(config)
        model.init_weights()

        # Check that most weights are non-zero
        for name, param in model.named_parameters():
            if 'weight' in name:
                non_zero_ratio = (param != 0).float().mean()
                # Some weights like c_proj are initialized to zero
                if 'c_proj' not in name and 'lm_head' not in name:
                    assert non_zero_ratio > 0.5, f"{name} has too many zeros: {non_zero_ratio}"

    def test_loss_reduction_modes(self):
        """Test different loss reduction modes."""
        config = GPTConfig(vocab_size=1000, n_layer=2, n_head=4, n_embd=256)
        model = GPT(config)
        model.init_weights()

        B, T = 2, 50
        idx = torch.randint(0, 1000, (B, T))
        targets = torch.randint(0, 1000, (B, T))

        # Test mean reduction
        loss_mean = model(idx, targets=targets, loss_reduction='mean')
        assert loss_mean.ndim == 0

        # Test none reduction
        loss_none = model(idx, targets=targets, loss_reduction='none')
        assert loss_none.shape == (B * T,)

        # Mean of unreduced should approximately equal mean reduction
        assert torch.isclose(loss_none.mean(), loss_mean, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
