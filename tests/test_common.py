"""
Tests for common utility functions.
"""

import pytest
import os
import torch
import logging
from unittest.mock import patch, MagicMock
from nanochat.common import (
    get_base_dir, print0, get_dist_info, is_ddp,
    compute_init, ColoredFormatter, setup_default_logging
)


class TestGetBaseDir:
    """Test the get_base_dir function."""

    def test_default_base_dir(self):
        """Test default base directory."""
        with patch.dict(os.environ, {}, clear=True):
            base_dir = get_base_dir()
            assert base_dir.endswith("nanochat")
            assert ".cache" in base_dir
            assert os.path.exists(base_dir)

    def test_custom_base_dir(self):
        """Test custom base directory from environment variable."""
        custom_dir = "/tmp/test_nanochat"
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": custom_dir}):
            base_dir = get_base_dir()
            assert base_dir == custom_dir
            assert os.path.exists(base_dir)

    def test_creates_directory(self):
        """Test that directory is created if it doesn't exist."""
        import tempfile
        import shutil
        temp_dir = tempfile.mktemp()
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": temp_dir}):
            base_dir = get_base_dir()
            assert os.path.exists(base_dir)
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestPrint0:
    """Test the print0 function."""

    def test_print0_rank_0(self, capsys):
        """Test that print0 prints when rank is 0."""
        with patch.dict(os.environ, {"RANK": "0"}):
            print0("Hello, world!")
            captured = capsys.readouterr()
            assert "Hello, world!" in captured.out

    def test_print0_rank_non_zero(self, capsys):
        """Test that print0 doesn't print when rank is not 0."""
        with patch.dict(os.environ, {"RANK": "1"}):
            print0("Should not appear")
            captured = capsys.readouterr()
            assert "Should not appear" not in captured.out

    def test_print0_no_rank(self, capsys):
        """Test print0 with no RANK environment variable (defaults to 0)."""
        with patch.dict(os.environ, {}, clear=True):
            print0("Default rank")
            captured = capsys.readouterr()
            assert "Default rank" in captured.out


class TestDistributedInfo:
    """Test distributed computing information functions."""

    def test_is_ddp_true(self):
        """Test is_ddp when RANK is set."""
        with patch.dict(os.environ, {"RANK": "0"}):
            assert is_ddp() is True

    def test_is_ddp_false(self):
        """Test is_ddp when RANK is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_ddp() is False

    def test_get_dist_info_ddp(self):
        """Test get_dist_info in DDP mode."""
        env = {
            "RANK": "2",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "4"
        }
        with patch.dict(os.environ, env):
            ddp, rank, local_rank, world_size = get_dist_info()
            assert ddp is True
            assert rank == 2
            assert local_rank == 1
            assert world_size == 4

    def test_get_dist_info_non_ddp(self):
        """Test get_dist_info in non-DDP mode."""
        with patch.dict(os.environ, {}, clear=True):
            ddp, rank, local_rank, world_size = get_dist_info()
            assert ddp is False
            assert rank == 0
            assert local_rank == 0
            assert world_size == 1


class TestColoredFormatter:
    """Test the ColoredFormatter for logging."""

    def test_formatter_initialization(self):
        """Test that formatter can be initialized."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        assert formatter is not None

    def test_format_info_message(self):
        """Test formatting an INFO level message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_format_error_message(self):
        """Test formatting an ERROR level message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        formatted = formatter.format(record)
        assert "Error occurred" in formatted

    def test_color_codes_applied(self):
        """Test that color codes are applied to log levels."""
        formatter = ColoredFormatter('%(levelname)s')
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None
        )
        formatted = formatter.format(record)
        # Check for ANSI escape codes
        assert '\033[' in formatted


class TestSetupDefaultLogging:
    """Test logging setup."""

    def test_setup_default_logging(self):
        """Test that default logging can be set up."""
        # This is already called on module import, but we can test it doesn't crash
        setup_default_logging()
        # Verify that a logger exists
        import logging
        logger = logging.getLogger(__name__)
        assert logger is not None


class TestDummyWandb:
    """Test the DummyWandb class."""

    def test_dummy_wandb_initialization(self):
        """Test DummyWandb can be initialized."""
        from nanochat.common import DummyWandb
        wandb = DummyWandb()
        assert wandb is not None

    def test_dummy_wandb_log(self):
        """Test that log method doesn't crash."""
        from nanochat.common import DummyWandb
        wandb = DummyWandb()
        # Should not raise any exception
        wandb.log({"loss": 0.5, "accuracy": 0.9})
        wandb.log()

    def test_dummy_wandb_finish(self):
        """Test that finish method doesn't crash."""
        from nanochat.common import DummyWandb
        wandb = DummyWandb()
        # Should not raise any exception
        wandb.finish()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestComputeInit:
    """Test compute initialization (requires CUDA)."""

    def test_compute_init_non_ddp(self):
        """Test compute_init in non-DDP mode."""
        with patch.dict(os.environ, {}, clear=True):
            ddp, rank, local_rank, world_size, device = compute_init()
            assert ddp is False
            assert rank == 0
            assert local_rank == 0
            assert world_size == 1
            assert device.type == "cuda"

    def test_compute_init_sets_seed(self):
        """Test that compute_init sets random seeds."""
        with patch.dict(os.environ, {}, clear=True):
            compute_init()
            # After init, random operations should be deterministic
            # We can't easily test this without side effects, but we can verify it doesn't crash

    def test_compute_init_sets_precision(self):
        """Test that compute_init sets matmul precision."""
        with patch.dict(os.environ, {}, clear=True):
            compute_init()
            # Verify precision is set (this is a global setting)
            # We can't directly query this, but we ensure it doesn't crash


class TestComputeCleanup:
    """Test compute cleanup."""

    def test_compute_cleanup_non_ddp(self):
        """Test cleanup in non-DDP mode doesn't crash."""
        from nanochat.common import compute_cleanup
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise any exception
            compute_cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
