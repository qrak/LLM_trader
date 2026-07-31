"""Tests for RLPolicyNetwork — config, status, error handling (no model download)."""

from unittest.mock import MagicMock, patch

import pytest


class TestRLPolicyNetwork:
    """Unit tests for RL inference module behavior."""

    @pytest.fixture
    def mock_config_disabled(self):
        config = MagicMock()
        config.RL_TRAINING_ENABLED = False
        config.RL_TRAINING_MODEL = "Qwen/Qwen3-0.6B-Instruct"
        config.RL_TRAINING_UPDATE_INTERVAL = 10
        config.RL_TRAINING_CHECKPOINT_DIR = "data/rl_checkpoints"
        config.RL_TRAINING_DEVICE = "auto"
        return config

    @pytest.fixture
    def mock_config_enabled(self):
        config = MagicMock()
        config.RL_TRAINING_ENABLED = True
        config.RL_TRAINING_MODEL = "Qwen/Qwen3-0.6B-Instruct"
        config.RL_TRAINING_UPDATE_INTERVAL = 10
        config.RL_TRAINING_CHECKPOINT_DIR = "data/rl_checkpoints"
        config.RL_TRAINING_DEVICE = "cpu"
        return config

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    def test_disabled_by_config(self, mock_config_disabled, mock_logger):
        from src.trading.rl_policy import RLPolicyNetwork
        rl = RLPolicyNetwork(config=mock_config_disabled, logger=mock_logger)
        assert rl.enabled is False
        assert "DISABLED" in rl.status

    @patch("src.trading.rl_policy.RLPolicyNetwork._check_ram", return_value=False)
    @patch("src.trading.rl_policy.RLPolicyNetwork._initialize_model")
    def test_insufficient_ram_disables(self, mock_init, mock_ram, mock_config_enabled, mock_logger):
        from src.trading.rl_policy import RLPolicyNetwork
        # Override _initialize_model to not actually download
        def noop_init(self):
            self._error = "Insufficient RAM"
        from unittest.mock import patch as mock_patch
        with mock_patch.object(RLPolicyNetwork, "_initialize_model", noop_init):
            rl = RLPolicyNetwork(config=mock_config_enabled, logger=mock_logger)
        assert rl.enabled is False

    def test_device_resolution_cpu(self, mock_config_enabled, mock_logger):
        from src.trading.rl_policy import RLPolicyNetwork
        rl = RLPolicyNetwork(config=mock_config_enabled, logger=mock_logger)
        # Can't actually init model, just test device resolution
        assert rl._resolve_device() == "cpu"

    def test_device_resolution_auto_no_torch(self):
        """force auto without torch → falls back to cpu."""
        config = MagicMock()
        config.RL_TRAINING_DEVICE = "auto"
        with patch.dict("sys.modules", {"torch": None}):
            from src.trading.rl_policy import RLPolicyNetwork
            rl = RLPolicyNetwork.__new__(RLPolicyNetwork)
            rl._config = config
            assert rl._resolve_device() == "cpu"

    def test_generate_decision_raises_when_disabled(self, mock_config_disabled, mock_logger):
        from src.trading.rl_policy import RLPolicyNetwork
        rl = RLPolicyNetwork(config=mock_config_disabled, logger=mock_logger)
        with pytest.raises(RuntimeError, match="not available"):
            rl.generate_decision("system", "user")

    def test_status_disabled(self, mock_config_disabled, mock_logger):
        from src.trading.rl_policy import RLPolicyNetwork
        rl = RLPolicyNetwork(config=mock_config_disabled, logger=mock_logger)
        assert "DISABLED (config)" in rl.status

    def test_missing_imports_error_message(self, mock_config_enabled, mock_logger):
        """Simulate missing transformers import."""
        from src.trading.rl_policy import RLPolicyNetwork
        with patch.dict("sys.modules", {"transformers": None}):
            rl = RLPolicyNetwork.__new__(RLPolicyNetwork)
            rl._config = mock_config_enabled
            rl._logger = mock_logger
            rl.enabled = False
            rl._error = "Missing dependencies: transformers"
            rl._model = None
            rl._tokenizer = None
            assert "Missing dependencies" in rl._error
            assert rl.enabled is False

    def test_ram_check_no_psutil(self):
        """Without psutil, assume RAM is OK."""
        config = MagicMock()
        config.RL_TRAINING_DEVICE = "auto"
        rl = MagicMock()
        rl._config = config
        rl.MIN_RAM_GB = 6.0
        with patch.dict("sys.modules", {"psutil": None}):
            from src.trading.rl_policy import RLPolicyNetwork
            result = RLPolicyNetwork._check_ram(rl)
            assert result is True
