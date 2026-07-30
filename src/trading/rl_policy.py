"""RL Policy Network — local LLM inference as replacement for external API calls.

Uses Qwen3-0.6B-Instruct (Apache 2.0 license) for CPU-only inference.
Model is auto-downloaded from HuggingFace on first use (~1.2 GB).
If hardware is insufficient (RAM < 6 GB free), module self-disables with a clear error.

This module provides a local, free alternative to external LLM providers.
Training (PPO fine-tuning) is a separate feature not yet implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config.loader import Config
    from src.logger.logger import Logger


class RLPolicyNetwork:
    """Local LLM inference engine for trading decisions.

    Loads Qwen3-0.6B-Instruct on CPU with half-precision (fp16→bf16 where supported)
    or 4-bit quantization for low-RAM systems.
    """

    # HuggingFace model identifier
    MODEL_ID = "Qwen/Qwen3-0.6B-Instruct"

    # Minimum free RAM in GB required to load the model
    MIN_RAM_GB = 6.0

    # Maximum tokens for generation (trading decisions are concise)
    MAX_NEW_TOKENS = 512

    def __init__(self, config: Config, logger: Logger) -> None:
        """Initialize the RL policy network.

        On first call, downloads the model from HuggingFace (~1.2 GB).
        If RAM is insufficient, self.enabled remains False.

        Args:
            config: Application configuration.
            logger: Logger instance.
        """
        self._config = config
        self.logger = logger
        self._model: Any = None
        self._tokenizer: Any = None
        self.enabled: bool = False
        self._error: str = ""

        if not config.RL_TRAINING_ENABLED:
            self._error = "RL training disabled in config (rl_training.enabled=false)"
            return

        self._initialize_model()

    def _check_ram(self) -> bool:
        """Check if the system has enough free RAM to load the model."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < self.MIN_RAM_GB:
                self._error = (
                    f"Insufficient RAM: {available_gb:.1f} GB available, "
                    f"need {self.MIN_RAM_GB:.0f} GB. Disable [rl_training] in config."
                )
                return False
            return True
        except ImportError:
            # psutil not installed — try anyway
            return True

    def _initialize_model(self) -> None:
        """Download and load the model. Sets self.enabled on success."""
        if not self._check_ram():
            self.logger.error(self._error)
            return

        self.logger.info(
            "Loading RL policy model: %s (first run will download ~1.2 GB)...",
            self.MODEL_ID,
        )

        try:
            device = self._resolve_device()
            self.logger.info("Using device: %s", device)

            # Try 4-bit quantization first (less RAM), fall back to full precision
            load_kwargs: dict[str, Any] = {}

            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                )

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = bnb_config
                self.logger.info("Using 4-bit quantization (~2 GB RAM)")
            except ImportError as e:
                self._error = (
                    f"Missing dependencies: {e}. "
                    "Install with: pip install transformers torch accelerate"
                )
                self.logger.error(self._error)
                return

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                device_map=device if device != "cpu" else None,
                trust_remote_code=True,
                **load_kwargs,
            )

            if device == "cpu":
                self._model = self._model.to("cpu")

            self._model.eval()
            self.enabled = True
            self._error = ""
            self.logger.info("RL policy model loaded successfully.")

        except ImportError as e:
            self._error = (
                f"Missing dependencies: {e}. "
                "Install with: pip install transformers torch accelerate"
            )
            self.logger.error(self._error)
        except MemoryError:
            self._error = (
                "Out of memory loading model. Need ~6 GB free RAM for CPU inference. "
                "Disable [rl_training] in config.ini."
            )
            self.logger.error(self._error)
        except OSError as e:
            if "401" in str(e) or "gated" in str(e).lower():
                self._error = (
                    f"Model {self.MODEL_ID} requires HuggingFace authentication. "
                    "Run: huggingface-cli login, accept the license at "
                    f"https://huggingface.co/{self.MODEL_ID}, then restart."
                )
            else:
                self._error = f"Failed to load model: {e}"
            self.logger.error(self._error)
        except Exception as e:  # noqa: BLE001
            self._error = f"Failed to load model: {e}"
            self.logger.error(self._error)

    def _resolve_device(self) -> str:
        """Determine which device to use based on config."""
        configured = self._config.RL_TRAINING_DEVICE
        if configured == "cpu":
            return "cpu"
        if configured == "cuda":
            return "cuda"

        # auto: prefer CUDA, fall back to CPU
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def generate_decision(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a trading decision from prompts.

        Args:
            system_prompt: The system instructions.
            user_prompt: The market data and analysis context.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If model is not loaded/enabled.
        """
        if not self.enabled or self._model is None:
            raise RuntimeError(
                f"RL policy model not available: {self._error}"
            )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self._tokenizer([text], return_tensors="pt")
            if self._model.device.type != "cpu":
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True,
            )
            return response.strip()

        except Exception as e:
            self.logger.error("RL inference failed: %s", e)
            raise RuntimeError(f"RL inference failed: {e}") from e

    @property
    def status(self) -> str:
        """Human-readable status for dashboard/logs."""
        if not self._config.RL_TRAINING_ENABLED:
            return "DISABLED (config)"
        if self.enabled:
            return "ACTIVE"
        return f"ERROR: {self._error}"
