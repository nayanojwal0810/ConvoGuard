"""Artifact persistence and bundling utilities for ConvoGuard model runs."""

from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from convoguard.config import serialize_run_config

logger = logging.getLogger(__name__)


def save_model_artifacts(
    model: Any,
    tokenizer: Any,
    checkpoint_dir: str | Path,
    config_dict: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Save complete, reproducible model checkpoint artifacts.

    Persists:
    1. Model weights (`model.pt` via `torch.save(model.state_dict(), ...)`).
    2. Tokenizer files via `tokenizer.save_pretrained(checkpoint_dir)`.
    3. Model architecture configuration via `model.config.save_pretrained(checkpoint_dir)`.
    4. Run configuration and runtime environment metadata (`run_config.json`).
    5. Standalone validation metrics (`val_metrics.json`).

    Args:
        model: PyTorch model or object with `state_dict()` and `config.save_pretrained()`.
        tokenizer: PreTrainedTokenizer or mock object with `save_pretrained()`.
        checkpoint_dir: Output directory path where checkpoint artifacts will be saved.
        config_dict: Hyperparameter configuration dictionary (supports `Path` instances).
        metrics: Dictionary of validation metrics to persist.

    Returns:
        Resolved `Path` to the checkpoint directory.
    """
    ckpt_dir = Path(checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Weights
    try:
        import torch

        if hasattr(model, "state_dict") and callable(model.state_dict):
            torch.save(model.state_dict(), ckpt_dir / "model.pt")
        elif hasattr(model, "save_pretrained") and callable(model.save_pretrained):
            model.save_pretrained(ckpt_dir)
        else:
            raise TypeError(
                "Model must provide state_dict() or save_pretrained() to persist weights."
            )
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for save_model_artifacts. Install with `pip install convoguard[bert]`."
        ) from exc

    # 2. Tokenizer
    if hasattr(tokenizer, "save_pretrained") and callable(tokenizer.save_pretrained):
        tokenizer.save_pretrained(ckpt_dir)
    else:
        logger.warning("Tokenizer does not implement save_pretrained(); skipping tokenizer files.")

    # 3. Model Configuration
    if (
        hasattr(model, "config")
        and hasattr(model.config, "save_pretrained")
        and callable(model.config.save_pretrained)
    ):
        model.config.save_pretrained(ckpt_dir)
    elif hasattr(model, "save_pretrained") and callable(model.save_pretrained):
        model.save_pretrained(ckpt_dir)

    # 4. Run Metadata & Library Versions
    try:
        import transformers

        transformers_version = getattr(transformers, "__version__", "unknown")
    except ImportError:
        transformers_version = "not_installed"

    run_metadata = {
        "hyperparameters": serialize_run_config(config_dict),
        "library_versions": {
            "torch": getattr(torch, "__version__", "unknown"),
            "transformers": transformers_version,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "best_validation_metrics": metrics,
    }
    with open(ckpt_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    # 5. Standalone Validation Metrics
    with open(ckpt_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Complete reproducible artifacts saved to %s", ckpt_dir)
    return ckpt_dir


def package_artifacts(
    output_dir: str | Path,
    zip_basename: str = "convoguard_phase2_bert_outputs",
) -> Path:
    """Package output directory into a ZIP archive for export/submission."""
    out_dir = Path(output_dir).resolve()
    zip_path = shutil.make_archive(
        base_name=str(out_dir.parent / zip_basename),
        format="zip",
        root_dir=out_dir,
    )
    return Path(zip_path).resolve()
