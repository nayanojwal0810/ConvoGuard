"""Public model APIs with BERT dependencies loaded only when requested.

CPU baselines, metrics, and artifact utilities remain importable without the optional
PyTorch/Transformers dependency group. BERT-specific names are exposed lazily below.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from convoguard.models.artifacts import package_artifacts, save_model_artifacts
from convoguard.models.baselines import (
    DynamicTfidfBaseline,
    PriorBaseline,
    StaticTfidfBaseline,
    build_tfidf_pipeline,
)
from convoguard.models.evaluation import (
    aggregate_dynamic_predictions,
    compute_binary_metrics,
    compute_forecast_horizon_metrics,
    evaluate_predictions,
    find_optimal_threshold,
)

_BERT_EXPORTS = frozenset(
    {
        "BertForecastingClassifier",
        "BertForecastingDataset",
        "pad_bert_collate",
        "turn_preserving_truncate",
    }
)

__all__ = [
    "PriorBaseline",
    "StaticTfidfBaseline",
    "DynamicTfidfBaseline",
    "build_tfidf_pipeline",
    "compute_binary_metrics",
    "find_optimal_threshold",
    "evaluate_predictions",
    "compute_forecast_horizon_metrics",
    "aggregate_dynamic_predictions",
    "BertForecastingClassifier",
    "BertForecastingDataset",
    "pad_bert_collate",
    "turn_preserving_truncate",
    "save_model_artifacts",
    "package_artifacts",
]


def __getattr__(name: str) -> Any:
    """Load BERT exports on demand so CPU-only imports need no GPU dependencies."""
    if name not in _BERT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        bert_module = import_module(".bert", package=__name__)
    except ImportError as exc:
        raise ImportError(
            "ConvoGuard BERT components require the optional PyTorch and Transformers "
            "dependencies. Install them with pip install -e '.[bert]'."
        ) from exc

    return getattr(bert_module, name)
