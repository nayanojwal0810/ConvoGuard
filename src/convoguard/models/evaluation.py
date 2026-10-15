"""Evaluation metrics, threshold selection, and forecast-horizon calculation.

Provides rigorous, publication-grade evaluation routines:
- Classification metrics: Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, PR-AUC.
- Threshold selection on validation data with explicit provenance.
- Forecast-horizon analysis: earliest detection, horizon distribution, early-warning rate,
  and negative conversation false-alert rate.
- Aggregation of dynamic prefix predictions to conversation-level outcomes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate comprehensive binary classification metrics for a given decision threshold.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for positive class (label 1).
        threshold: Decision threshold for converting probabilities to binary predictions.

    Returns:
        Dictionary of computed metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    y_pred = (y_prob_arr >= threshold).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0))

    # ROC-AUC and PR-AUC require at least one sample from each class
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "macro_f1": round(macro_f1, 6),
        "roc_auc": round(roc_auc, 6) if not np.isnan(roc_auc) else None,
        "pr_auc": round(pr_auc, 6) if not np.isnan(pr_auc) else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "total": len(y_true_arr)},
    }


def find_optimal_threshold(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    metric: str = "f1",
    num_candidates: int = 181,
) -> float:
    """Find the probability threshold that maximizes a target metric on validation data.

    Args:
        y_true: Validation ground truth binary labels.
        y_prob: Validation predicted probabilities.
        metric: Metric to optimize ('f1' or 'macro_f1').
        num_candidates: Number of candidate thresholds between 0.05 and 0.95.

    Returns:
        Optimal float threshold.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.05, 0.95, num_candidates)
    best_score = -1.0
    best_th = 0.5

    for th in thresholds:
        y_pred = (y_prob_arr >= th).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true_arr, y_pred, average="macro", zero_division=0)
        else:
            score = f1_score(y_true_arr, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_th = float(th)

    return round(best_th, 4)


def evaluate_predictions(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    fixed_threshold: float = 0.5,
    select_optimal: bool = True,
) -> dict[str, Any]:
    """Compute metrics at default fixed threshold and optionally at validation-selected threshold."""
    fixed_metrics = compute_binary_metrics(y_true, y_prob, threshold=fixed_threshold)
    result: dict[str, Any] = {
        "fixed_threshold_0.5": fixed_metrics,
    }

    if select_optimal:
        opt_th = find_optimal_threshold(y_true, y_prob, metric="f1")
        opt_metrics = compute_binary_metrics(y_true, y_prob, threshold=opt_th)
        result["validation_selected_threshold"] = {
            "selected_threshold": opt_th,
            "selection_criterion": "max_f1_on_validation",
            "note": "Optimized on validation data; not an independent estimate of generalization.",
            "metrics": opt_metrics,
        }

    return result


def compute_forecast_horizon_metrics(
    dynamic_df: pd.DataFrame,
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate forecast horizon and early alert metrics from dynamic prefix predictions.

    Turn Index Convention & Horizon Formula:
    - For a conversation with N total turns (0, 1, ..., N-1), the terminal target turn
      (turn N-1) is excluded from all inputs.
    - The observed history contains turns 0..N-2 (N-1 available turns).
    - A dynamic prefix of length k (1 <= k <= N-1) consists of turns 0..k-1.
    - The horizon is the number of dialogue turns remaining until the excluded terminal turn:
        horizon = (N - 1) - (k - 1) = N - k.
    - When k = N - 1 (the full pre-target history is observed), horizon = 1.
    - When k = 1 (only the first turn is observed), horizon = N - 1.
    - An early warning occurs when the model first alerts at k < N - 1 (horizon > 1).

    Args:
        dynamic_df: DataFrame containing at least:
            ['conversation_id', 'prefix_length', 'total_conversation_turns', 'label']
        y_prob: Model predicted probabilities corresponding to rows in dynamic_df.
        threshold: Decision threshold for issuing a derailment alert.

    Returns:
        Structured dictionary of forecast-horizon metrics.
    """
    df = dynamic_df.copy()
    df["predicted_prob"] = np.asarray(y_prob, dtype=float)
    df["is_alert"] = df["predicted_prob"] >= threshold

    positive_conv_ids = df[df["label"] == 1]["conversation_id"].unique()
    negative_conv_ids = df[df["label"] == 0]["conversation_id"].unique()

    total_positives = len(positive_conv_ids)
    total_negatives = len(negative_conv_ids)

    # Positive conversation analysis
    alerted_positives = 0
    horizons: list[int] = []
    warned_early_count = 0  # Warned before the latest possible prefix (horizon > 1)

    for conv_id in positive_conv_ids:
        conv_prefixes = df[df["conversation_id"] == conv_id].sort_values("prefix_length")
        alerts = conv_prefixes[conv_prefixes["is_alert"]]

        if not alerts.empty:
            alerted_positives += 1
            earliest_alert = alerts.iloc[0]
            k = int(earliest_alert["prefix_length"])
            n = int(earliest_alert["total_conversation_turns"])
            h = n - k
            horizons.append(h)
            if h > 1:
                warned_early_count += 1

    # Negative conversation analysis (false alerts)
    false_alert_negatives = 0
    for conv_id in negative_conv_ids:
        conv_prefixes = df[df["conversation_id"] == conv_id]
        if (conv_prefixes["is_alert"]).any():
            false_alert_negatives += 1

    horizon_dist: dict[int, int] = {}
    for h in sorted(horizons):
        horizon_dist[h] = horizon_dist.get(h, 0) + 1

    mean_h = float(np.mean(horizons)) if horizons else 0.0
    median_h = float(np.median(horizons)) if horizons else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "formula_definition": (
            "For a conversation with N turns and prefix length k (1 <= k <= N-1), "
            "horizon = N - k turns remaining until the excluded terminal target turn. "
            "When k = N - 1 (full pre-target history), horizon = 1. "
            "The terminal attack message is excluded from all inputs and is never an observed warning turn."
        ),
        "positive_conversations_total": total_positives,
        "negative_conversations_total": total_negatives,
        "alerted_positive_conversations": alerted_positives,
        "recall_at_conversation_level": round(alerted_positives / total_positives, 6)
        if total_positives
        else 0.0,
        "mean_horizon": round(mean_h, 3),
        "median_horizon": round(median_h, 3),
        "min_horizon": int(np.min(horizons)) if horizons else 0,
        "max_horizon": int(np.max(horizons)) if horizons else 0,
        "horizon_distribution": horizon_dist,
        "warned_before_latest_prefix_count": warned_early_count,
        "warned_before_latest_prefix_proportion": round(warned_early_count / total_positives, 6)
        if total_positives
        else 0.0,
        "false_alert_conversations_count": false_alert_negatives,
        "false_alert_rate": round(false_alert_negatives / total_negatives, 6)
        if total_negatives
        else 0.0,
    }


def aggregate_dynamic_predictions(
    dynamic_df: pd.DataFrame,
    y_prob: np.ndarray | list[float],
) -> pd.DataFrame:
    """Aggregate dynamic prefix predictions into conversation-level predictions.

    Provides three aggregation strategies:
    1. full_prefix: Prediction using all available pre-target turns (k = N - 1).
    2. max_pooling: Maximum predicted probability across all prefixes of the conversation.
    3. mean_pooling: Average predicted probability across all prefixes of the conversation.

    Returns:
        DataFrame indexed by conversation_id with aggregated probabilities and ground truth label.
    """
    df = dynamic_df.copy()
    df["y_prob"] = np.asarray(y_prob, dtype=float)

    records: list[dict[str, Any]] = []
    for conv_id, group in df.groupby("conversation_id", sort=False):
        label = int(group.iloc[0]["label"])
        full_prefix_row = group[group["is_full_history"]]
        full_prefix_prob = (
            float(full_prefix_row.iloc[0]["y_prob"])
            if not full_prefix_row.empty
            else float(group.iloc[-1]["y_prob"])
        )
        max_prob = float(group["y_prob"].max())
        mean_prob = float(group["y_prob"].mean())

        records.append(
            {
                "conversation_id": conv_id,
                "label": label,
                "full_prefix_prob": full_prefix_prob,
                "max_prob": max_prob,
                "mean_prob": mean_prob,
                "num_prefixes": len(group),
            }
        )

    return pd.DataFrame(records)
