#!/usr/bin/env python3
"""Run and evaluate CPU baselines on train and validation splits.

Models evaluated:
1. ClassPriorBaseline: Majority / empirical training class prior.
2. StaticTfidfBaseline: Word-and-character TF-IDF + Logistic Regression on full pre-target prefixes.
3. DynamicTfidfBaseline: Word-and-character TF-IDF + Logistic Regression on successive prefix histories.

Strictly follows Phase 2 evaluation protocol:
- Fits all feature vectorizers and classifiers on training data only.
- Evaluates on validation data only (never touches the official test split).
- Reports fixed threshold (0.5) and validation-selected threshold metrics with explicit provenance.
- Evaluates forecast-horizon dynamics and conversation-level aggregations.
- Saves reproducible aggregate metrics as a small JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src is on sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from convoguard.config import get_config  # noqa: E402
from convoguard.data.forecasting import (  # noqa: E402
    assert_forecasting_contract,
    build_dynamic_examples,
    build_static_examples,
)
from convoguard.data.loader import load_split  # noqa: E402
from convoguard.models.baselines import (  # noqa: E402
    DynamicTfidfBaseline,
    PriorBaseline,
    StaticTfidfBaseline,
)
from convoguard.models.evaluation import (  # noqa: E402
    aggregate_dynamic_predictions,
    compute_binary_metrics,
    compute_forecast_horizon_metrics,
    evaluate_predictions,
    find_optimal_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convoguard.run_cpu_baselines")


def run_baselines(
    data_root: str | Path,
    output_path: str | Path = "artifacts/benchmarks/cpu_baselines_metrics.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Execute complete CPU baseline training and validation evaluation."""
    root = Path(data_root).resolve()
    logger.info("Starting ConvoGuard CPU baselines evaluation from data root: %s", root)

    train_path = root / "train_conversations.parquet"
    val_path = root / "val_conversations.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Training split file not found at: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation split file not found at: {val_path}")

    # Explicit safety assertion: ensure we never touch test split
    test_path = root / "test_conversations.parquet"
    logger.info(
        "Protocol enforcement: Loading train (%s) and val (%s). Test split is ignored (%s exists=%s).",
        train_path.name,
        val_path.name,
        test_path.name,
        test_path.exists(),
    )

    df_train = load_split(train_path, validate=True)
    df_val = load_split(val_path, validate=True)

    logger.info("Loaded train rows: %d, val rows: %d", len(df_train), len(df_val))

    # Build static examples
    logger.info("Building static pre-target examples...")
    static_train = build_static_examples(df_train)
    static_val = build_static_examples(df_val)
    logger.info(
        "Static examples: train=%d convs (derailment rate=%.2f), val=%d convs (derailment rate=%.2f)",
        len(static_train),
        static_train["label"].mean(),
        len(static_val),
        static_val["label"].mean(),
    )

    # Build dynamic examples
    logger.info("Building dynamic prefix examples...")
    dynamic_train = build_dynamic_examples(df_train)
    dynamic_val = build_dynamic_examples(df_val)
    logger.info(
        "Dynamic examples: train=%d prefixes, val=%d prefixes",
        len(dynamic_train),
        len(dynamic_val),
    )

    # Contract Verification: structural validation against raw data
    logger.info("Executing rigorous forecasting contract assertions on train and val...")
    assert_forecasting_contract(df_train, static_train, dynamic_train)
    assert_forecasting_contract(df_val, static_val, dynamic_val)
    logger.info("[OK] Forecasting contract structurally verified for both train and val splits.")

    results: dict[str, Any] = {
        "metadata": {
            "task": "conversational_derailment_early_forecasting",
            "splits_evaluated": ["train", "val"],
            "test_split_accessed": False,
            "random_seed": seed,
            "train_conversations": len(static_train),
            "val_conversations": len(static_val),
            "train_dynamic_prefixes": len(dynamic_train),
            "val_dynamic_prefixes": len(dynamic_val),
        },
        "baselines": {},
    }

    # ---------------------------------------------------------
    # Baseline 1: Class-Prior / Majority Baseline
    # ---------------------------------------------------------
    logger.info("Fitting Baseline 1: Class-Prior Baseline...")
    prior_model = PriorBaseline().fit(static_train["text"].to_numpy(), static_train["label"].to_numpy())
    prior_val_probs = prior_model.predict_proba(static_val["text"].to_numpy())[:, 1]
    prior_metrics = compute_binary_metrics(static_val["label"].to_numpy(), prior_val_probs, threshold=0.5)

    results["baselines"]["class_prior"] = {
        "model_type": "empirical_class_prior",
        "training_prior_probability": prior_model.prior_prob_,
        "validation_metrics": prior_metrics,
    }

    # ---------------------------------------------------------
    # Baseline 2: Static TF-IDF + Logistic Regression
    # ---------------------------------------------------------
    logger.info("Fitting Baseline 2: Static TF-IDF + Logistic Regression...")
    static_model = StaticTfidfBaseline(random_state=seed)
    static_model.fit(static_train["text"].to_numpy(), static_train["label"].to_numpy())
    static_val_probs = static_model.predict_proba(static_val["text"].to_numpy())[:, 1]
    static_eval = evaluate_predictions(static_val["label"].to_numpy(), static_val_probs, fixed_threshold=0.5)

    results["baselines"]["static_tfidf_lr"] = {
        "model_type": "word_and_char_tfidf_logistic_regression",
        "training_scope": "static_full_pre_target_history",
        "evaluation": static_eval,
    }

    # ---------------------------------------------------------
    # Baseline 3: Dynamic TF-IDF + Logistic Regression
    # ---------------------------------------------------------
    logger.info("Fitting Baseline 3: Dynamic TF-IDF + Logistic Regression on prefixes...")
    dynamic_model = DynamicTfidfBaseline(random_state=seed)
    dynamic_model.fit(dynamic_train["text"].to_numpy(), dynamic_train["label"].to_numpy())
    dynamic_val_probs = dynamic_model.predict_proba(dynamic_val["text"].to_numpy())[:, 1]

    # Prefix-level metrics
    prefix_eval = evaluate_predictions(dynamic_val["label"].to_numpy(), dynamic_val_probs, fixed_threshold=0.5)

    # Conversation aggregation
    conv_preds = aggregate_dynamic_predictions(dynamic_val, dynamic_val_probs)

    # Full-prefix conversation-level evaluation (prediction using all observed turns)
    full_prefix_eval = evaluate_predictions(
        conv_preds["label"].to_numpy(),
        conv_preds["full_prefix_prob"].to_numpy(),
        fixed_threshold=0.5,
    )

    # Max-pooling conversation-level evaluation
    max_pooling_eval = evaluate_predictions(
        conv_preds["label"].to_numpy(),
        conv_preds["max_prob"].to_numpy(),
        fixed_threshold=0.5,
    )

    # Threshold selection using validation full-history conversation predictions only
    # Primary: Macro-F1 optimization
    opt_th_dynamic_macro = find_optimal_threshold(
        conv_preds["label"].to_numpy(), conv_preds["full_prefix_prob"].to_numpy(), metric="macro_f1"
    )
    horizon_metrics_05 = compute_forecast_horizon_metrics(dynamic_val, dynamic_val_probs, threshold=0.5)
    horizon_metrics_macro = compute_forecast_horizon_metrics(
        dynamic_val, dynamic_val_probs, threshold=opt_th_dynamic_macro
    )

    # Secondary diagnostic: Positive-F1 optimization (recall-heavy diagnostic)
    opt_th_dynamic_f1 = find_optimal_threshold(
        conv_preds["label"].to_numpy(), conv_preds["full_prefix_prob"].to_numpy(), metric="f1"
    )
    horizon_metrics_f1 = compute_forecast_horizon_metrics(
        dynamic_val, dynamic_val_probs, threshold=opt_th_dynamic_f1
    )

    results["baselines"]["dynamic_tfidf_lr"] = {
        "model_type": "word_and_char_tfidf_logistic_regression",
        "training_scope": "dynamic_successive_prefixes",
        "prefix_level_evaluation": prefix_eval,
        "conversation_level_full_prefix_evaluation": full_prefix_eval,
        "conversation_level_max_pooling_evaluation": max_pooling_eval,
        "forecast_horizon_evaluation_fixed_th_0.5": horizon_metrics_05,
        "forecast_horizon_evaluation_val_selected_th": {
            "selected_threshold": opt_th_dynamic_macro,
            "selection_criterion": "max_macro_f1_on_validation_full_history",
            "selection_source": "validation_full_history_conversation_predictions",
            "note": "Optimized on validation data; not an independent estimate of generalization.",
            "metrics": horizon_metrics_macro,
        },
        "forecast_horizon_evaluation_positive_f1_diagnostic": {
            "selected_threshold": opt_th_dynamic_f1,
            "selection_criterion": "max_positive_f1_on_validation_full_history",
            "selection_source": "validation_full_history_conversation_predictions",
            "note": "Recall-heavy diagnostic optimized for positive-class F1 on validation; not an independent estimate of generalization.",
            "metrics": horizon_metrics_f1,
        },
    }

    # Save output artifact
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved baseline metrics to: %s", out_file)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ConvoGuard CPU baselines on train/val data.")
    config = get_config()
    default_data = str(config.resolve_data_root())

    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data,
        help="Path to dataset directory containing train_conversations.parquet and val_conversations.parquet.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/benchmarks/cpu_baselines_metrics.json",
        help="Target output JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    results = run_baselines(data_root=args.data_root, output_path=args.output, seed=args.seed)

    print("\n" + "=" * 70)
    print("CONVOGUARD PHASE 2 — CPU BASELINE VALIDATION RESULTS")
    print("=" * 70)

    # Print Class Prior
    cp = results["baselines"]["class_prior"]["validation_metrics"]
    print("\n1. Class-Prior Baseline (th=0.5):")
    print(f"   Accuracy: {cp['accuracy']:.4f} | Macro-F1: {cp['macro_f1']:.4f} | Pos-F1: {cp['f1']:.4f}")
    print(f"   Positive Alert Coverage: {cp['positive_alert_coverage']*100:.1f}% | Negative False-Alert Rate: {cp['negative_false_alert_rate']*100:.1f}%")
    print(f"   Confusion Matrix: {cp['confusion_matrix']}")

    # Print Static TF-IDF
    st_fixed = results["baselines"]["static_tfidf_lr"]["evaluation"]["fixed_threshold_0.5"]
    st_opt = results["baselines"]["static_tfidf_lr"]["evaluation"]["validation_selected_threshold"]
    st_diag = results["baselines"]["static_tfidf_lr"]["evaluation"].get("validation_selected_threshold_positive_f1_diagnostic")
    print("\n2. Static TF-IDF + Logistic Regression (Full Pre-Target History):")
    print(f"   [Fixed th=0.5] Acc: {st_fixed['accuracy']:.4f} | Pos-F1: {st_fixed['f1']:.4f} | Macro-F1: {st_fixed['macro_f1']:.4f}")
    print(f"                  ROC-AUC: {st_fixed['roc_auc']:.4f} | PR-AUC: {st_fixed['pr_auc']:.4f}")
    print(f"                  Positive Alert Coverage: {st_fixed['positive_alert_coverage']*100:.1f}% | Negative False-Alert Rate: {st_fixed['negative_false_alert_rate']*100:.1f}%")
    print(f"                  Confusion Matrix: {st_fixed['confusion_matrix']}")
    print(f"   [Val-Tuned Macro-F1 th={st_opt['selected_threshold']:.4f}] Acc: {st_opt['metrics']['accuracy']:.4f} | Pos-F1: {st_opt['metrics']['f1']:.4f} | Macro-F1: {st_opt['metrics']['macro_f1']:.4f}")
    print(f"                  Positive Alert Coverage: {st_opt['metrics']['positive_alert_coverage']*100:.1f}% | Negative False-Alert Rate: {st_opt['metrics']['negative_false_alert_rate']*100:.1f}%")
    print("                  (Note: Threshold optimized on validation Macro-F1; diagnostic only, not an independent generalization estimate)")
    if st_diag:
        print(f"   [Val-Tuned Pos-F1 Diagnostic th={st_diag['selected_threshold']:.4f}] Acc: {st_diag['metrics']['accuracy']:.4f} | Pos-F1: {st_diag['metrics']['f1']:.4f} | Macro-F1: {st_diag['metrics']['macro_f1']:.4f}")
        print(f"                  Positive Alert Coverage: {st_diag['metrics']['positive_alert_coverage']*100:.1f}% | Negative False-Alert Rate: {st_diag['metrics']['negative_false_alert_rate']*100:.1f}%")
        print("                  (Note: Recall-heavy diagnostic optimized for positive-class F1 on validation)")

    # Print Dynamic TF-IDF
    dyn_full = results["baselines"]["dynamic_tfidf_lr"]["conversation_level_full_prefix_evaluation"]["fixed_threshold_0.5"]
    dyn_max = results["baselines"]["dynamic_tfidf_lr"]["conversation_level_max_pooling_evaluation"]["fixed_threshold_0.5"]
    fh_05 = results["baselines"]["dynamic_tfidf_lr"]["forecast_horizon_evaluation_fixed_th_0.5"]
    fh_opt_data = results["baselines"]["dynamic_tfidf_lr"]["forecast_horizon_evaluation_val_selected_th"]
    opt_th = fh_opt_data["selected_threshold"]
    fh_opt = fh_opt_data["metrics"]
    fh_diag_data = results["baselines"]["dynamic_tfidf_lr"].get("forecast_horizon_evaluation_positive_f1_diagnostic")

    print("\n3. Dynamic TF-IDF + Logistic Regression (Prefix Trained):")
    print(f"   [Full-Prefix th=0.5] Acc: {dyn_full['accuracy']:.4f} | Pos-F1: {dyn_full['f1']:.4f} | Macro-F1: {dyn_full['macro_f1']:.4f}")
    print(f"                        ROC-AUC: {dyn_full['roc_auc']:.4f} | PR-AUC: {dyn_full['pr_auc']:.4f}")
    print(f"                        Positive Alert Coverage: {dyn_full['positive_alert_coverage']*100:.1f}% | Negative False-Alert Rate: {dyn_full['negative_false_alert_rate']*100:.1f}%")
    print(f"   [Max-Pooling th=0.5] Acc: {dyn_max['accuracy']:.4f} | Pos-F1: {dyn_max['f1']:.4f} | Macro-F1: {dyn_max['macro_f1']:.4f}")
    print(f"                        Positive Alert Coverage: {dyn_max['positive_alert_coverage']*100:.1f}% | Negative False-Alert Rate: {dyn_max['negative_false_alert_rate']*100:.1f}%")

    print("\n4. Forecast-Horizon Analysis at Fixed Decision Threshold (th=0.5):")
    print(f"   Mean Horizon: {fh_05['mean_horizon']:.2f} turns | Median: {fh_05['median_horizon']:.1f} turns (range: {fh_05['min_horizon']}-{fh_05['max_horizon']})")
    print(f"   Positive Alert Coverage: {fh_05['alerted_positive_conversations']}/{fh_05['positive_conversations_total']} ({fh_05['recall_at_conversation_level']*100:.1f}%)")
    print(f"   Warned Before Terminal Prefix (H > 1): {fh_05['warned_before_latest_prefix_count']}/{fh_05['positive_conversations_total']} ({fh_05['warned_before_latest_prefix_proportion']*100:.1f}%)")
    print(f"   Negative Conversation False-Alert Rate: {fh_05['false_alert_conversations_count']}/{fh_05['negative_conversations_total']} ({fh_05['false_alert_rate']*100:.1f}%)")
    print(f"   Horizon Distribution (turns ahead -> count): {fh_05['horizon_distribution']}")

    print(f"\n5. Forecast-Horizon Analysis at Validation-Selected Macro-F1 Threshold (th={opt_th:.4f}):")
    print("   [NOTE: Primary criterion is max Macro-F1 on validation full-history conversation predictions. Validation diagnostic only, not an independent generalization estimate.]")
    print(f"   Mean Horizon: {fh_opt['mean_horizon']:.2f} turns | Median: {fh_opt['median_horizon']:.1f} turns (range: {fh_opt['min_horizon']}-{fh_opt['max_horizon']})")
    print(f"   Positive Alert Coverage: {fh_opt['alerted_positive_conversations']}/{fh_opt['positive_conversations_total']} ({fh_opt['recall_at_conversation_level']*100:.1f}%)")
    print(f"   Warned Before Terminal Prefix (H > 1): {fh_opt['warned_before_latest_prefix_count']}/{fh_opt['positive_conversations_total']} ({fh_opt['warned_before_latest_prefix_proportion']*100:.1f}%)")
    print(f"   Negative Conversation False-Alert Rate: {fh_opt['false_alert_conversations_count']}/{fh_opt['negative_conversations_total']} ({fh_opt['false_alert_rate']*100:.1f}%)")
    print(f"   Horizon Distribution (turns ahead -> count): {fh_opt['horizon_distribution']}")

    if fh_diag_data:
        diag_th = fh_diag_data["selected_threshold"]
        fh_diag = fh_diag_data["metrics"]
        print(f"\n6. (Diagnostic) Forecast-Horizon Analysis at Validation-Selected Positive-F1 Threshold (th={diag_th:.4f}):")
        print("   [NOTE: Recall-heavy diagnostic optimizing positive-class F1 on validation full-history. Not an independent generalization estimate.]")
        print(f"   Mean Horizon: {fh_diag['mean_horizon']:.2f} turns | Median: {fh_diag['median_horizon']:.1f} turns (range: {fh_diag['min_horizon']}-{fh_diag['max_horizon']})")
        print(f"   Positive Alert Coverage: {fh_diag['alerted_positive_conversations']}/{fh_diag['positive_conversations_total']} ({fh_diag['recall_at_conversation_level']*100:.1f}%)")
        print(f"   Warned Before Terminal Prefix (H > 1): {fh_diag['warned_before_latest_prefix_count']}/{fh_diag['positive_conversations_total']} ({fh_diag['warned_before_latest_prefix_proportion']*100:.1f}%)")
        print(f"   Negative Conversation False-Alert Rate: {fh_diag['false_alert_conversations_count']}/{fh_diag['negative_conversations_total']} ({fh_diag['false_alert_rate']*100:.1f}%)")
        print(f"   Horizon Distribution (turns ahead -> count): {fh_diag['horizon_distribution']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
