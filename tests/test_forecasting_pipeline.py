"""Automated tests for ConvoGuard forecasting dataset construction, baselines, and evaluation.

Tests all Task G requirements using small synthetic fixtures:
- Target-turn exclusion from static examples.
- Target-turn exclusion from every dynamic prefix.
- Sorting and contiguous index verification.
- Mapping between has_attack and terminal message label.
- Deterministic prefix generation.
- Dynamic prediction aggregation by conversation.
- Forecast-horizon arithmetic and false-alert counting.
- Handling of short conversations and finite probability outputs.
- Turn-preserving recursive truncation behavior.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from convoguard.config import serialize_run_config
from convoguard.data.forecasting import (
    assert_forecasting_contract,
    build_dynamic_examples,
    build_static_examples,
    validate_forecasting_conversation,
)
from convoguard.data.loader import DatasetValidationError
from convoguard.models.artifacts import (
    package_artifacts,
    save_model_artifacts,
)
from convoguard.models.baselines import PriorBaseline, StaticTfidfBaseline
from convoguard.models.bert import turn_preserving_truncate
from convoguard.models.evaluation import (
    aggregate_dynamic_predictions,
    compute_forecast_horizon_metrics,
    evaluate_predictions,
    find_optimal_threshold,
)


class MockTokenizer:
    """Lightweight mock tokenizer for testing turn-preserving truncation without downloading weights."""

    def __init__(self):
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Return mock token IDs based on word split
        words = text.strip().split()
        return [hash(w) % 1000 + 200 for w in words]


@pytest.fixture
def sample_conversations_df() -> pd.DataFrame:
    """Fixture with 1 clean conversation (4 turns) and 1 derailed conversation (5 turns)."""
    return pd.DataFrame(
        [
            # Clean conversation: c_clean (N=4)
            {
                "conversation_id": "c_clean",
                "split": "train",
                "has_attack": False,
                "is_comment_attack": False,
                "utterance_id": "u_c0",
                "text": "First turn clean text",
                "speaker": "Alice",
                "timestamp": 100.0,
                "total_utterances": 4,
                "utterance_index": 0,
            },
            {
                "conversation_id": "c_clean",
                "split": "train",
                "has_attack": False,
                "is_comment_attack": False,
                "utterance_id": "u_c1",
                "text": "Second turn clean text",
                "speaker": "Bob",
                "timestamp": 110.0,
                "total_utterances": 4,
                "utterance_index": 1,
            },
            {
                "conversation_id": "c_clean",
                "split": "train",
                "has_attack": False,
                "is_comment_attack": False,
                "utterance_id": "u_c2",
                "text": "Third turn clean text",
                "speaker": "Alice",
                "timestamp": 120.0,
                "total_utterances": 4,
                "utterance_index": 2,
            },
            {
                "conversation_id": "c_clean",
                "split": "train",
                "has_attack": False,
                "is_comment_attack": False,
                "utterance_id": "u_c3",
                "text": "Final terminal clean message",
                "speaker": "Bob",
                "timestamp": 130.0,
                "total_utterances": 4,
                "utterance_index": 3,
            },
            # Derailed conversation: c_derail (N=5, terminal turn is attack)
            {
                "conversation_id": "c_derail",
                "split": "train",
                "has_attack": True,
                "is_comment_attack": False,
                "utterance_id": "u_d0",
                "text": "Opening remark before dispute",
                "speaker": "Charlie",
                "timestamp": 200.0,
                "total_utterances": 5,
                "utterance_index": 0,
            },
            {
                "conversation_id": "c_derail",
                "split": "train",
                "has_attack": True,
                "is_comment_attack": False,
                "utterance_id": "u_d1",
                "text": "Disagreeing reply",
                "speaker": "Dave",
                "timestamp": 210.0,
                "total_utterances": 5,
                "utterance_index": 1,
            },
            {
                "conversation_id": "c_derail",
                "split": "train",
                "has_attack": True,
                "is_comment_attack": False,
                "utterance_id": "u_d2",
                "text": "Escalating hostility reply",
                "speaker": "Charlie",
                "timestamp": 220.0,
                "total_utterances": 5,
                "utterance_index": 2,
            },
            {
                "conversation_id": "c_derail",
                "split": "train",
                "has_attack": True,
                "is_comment_attack": False,
                "utterance_id": "u_d3",
                "text": "Final warning before blowup",
                "speaker": "Dave",
                "timestamp": 230.0,
                "total_utterances": 5,
                "utterance_index": 3,
            },
            {
                "conversation_id": "c_derail",
                "split": "train",
                "has_attack": True,
                "is_comment_attack": True,
                "utterance_id": "u_d4",
                "text": "TARGET ATTACK COMMENT THAT MUST BE EXCLUDED FROM FORECASTING",
                "speaker": "Charlie",
                "timestamp": 240.0,
                "total_utterances": 5,
                "utterance_index": 4,
            },
        ]
    )


# 1. Target-turn exclusion from static examples
def test_static_examples_exclude_target_turn(sample_conversations_df: pd.DataFrame):
    static_df = build_static_examples(sample_conversations_df)
    assert len(static_df) == 2

    target_attack_text = "TARGET ATTACK COMMENT THAT MUST BE EXCLUDED FROM FORECASTING"
    terminal_clean_text = "Final terminal clean message"

    for _, row in static_df.iterrows():
        assert target_attack_text not in row["text"]
        assert target_attack_text not in row["turns"]
        assert terminal_clean_text not in row["text"]
        assert terminal_clean_text not in row["turns"]

    # Verify lengths
    clean_row = static_df[static_df["conversation_id"] == "c_clean"].iloc[0]
    assert clean_row["prefix_length"] == 3  # 4 - 1
    assert clean_row["total_conversation_turns"] == 4
    assert clean_row["label"] == 0

    derail_row = static_df[static_df["conversation_id"] == "c_derail"].iloc[0]
    assert derail_row["prefix_length"] == 4  # 5 - 1
    assert derail_row["total_conversation_turns"] == 5
    assert derail_row["label"] == 1


# 2. Target-turn exclusion from every dynamic prefix
def test_dynamic_examples_exclude_target_turn(sample_conversations_df: pd.DataFrame):
    dynamic_df = build_dynamic_examples(sample_conversations_df)

    # c_clean has N=4, so 3 prefixes (k=1, 2, 3)
    # c_derail has N=5, so 4 prefixes (k=1, 2, 3, 4)
    # Total = 7 dynamic examples
    assert len(dynamic_df) == 7

    target_attack_text = "TARGET ATTACK COMMENT THAT MUST BE EXCLUDED FROM FORECASTING"
    terminal_clean_text = "Final terminal clean message"

    for _, row in dynamic_df.iterrows():
        assert target_attack_text not in row["text"]
        assert target_attack_text not in row["turns"]
        assert terminal_clean_text not in row["text"]
        assert terminal_clean_text not in row["turns"]

    # Check prefix lengths for c_derail
    derail_prefixes = dynamic_df[dynamic_df["conversation_id"] == "c_derail"].sort_values("prefix_length")
    assert list(derail_prefixes["prefix_length"]) == [1, 2, 3, 4]
    assert list(derail_prefixes["is_full_history"]) == [False, False, False, True]
    assert (derail_prefixes["label"] == 1).all()


# 3. Sorting and contiguous index verification
def test_validation_fails_on_non_contiguous_indices(sample_conversations_df: pd.DataFrame):
    df_broken = sample_conversations_df.copy()
    # Introduce gap: indices 0, 1, 3, 4 for c_clean (missing 2)
    df_broken.loc[df_broken["conversation_id"] == "c_clean", "utterance_index"] = [0, 1, 3, 4]

    with pytest.raises(DatasetValidationError, match="non-contiguous"):
        build_static_examples(df_broken)


def test_validation_fails_on_total_utterances_mismatch(sample_conversations_df: pd.DataFrame):
    df_broken = sample_conversations_df.copy()
    df_broken.loc[df_broken["conversation_id"] == "c_clean", "total_utterances"] = 99

    with pytest.raises(DatasetValidationError, match="does not agree with actual row count"):
        build_static_examples(df_broken)


# 4. Correct mapping between has_attack and terminal message label
def test_validation_fails_on_terminal_label_mismatch(sample_conversations_df: pd.DataFrame):
    df_broken = sample_conversations_df.copy()
    # c_derail has has_attack=True, but change terminal comment is_comment_attack to False
    mask = (df_broken["conversation_id"] == "c_derail") & (df_broken["utterance_index"] == 4)
    df_broken.loc[mask, "is_comment_attack"] = False

    with pytest.raises(DatasetValidationError, match="terminal turn label agreement"):
        build_static_examples(df_broken)


def test_validation_rejects_non_terminal_attack_turn(sample_conversations_df: pd.DataFrame):
    # Test positive conversation with an earlier non-terminal attack turn (e.g. index 1)
    df_early_attack = sample_conversations_df.copy()
    mask = (df_early_attack["conversation_id"] == "c_derail") & (df_early_attack["utterance_index"] == 1)
    df_early_attack.loc[mask, "is_comment_attack"] = True

    with pytest.raises(DatasetValidationError, match="attack message\\(s\\) at non-terminal turn"):
        build_static_examples(df_early_attack)

    # Test negative conversation with an earlier non-terminal attack turn
    df_neg_early = sample_conversations_df.copy()
    mask_neg = (df_neg_early["conversation_id"] == "c_clean") & (df_neg_early["utterance_index"] == 1)
    df_neg_early.loc[mask_neg, "is_comment_attack"] = True

    with pytest.raises(DatasetValidationError, match="attack message\\(s\\) at non-terminal turn"):
        build_static_examples(df_neg_early)


# 5. Deterministic prefix generation
def test_deterministic_prefix_generation(sample_conversations_df: pd.DataFrame):
    df1 = build_dynamic_examples(sample_conversations_df)
    df2 = build_dynamic_examples(sample_conversations_df)
    pd.testing.assert_frame_equal(df1, df2)


# 6. Dynamic prediction aggregation by conversation
def test_dynamic_prediction_aggregation():
    dynamic_df = pd.DataFrame(
        [
            {"conversation_id": "c1", "prefix_length": 1, "is_full_history": False, "label": 1},
            {"conversation_id": "c1", "prefix_length": 2, "is_full_history": False, "label": 1},
            {"conversation_id": "c1", "prefix_length": 3, "is_full_history": True, "label": 1},
            {"conversation_id": "c2", "prefix_length": 1, "is_full_history": False, "label": 0},
            {"conversation_id": "c2", "prefix_length": 2, "is_full_history": True, "label": 0},
        ]
    )
    probs = [0.2, 0.6, 0.4, 0.1, 0.3]

    agg = aggregate_dynamic_predictions(dynamic_df, probs)
    assert len(agg) == 2

    c1 = agg[agg["conversation_id"] == "c1"].iloc[0]
    assert c1["full_prefix_prob"] == 0.4
    assert c1["max_prob"] == 0.6
    assert round(c1["mean_prob"], 4) == round((0.2 + 0.6 + 0.4) / 3, 4)

    c2 = agg[agg["conversation_id"] == "c2"].iloc[0]
    assert c2["full_prefix_prob"] == 0.3
    assert c2["max_prob"] == 0.3


# 7. Forecast-horizon arithmetic and false-alert counting
def test_forecast_horizon_arithmetic():
    # N=5 conversation with 4 prefixes (k=1, 2, 3, 4)
    # Earliest alert occurs at k=2
    # Horizon must be N - k = 5 - 2 = 3 turns remaining
    dynamic_df = pd.DataFrame(
        [
            {"conversation_id": "pos_1", "prefix_length": 1, "total_conversation_turns": 5, "label": 1},
            {"conversation_id": "pos_1", "prefix_length": 2, "total_conversation_turns": 5, "label": 1},
            {"conversation_id": "pos_1", "prefix_length": 3, "total_conversation_turns": 5, "label": 1},
            {"conversation_id": "pos_1", "prefix_length": 4, "total_conversation_turns": 5, "label": 1},
            # Negative conversation (N=3, prefixes 1, 2), triggers false alert at k=2
            {"conversation_id": "neg_1", "prefix_length": 1, "total_conversation_turns": 3, "label": 0},
            {"conversation_id": "neg_1", "prefix_length": 2, "total_conversation_turns": 3, "label": 0},
        ]
    )
    # Alert at pos_1 k=2 (prob 0.8) and neg_1 k=2 (prob 0.7)
    probs = [0.1, 0.8, 0.9, 0.85, 0.2, 0.7]

    fh = compute_forecast_horizon_metrics(dynamic_df, probs, threshold=0.5)
    assert fh["positive_conversations_total"] == 1
    assert fh["alerted_positive_conversations"] == 1
    assert fh["mean_horizon"] == 3.0
    assert fh["min_horizon"] == 3
    assert fh["max_horizon"] == 3
    assert fh["warned_before_latest_prefix_count"] == 1
    assert fh["false_alert_conversations_count"] == 1
    assert fh["false_alert_rate"] == 1.0


# 8. Handling of short conversations and finite probability outputs
def test_short_conversation_rejection():
    short_df = pd.DataFrame(
        [
            {
                "conversation_id": "short_c",
                "utterance_id": "u1",
                "text": "Single turn only",
                "speaker": "Alice",
                "timestamp": 1.0,
                "total_utterances": 1,
                "utterance_index": 0,
                "has_attack": False,
                "is_comment_attack": False,
            }
        ]
    )
    with pytest.raises(DatasetValidationError, match="minimum required for forecasting is 2"):
        validate_forecasting_conversation("short_c", short_df, min_turns=2)


def test_baseline_finite_probability_outputs(sample_conversations_df: pd.DataFrame):
    static_df = build_static_examples(sample_conversations_df)
    X = static_df["text"].to_numpy()
    y = static_df["label"].to_numpy()

    prior = PriorBaseline().fit(X, y)
    prior_probs = prior.predict_proba(X)
    assert np.all(np.isfinite(prior_probs))
    assert np.all((prior_probs >= 0.0) & (prior_probs <= 1.0))

    tfidf_lr = StaticTfidfBaseline().fit(X, y)
    lr_probs = tfidf_lr.predict_proba(X)
    assert np.all(np.isfinite(lr_probs))
    assert np.all((lr_probs >= 0.0) & (lr_probs <= 1.0))


# 9. Turn-preserving recursive truncation behavior
def test_turn_preserving_truncation_behavior():
    tokenizer = MockTokenizer()
    turns = [
        "short turn one",
        "a very very very very very very long turn with many words that needs to be truncated",
        "another short turn three",
    ]

    # Max length: enough to fit CLS + turns + SEPs only if longest turn is cropped
    max_len = 12
    input_ids, attention_mask = turn_preserving_truncate(turns, tokenizer, max_length=max_len)

    assert len(input_ids) <= max_len
    assert len(input_ids) == len(attention_mask)
    assert input_ids[0] == tokenizer.cls_token_id
    assert input_ids[-1] == tokenizer.sep_token_id

    # Verify that all 3 turns still have representation ([SEP] follows each turn)
    sep_count = sum(1 for tid in input_ids if tid == tokenizer.sep_token_id)
    assert sep_count == 3


# 10. Structural contract assertions
def test_assert_forecasting_contract_success(sample_conversations_df: pd.DataFrame):
    static_df = build_static_examples(sample_conversations_df)
    dynamic_df = build_dynamic_examples(sample_conversations_df)
    # Must succeed without error
    assert_forecasting_contract(sample_conversations_df, static_df, dynamic_df)


def test_assert_forecasting_contract_catches_violations(sample_conversations_df: pd.DataFrame):
    static_df = build_static_examples(sample_conversations_df)
    dynamic_df = build_dynamic_examples(sample_conversations_df)

    # Violation 1: Tampered static turns (including terminal turn)
    bad_static = static_df.copy()
    bad_static.at[0, "turns"] = bad_static.at[0, "turns"] + ["LEAKED TARGET TURN"]
    with pytest.raises(AssertionError, match="static turns do not match"):
        assert_forecasting_contract(sample_conversations_df, bad_static, dynamic_df)

    # Violation 2: Tampered dynamic prefix length
    bad_dynamic = dynamic_df.copy()
    bad_dynamic.at[0, "prefix_length"] = 999
    with pytest.raises(AssertionError, match="prefix length mismatch"):
        assert_forecasting_contract(sample_conversations_df, static_df, bad_dynamic)


# 11. Defensible Macro-F1 threshold selection regression test
def test_macro_f1_threshold_selection_avoids_all_positive_collapse():
    # Construct a dataset where an all-positive classifier yields high positive F1 (0.667)
    # but a balanced threshold achieves substantially superior Macro-F1 (0.792).
    # 5 negatives, 5 positives
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.10, 0.20, 0.30, 0.40, 0.45, 0.25, 0.35, 0.55, 0.65, 0.75])

    # All-positive threshold (e.g. 0.05) predicts all 1s
    all_pos_preds = (y_prob >= 0.05).astype(int)
    assert all_pos_preds.sum() == 10
    from sklearn.metrics import f1_score
    macro_all_pos = f1_score(y_true, all_pos_preds, average="macro", zero_division=0)
    assert np.isclose(macro_all_pos, 0.333333, atol=1e-3)

    # Macro-F1 selection must pick a balanced threshold and NOT reproduce all-positive classifier
    opt_th_macro = find_optimal_threshold(y_true, y_prob, metric="macro_f1")
    assert opt_th_macro > 0.40, f"Expected balanced threshold > 0.40, got {opt_th_macro}"

    preds_opt = (y_prob >= opt_th_macro).astype(int)
    # Must not predict all positives
    assert preds_opt.sum() < 10, "Threshold selection reproduced an all-positive classifier!"
    # Must have non-zero true negatives
    assert ((preds_opt == 0) & (y_true == 0)).sum() > 0

    macro_opt = f1_score(y_true, preds_opt, average="macro", zero_division=0)
    assert macro_opt > macro_all_pos, f"Macro-F1 {macro_opt} should exceed all-positive Macro-F1 {macro_all_pos}"

    # Verify evaluate_predictions records the correct criterion
    eval_res = evaluate_predictions(y_true, y_prob, fixed_threshold=0.5, optimal_metric="macro_f1")
    assert eval_res["validation_selected_threshold"]["selection_criterion"] == "max_macro_f1_on_validation"
    assert "positive_alert_coverage" in eval_res["fixed_threshold_0.5"]
    assert "negative_false_alert_rate" in eval_res["fixed_threshold_0.5"]
    assert "positive_alert_coverage" in eval_res["validation_selected_threshold"]["metrics"]
    assert "negative_false_alert_rate" in eval_res["validation_selected_threshold"]["metrics"]


# 12. Run configuration serialization regression test
def test_serialize_run_config_with_path_objects():
    raw_config = {
        "model_name": "bert-base-uncased",
        "output_dir": Path("outputs_bert_baselines"),
        "checkpoints_dir": Path("outputs_bert_baselines/checkpoints"),
        "epochs": 3,
        "learning_rate": 2e-5,
        "seed": 42,
        "nested": {
            "sub_path": Path("artifacts/metrics.json"),
            "numpy_val": np.int64(10),
            "numpy_float": np.float64(0.5),
        },
        "path_list": [Path("dir1"), Path("dir2")],
    }

    # Direct json.dumps must raise TypeError because of Path objects
    with pytest.raises(TypeError, match="is not JSON serializable"):
        json.dumps(raw_config)

    # serialize_run_config must convert Paths and numpy objects to JSON-serializable primitives
    serialized = serialize_run_config(raw_config)
    json_str = json.dumps(serialized, indent=2)
    assert isinstance(json_str, str)

    loaded = json.loads(json_str)
    assert isinstance(loaded["output_dir"], str)
    assert "outputs_bert_baselines" in loaded["output_dir"]
    assert loaded["nested"]["numpy_val"] == 10
    assert loaded["nested"]["numpy_float"] == 0.5
    assert isinstance(loaded["path_list"][0], str)


# 13. Runtime path validation for artifact saving and ZIP packaging
def test_artifact_saving_and_packaging_runtime_path(tmp_path: Path):
    output_dir = tmp_path / "outputs_bert_baselines"
    ckpt_dir = output_dir / "checkpoints" / "bert_sc_static"

    config = {
        "model_name": "bert-base-uncased",
        "output_dir": output_dir,
        "epochs": 3,
        "seed": 42,
    }

    # 1. Preflight check: verify configuration containing Path values serializes cleanly
    safe_config = serialize_run_config(config)
    preflight_json = json.dumps(safe_config, indent=2)
    assert "outputs_bert_baselines" in preflight_json
    recovered_preflight = json.loads(preflight_json)
    assert isinstance(recovered_preflight["output_dir"], str)

    # 2. Lightweight mock model and tokenizer objects
    class MockConfig:
        def save_pretrained(self, save_directory: str | Path) -> None:
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "config.json", "w", encoding="utf-8") as f:
                json.dump({"architectures": ["BertForSequenceClassification"]}, f)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)
            self.config = MockConfig()

    class MockTokenizer:
        def save_pretrained(self, save_directory: str | Path) -> None:
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
                json.dump({"do_lower_case": True}, f)
            with open(save_path / "vocab.txt", "w", encoding="utf-8") as f:
                f.write("[CLS]\n[SEP]\n")

    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()

    val_metrics = {
        "accuracy": 0.65,
        "macro_f1": 0.64,
        "f1": 0.66,
        "positive_alert_coverage": 0.70,
        "negative_false_alert_rate": 0.35,
    }

    # 3. Call the actual production save_model_artifacts function
    saved_ckpt_dir = save_model_artifacts(
        model=mock_model,
        tokenizer=mock_tokenizer,
        checkpoint_dir=ckpt_dir,
        config_dict=config,
        metrics=val_metrics,
    )
    assert saved_ckpt_dir == ckpt_dir.resolve()

    # 4. Verify model weights are saved
    weights_file = ckpt_dir / "model.pt"
    assert weights_file.exists(), "Production save_model_artifacts failed to save model.pt"
    assert weights_file.stat().st_size > 0

    # 5. Verify tokenizer and model configuration files are written
    assert (ckpt_dir / "config.json").exists(), "Model configuration file was not written"
    assert (ckpt_dir / "tokenizer_config.json").exists(), "Tokenizer configuration was not written"
    assert (ckpt_dir / "vocab.txt").exists(), "Tokenizer vocab file was not written"

    # 6. Verify run_config.json serializes a configuration containing Path values
    run_cfg_file = ckpt_dir / "run_config.json"
    assert run_cfg_file.exists(), "run_config.json was not created by production save_model_artifacts"
    with open(run_cfg_file, encoding="utf-8") as f:
        run_metadata = json.load(f)
    assert isinstance(
        run_metadata["hyperparameters"]["output_dir"], str
    ), "Path was not serialized to string in run_config.json"
    assert "outputs_bert_baselines" in run_metadata["hyperparameters"]["output_dir"]
    assert "library_versions" in run_metadata
    assert "torch" in run_metadata["library_versions"]

    # 7. Verify val_metrics.json contains expected validation metrics
    val_metrics_file = ckpt_dir / "val_metrics.json"
    assert val_metrics_file.exists(), "val_metrics.json was not created by production save_model_artifacts"
    with open(val_metrics_file, encoding="utf-8") as f:
        loaded_val_metrics = json.load(f)
    assert loaded_val_metrics["macro_f1"] == 0.64
    assert loaded_val_metrics["positive_alert_coverage"] == 0.70

    # 8. Create consolidated metrics file before ZIP packaging
    final_metrics_file = output_dir / "bert_baselines_validation_metrics.json"
    final_results = {
        "metadata": {"hyperparameters": safe_config},
        "bert_sc_static": {"validation_evaluation": val_metrics},
    }
    with open(final_metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    # 9. Call the actual production package_artifacts function
    zip_archive = package_artifacts(output_dir=output_dir, zip_basename="convoguard_phase2_bert_outputs")
    assert zip_archive.exists(), "ZIP packaging failed to produce an archive"
    assert zip_archive.stat().st_size > 0

    # 10. Verify ZIP contents contain the resulting real output files
    with zipfile.ZipFile(zip_archive, "r") as zf:
        namelist = zf.namelist()
        assert any("model.pt" in name for name in namelist), "ZIP archive missing model.pt"
        assert any("config.json" in name for name in namelist), "ZIP archive missing config.json"
        assert any("tokenizer_config.json" in name for name in namelist), "ZIP archive missing tokenizer_config.json"
        assert any("run_config.json" in name for name in namelist), "ZIP archive missing run_config.json"
        assert any("val_metrics.json" in name for name in namelist), "ZIP archive missing val_metrics.json"
        assert any("bert_baselines_validation_metrics.json" in name for name in namelist), (
            "ZIP archive missing bert_baselines_validation_metrics.json"
        )

        # Inspect uncorrupted extraction of run_config.json
        run_cfg_name = [n for n in namelist if "run_config.json" in n][0]
        with zf.open(run_cfg_name) as rcf:
            recovered_meta = json.load(rcf)
            assert recovered_meta["hyperparameters"]["seed"] == 42
            assert recovered_meta["best_validation_metrics"]["macro_f1"] == 0.64




def test_core_model_imports_without_optional_bert_dependencies():
    """CPU APIs import without torch/transformers; BERT requests explain the extra."""
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "src"
    blocker_script = r"""
import importlib.abc
import sys

BLOCKED_ROOTS = {"torch", "transformers"}

class BlockOptionalBertDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in BLOCKED_ROOTS:
            raise ModuleNotFoundError(
                f"No module named {fullname!r} (blocked by dependency-isolation test)",
                name=fullname,
            )
        return None

sys.meta_path.insert(0, BlockOptionalBertDependencies())

from convoguard.models.baselines import StaticTfidfBaseline
from convoguard.models.evaluation import find_optimal_threshold
from convoguard.models.artifacts import package_artifacts, save_model_artifacts

assert StaticTfidfBaseline.__name__ == "StaticTfidfBaseline"
assert callable(find_optimal_threshold)
assert callable(package_artifacts)
assert callable(save_model_artifacts)
assert "torch" not in sys.modules
assert "transformers" not in sys.modules

def expect_clear_bert_dependency_error(statement):
    try:
        exec(statement, {})
    except ImportError as exc:
        message = str(exc).lower()
        assert "optional" in message, message
        assert "pip install -e" in message, message
        assert ".[bert]" in message, message
    else:
        raise AssertionError(f"Expected missing-optional-dependency error for: {statement}")

expect_clear_bert_dependency_error("from convoguard.models import BertForecastingClassifier")
expect_clear_bert_dependency_error("from convoguard.models.bert import BertForecastingClassifier")
print("CPU imports passed; BERT imports fail with actionable optional-dependency guidance.")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", blocker_script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        "Optional-dependency isolation subprocess failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "CPU imports passed" in result.stdout
