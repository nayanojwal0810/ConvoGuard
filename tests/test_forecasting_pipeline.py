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

import numpy as np
import pandas as pd
import pytest

from convoguard.data.forecasting import (
    assert_forecasting_contract,
    build_dynamic_examples,
    build_static_examples,
    validate_forecasting_conversation,
)
from convoguard.data.loader import DatasetValidationError
from convoguard.models.baselines import PriorBaseline, StaticTfidfBaseline
from convoguard.models.bert import turn_preserving_truncate
from convoguard.models.evaluation import (
    aggregate_dynamic_predictions,
    compute_forecast_horizon_metrics,
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

