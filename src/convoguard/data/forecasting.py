"""Forecasting dataset construction and contract validation.

Implements the canonical early conversational derailment forecasting task:
- Strict contract validation ensuring valid chronological ordering and label consistency.
- Exclusion of the final target turn from all input features.
- Construction of static (full pre-target history) and dynamic (successive prefix) examples.
- Preservation of conversation-level outcome labels and prefix metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from convoguard.data.loader import DatasetValidationError


@dataclass(frozen=True)
class StaticExample:
    """Represents a single static conversation forecasting instance."""

    conversation_id: str
    text: str
    turns: list[str]
    label: int
    prefix_length: int
    total_conversation_turns: int
    split: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "text": self.text,
            "turns": self.turns,
            "label": self.label,
            "prefix_length": self.prefix_length,
            "total_conversation_turns": self.total_conversation_turns,
            "split": self.split,
        }


@dataclass(frozen=True)
class DynamicExample:
    """Represents a single dynamic prefix forecasting instance."""

    conversation_id: str
    prefix_index: int
    prefix_length: int
    total_conversation_turns: int
    text: str
    turns: list[str]
    label: int
    is_full_history: bool
    split: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "prefix_index": self.prefix_index,
            "prefix_length": self.prefix_length,
            "total_conversation_turns": self.total_conversation_turns,
            "text": self.text,
            "turns": self.turns,
            "label": self.label,
            "is_full_history": self.is_full_history,
            "split": self.split,
        }


def validate_forecasting_conversation(
    conv_id: str,
    group: pd.DataFrame,
    min_turns: int = 2,
) -> pd.DataFrame:
    """Validate that a conversation satisfies all forecasting contract requirements.

    Requirements:
    1. At least `min_turns` (default 2) so that at least 1 turn remains after excluding target.
    2. Sortable by `utterance_index`, producing contiguous indices 0, 1, ..., N-1.
    3. `total_utterances` column agrees with actual turn count N.
    4. `has_attack` is uniform across all turns.
    5. Terminal turn (`utterance_index == N - 1`) has `is_comment_attack == has_attack`.
    6. Non-terminal turns (0 to N-2) must have `is_comment_attack == False` (applies to both
       positive and negative conversations; attack occurs strictly at terminal target turn).

    Returns:
        Sorted pd.DataFrame of conversation turns.

    Raises:
        DatasetValidationError: If any contract requirement fails.
    """
    n_turns = len(group)
    if n_turns < min_turns:
        raise DatasetValidationError(
            f"Conversation '{conv_id}' has {n_turns} turns, but minimum required for "
            f"forecasting is {min_turns}."
        )

    # Sort strictly by utterance_index
    if "utterance_index" not in group.columns:
        raise DatasetValidationError(f"Missing 'utterance_index' in conversation '{conv_id}'.")

    sorted_group = group.sort_values("utterance_index").reset_index(drop=True)

    # Verify contiguous 0..N-1 indices
    indices = sorted_group["utterance_index"].to_numpy()
    expected_indices = np.arange(n_turns)
    if not np.array_equal(indices, expected_indices):
        raise DatasetValidationError(
            f"Conversation '{conv_id}' has non-contiguous utterance indices: "
            f"{indices.tolist()} (expected {expected_indices.tolist()})"
        )

    # Verify total_utterances agrees with row count if column exists
    if "total_utterances" in sorted_group.columns:
        reported_totals = sorted_group["total_utterances"].unique()
        if len(reported_totals) != 1 or reported_totals[0] != n_turns:
            raise DatasetValidationError(
                f"Conversation '{conv_id}' reported total_utterances={reported_totals} "
                f"does not agree with actual row count={n_turns}."
            )

    # Verify conversation-level label consistency
    if "has_attack" not in sorted_group.columns:
        raise DatasetValidationError(f"Missing 'has_attack' in conversation '{conv_id}'.")

    unique_conv_labels = sorted_group["has_attack"].unique()
    if len(unique_conv_labels) != 1:
        raise DatasetValidationError(
            f"Conversation '{conv_id}' has conflicting has_attack labels: {unique_conv_labels}."
        )

    conv_outcome = bool(unique_conv_labels[0])

    # Verify terminal turn label agreement
    if "is_comment_attack" not in sorted_group.columns:
        raise DatasetValidationError(f"Missing 'is_comment_attack' in conversation '{conv_id}'.")

    terminal_turn = sorted_group.iloc[-1]
    terminal_comment_attack = bool(terminal_turn["is_comment_attack"])
    if terminal_comment_attack != conv_outcome:
        raise DatasetValidationError(
            f"Conversation '{conv_id}' has has_attack={conv_outcome}, but terminal turn "
            f"has is_comment_attack={terminal_comment_attack}. Forecasting target requires "
            f"terminal turn label agreement."
        )

    # Enforce contract: no non-terminal turn may be an attack (applies to both positive and negative conversations)
    non_terminal_turns = sorted_group.iloc[:-1]
    non_terminal_attacks = non_terminal_turns[non_terminal_turns["is_comment_attack"] == True]  # noqa: E712
    if not non_terminal_attacks.empty:
        earlier_indices = non_terminal_attacks["utterance_index"].tolist()
        raise DatasetValidationError(
            f"Conversation '{conv_id}' has attack message(s) at non-terminal turn(s) {earlier_indices}. "
            f"Forecasting contract requires all pre-target turns to have is_comment_attack=False "
            f"(derailment attack must occur exclusively at the terminal target turn)."
        )

    return sorted_group


def build_static_examples(
    df: pd.DataFrame,
    turn_separator: str = " [SEP] ",
    min_turns: int = 2,
) -> pd.DataFrame:
    """Build static forecasting dataset (exactly one sample per conversation).

    Input: all messages preceding the final target message.
    Target: binary outcome indicating whether the excluded final turn derails (1) or not (0).

    Args:
        df: Raw conversation DataFrame.
        turn_separator: String used to join observed turn texts.
        min_turns: Minimum required conversation turns.

    Returns:
        DataFrame of static examples with columns:
        ['conversation_id', 'text', 'turns', 'label', 'prefix_length',
         'total_conversation_turns', 'split']
    """
    records: list[dict[str, Any]] = []

    for conv_id, group in df.groupby("conversation_id", sort=False):
        valid_group = validate_forecasting_conversation(str(conv_id), group, min_turns=min_turns)
        n_turns = len(valid_group)

        # Strictly exclude the terminal target turn (index n_turns - 1)
        observed_history = valid_group.iloc[:-1]
        observed_turns = [str(t) for t in observed_history["text"].tolist()]
        concatenated_text = turn_separator.join(observed_turns)
        label = int(bool(valid_group.iloc[0]["has_attack"]))
        split = str(valid_group.iloc[0].get("split", "unknown"))

        example = StaticExample(
            conversation_id=str(conv_id),
            text=concatenated_text,
            turns=observed_turns,
            label=label,
            prefix_length=len(observed_turns),
            total_conversation_turns=n_turns,
            split=split,
        )
        records.append(example.to_dict())

    return pd.DataFrame(records)


def build_dynamic_examples(
    df: pd.DataFrame,
    turn_separator: str = " [SEP] ",
    min_turns: int = 2,
) -> pd.DataFrame:
    """Build dynamic forecasting dataset (successive prefixes of pre-target history).

    For a conversation with N turns, observed history contains turns 0..N-2 (N-1 turns).
    Generates N-1 prefix examples for prefix lengths k = 1, 2, ..., N-1.
    Each prefix inherits the eventual conversation outcome label.

    Args:
        df: Raw conversation DataFrame.
        turn_separator: String used to join observed turn texts.
        min_turns: Minimum required conversation turns.

    Returns:
        DataFrame of dynamic examples with columns:
        ['conversation_id', 'prefix_index', 'prefix_length', 'total_conversation_turns',
         'text', 'turns', 'label', 'is_full_history', 'split']
    """
    records: list[dict[str, Any]] = []

    for conv_id, group in df.groupby("conversation_id", sort=False):
        valid_group = validate_forecasting_conversation(str(conv_id), group, min_turns=min_turns)
        n_turns = len(valid_group)

        # Observed history: turns 0 to N-2
        observed_history = valid_group.iloc[:-1]
        all_observed_turns = [str(t) for t in observed_history["text"].tolist()]
        label = int(bool(valid_group.iloc[0]["has_attack"]))
        split = str(valid_group.iloc[0].get("split", "unknown"))

        # Generate successive prefixes of length 1, 2, ..., len(observed_turns)
        for k in range(1, len(all_observed_turns) + 1):
            prefix_turns = all_observed_turns[:k]
            concatenated_text = turn_separator.join(prefix_turns)
            is_full = (k == len(all_observed_turns))

            example = DynamicExample(
                conversation_id=str(conv_id),
                prefix_index=k - 1,  # 0-indexed observed position
                prefix_length=k,     # 1-indexed count of observed turns
                total_conversation_turns=n_turns,
                text=concatenated_text,
                turns=prefix_turns,
                label=label,
                is_full_history=is_full,
                split=split,
            )
            records.append(example.to_dict())

    return pd.DataFrame(records)


def assert_forecasting_contract(
    raw_df: pd.DataFrame,
    static_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
) -> None:
    """Rigorous structural assertion verifying the forecasting contract against raw conversations.

    Verifies:
    1. Static examples count equals unique conversations count in raw_df.
    2. For each conversation with N turns, static example turns equal exactly the first N-1 source turns.
    3. Terminal turn N-1 is never included in static turns.
    4. For each conversation with N turns, there are exactly N-1 dynamic prefixes (lengths 1 to N-1).
    5. Each dynamic prefix of length k has turns matching exactly the first k source turns.
    6. Terminal turn N-1 is never included in any dynamic prefix.
    7. All static and dynamic examples inherit the validated conversation outcome label.
    8. Prefix lengths and total conversation turns metadata match actual turn counts.

    Raises:
        AssertionError: If any structural contract invariant is violated.
    """
    raw_conv_groups = {str(cid): grp for cid, grp in raw_df.groupby("conversation_id", sort=False)}
    static_by_id = {str(row["conversation_id"]): row for _, row in static_df.iterrows()}
    dynamic_by_id: dict[str, list[dict[str, Any]]] = {}
    for _, row in dynamic_df.iterrows():
        cid = str(row["conversation_id"])
        dynamic_by_id.setdefault(cid, []).append(row.to_dict())

    assert len(static_by_id) == len(raw_conv_groups), (
        f"Static examples count ({len(static_by_id)}) does not match raw conversation count ({len(raw_conv_groups)})."
    )

    for conv_id, raw_group in raw_conv_groups.items():
        sorted_raw = raw_group.sort_values("utterance_index")
        source_turns = [str(t) for t in sorted_raw["text"].tolist()]
        n_turns = len(source_turns)
        expected_label = int(bool(sorted_raw["has_attack"].iloc[0]))

        # --- Static Contract Verification ---
        assert conv_id in static_by_id, f"Conversation '{conv_id}' missing from static examples."
        static_ex = static_by_id[conv_id]
        expected_static_turns = source_turns[:-1]

        assert static_ex["turns"] == expected_static_turns, (
            f"Conversation '{conv_id}' static turns do not match first N-1 source turns. "
            f"Expected {len(expected_static_turns)} turns, got {len(static_ex['turns'])}."
        )
        assert static_ex["prefix_length"] == n_turns - 1, (
            f"Conversation '{conv_id}' static prefix_length ({static_ex['prefix_length']}) != N-1 ({n_turns - 1})."
        )
        assert static_ex["total_conversation_turns"] == n_turns, (
            f"Conversation '{conv_id}' static total_turns ({static_ex['total_conversation_turns']}) != N ({n_turns})."
        )
        assert static_ex["label"] == expected_label, (
            f"Conversation '{conv_id}' static label ({static_ex['label']}) != expected ({expected_label})."
        )

        # --- Dynamic Contract Verification ---
        assert conv_id in dynamic_by_id, f"Conversation '{conv_id}' missing from dynamic examples."
        prefixes = sorted(dynamic_by_id[conv_id], key=lambda x: int(x["prefix_length"]))
        assert len(prefixes) == n_turns - 1, (
            f"Conversation '{conv_id}' expected {n_turns - 1} dynamic prefixes, got {len(prefixes)}."
        )

        for k_idx, prefix_ex in enumerate(prefixes, start=1):
            assert prefix_ex["prefix_length"] == k_idx, (
                f"Conversation '{conv_id}' prefix length mismatch: {prefix_ex['prefix_length']} != {k_idx}."
            )
            expected_prefix_turns = source_turns[:k_idx]
            assert prefix_ex["turns"] == expected_prefix_turns, (
                f"Conversation '{conv_id}' prefix {k_idx} turns do not match first {k_idx} source turns."
            )
            assert prefix_ex["label"] == expected_label, (
                f"Conversation '{conv_id}' prefix {k_idx} label ({prefix_ex['label']}) != expected ({expected_label})."
            )
            assert prefix_ex["total_conversation_turns"] == n_turns, (
                f"Conversation '{conv_id}' prefix {k_idx} total_turns != N."
            )
            assert prefix_ex["is_full_history"] == (k_idx == n_turns - 1), (
                f"Conversation '{conv_id}' is_full_history flag incorrect for prefix {k_idx}."
            )
