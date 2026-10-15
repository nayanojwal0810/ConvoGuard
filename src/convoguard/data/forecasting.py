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
    6. Non-terminal turns in a clean conversation must have `is_comment_attack == False`.

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
