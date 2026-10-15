"""Dataset loader and validator for Wikipedia Talk-page conversations.

Supports Parquet and CSV formats with strict data integrity checks:
- Required column schema validation
- Duplicate message identifier detection
- Monotonic sequence ordering verification
- Conversation-level vs message-level label consistency
- Cross-split conversation ID leakage detection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when dataset integrity or schema validation fails."""

    pass


# Core canonical schema required for conversational derailment forecasting
REQUIRED_COLUMNS: set[str] = {
    "conversation_id",
    "utterance_id",
    "text",
    "speaker",
    "timestamp",
    "utterance_index",
    "has_attack",
    "is_comment_attack",
}

EXPECTED_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass
class DatasetSplits:
    """Container holding loaded DataFrames for train, val, and test splits."""

    train: pd.DataFrame | None = None
    val: pd.DataFrame | None = None
    test: pd.DataFrame | None = None

    def get_split(self, split_name: str) -> pd.DataFrame | None:
        return getattr(self, split_name, None)

    @property
    def available_splits(self) -> list[str]:
        return [s for s in EXPECTED_SPLIT_NAMES if getattr(self, s, None) is not None]

    def all_splits(self) -> dict[str, pd.DataFrame]:
        return {s: getattr(self, s) for s in self.available_splits}


def check_required_columns(
    df: pd.DataFrame, required_columns: set[str] | None = None
) -> None:
    """Ensure all required columns exist in the DataFrame."""
    if df is None:
        raise DatasetValidationError("Cannot validate schema on None DataFrame")
    if df.empty:
        raise DatasetValidationError("DataFrame is empty (0 rows)")

    cols = required_columns or REQUIRED_COLUMNS
    missing = cols - set(df.columns)
    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def check_duplicate_messages(df: pd.DataFrame, message_id_col: str = "utterance_id") -> list[str]:
    """Find any duplicate message identifiers.

    Returns:
        List of duplicate message IDs found.
    """
    if message_id_col not in df.columns:
        raise DatasetValidationError(f"Column '{message_id_col}' not found in DataFrame.")

    duplicates = df[df.duplicated(subset=[message_id_col], keep=False)]
    if not duplicates.empty:
        dup_ids = duplicates[message_id_col].drop_duplicates().tolist()
        return [str(d) for d in dup_ids]
    return []


def check_order_consistency(
    df: pd.DataFrame,
    conv_id_col: str = "conversation_id",
    index_col: str = "utterance_index",
    timestamp_col: str = "timestamp",
) -> tuple[list[str], list[str]]:
    """Check whether messages within each conversation are ordered chronologically.

    Returns:
        Tuple of (invalid_index_conv_ids, invalid_timestamp_conv_ids).
    """
    invalid_index: list[str] = []
    invalid_timestamp: list[str] = []

    grouped = df.groupby(conv_id_col, sort=False)
    for conv_id, group in grouped:
        if index_col in group.columns:
            indices = group[index_col].to_numpy()
            if len(indices) > 1 and not (indices[:-1] < indices[1:]).all():
                invalid_index.append(str(conv_id))

        if timestamp_col in group.columns:
            timestamps = group[timestamp_col].to_numpy()
            if len(timestamps) > 1 and not (timestamps[:-1] <= timestamps[1:]).all():
                invalid_timestamp.append(str(conv_id))

    return invalid_index, invalid_timestamp


def check_conversation_label_consistency(
    df: pd.DataFrame,
    conv_id_col: str = "conversation_id",
    conv_label_col: str = "has_attack",
    msg_label_col: str = "is_comment_attack",
) -> tuple[list[str], list[str]]:
    """Check label consistency across messages within each conversation.

    Verifies:
    1. Every message in a conversation has the identical conversation-level label (`has_attack`).
    2. Conversations labeled has_attack=True contain at least one attack comment (`is_comment_attack=True`),
       and conversations labeled has_attack=False contain no attack comments.

    Returns:
        Tuple of (inconsistent_conv_label_ids, mismatched_message_label_conv_ids).
    """
    inconsistent_conv_labels: list[str] = []
    mismatched_message_labels: list[str] = []

    for conv_id, group in df.groupby(conv_id_col, sort=False):
        unique_conv_labels = group[conv_label_col].unique()
        if len(unique_conv_labels) > 1:
            inconsistent_conv_labels.append(str(conv_id))
            continue

        conv_label = bool(unique_conv_labels[0])
        has_attack_msg = bool((group[msg_label_col] == True).any())  # noqa: E712

        if conv_label != has_attack_msg:
            mismatched_message_labels.append(str(conv_id))

    return inconsistent_conv_labels, mismatched_message_labels


def find_attack_turns(
    df: pd.DataFrame,
    conv_id_col: str = "conversation_id",
    index_col: str = "utterance_index",
    msg_label_col: str = "is_comment_attack",
) -> pd.DataFrame:
    """Identify the turn index and utterance ID where the attack occurs for derailed conversations.

    Returns:
        DataFrame indexed by conversation_id with attack_utterance_id,
        attack_index, and total_conversation_turns.
    """
    records = []
    for conv_id, group in df.groupby(conv_id_col, sort=False):
        attacks = group[group[msg_label_col] == True]  # noqa: E712
        total_turns = len(group)
        if not attacks.empty:
            for _, row in attacks.iterrows():
                records.append(
                    {
                        "conversation_id": str(conv_id),
                        "attack_utterance_id": str(row.get("utterance_id")),
                        "attack_index": int(row.get(index_col)),
                        "total_conversation_turns": int(total_turns),
                        "is_terminal_turn": bool(row.get(index_col) == total_turns - 1),
                        "prefix_length": int(row.get(index_col)),
                    }
                )
    return pd.DataFrame(records)


def check_split_overlap(
    splits: dict[str, pd.DataFrame],
    id_col: str = "conversation_id",
) -> dict[str, list[str]]:
    """Check for ID overlap across splits (e.g., train vs val, train vs test, val vs test).

    Returns:
        Dictionary mapping split pair (e.g., 'train_vs_val') to list of overlapping IDs.
    """
    overlaps: dict[str, list[str]] = {}
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            df1, df2 = splits[s1], splits[s2]
            if id_col in df1.columns and id_col in df2.columns:
                set1 = set(df1[id_col].dropna().astype(str))
                set2 = set(df2[id_col].dropna().astype(str))
                common = sorted(set1.intersection(set2))
                if common:
                    overlaps[f"{s1}_vs_{s2}"] = common
    return overlaps


def load_split(
    file_path: str | Path,
    validate: bool = True,
    required_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Load a single split file (Parquet or CSV) and optionally run validation.

    Args:
        file_path: Path to .parquet or .csv file.
        validate: If True, validates required columns and duplicate message IDs.
        required_columns: Custom set of required columns, or REQUIRED_COLUMNS by default.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetValidationError: If schema or validation checks fail.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split file not found at: {path.resolve()}")

    if path.stat().st_size == 0:
        raise DatasetValidationError(f"File is empty (0 bytes): {path.resolve()}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise DatasetValidationError(f"Failed to read Parquet file {path}: {e}") from e
    elif suffix in (".csv", ".txt"):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DatasetValidationError(f"Failed to read CSV file {path}: {e}") from e
    else:
        raise DatasetValidationError(
            f"Unsupported file format '{suffix}' for {path}. Expected .parquet or .csv"
        )

    if validate:
        check_required_columns(df, required_columns=required_columns)
        dup_msgs = check_duplicate_messages(df)
        if dup_msgs:
            raise DatasetValidationError(
                f"Found {len(dup_msgs)} duplicate message identifiers in {path.name}: {dup_msgs[:5]}"
            )
        inconsistent_conv, mismatched_msgs = check_conversation_label_consistency(df)
        if inconsistent_conv:
            raise DatasetValidationError(
                f"Found {len(inconsistent_conv)} conversations with inconsistent conversation labels: "
                f"{inconsistent_conv[:5]}"
            )
        if mismatched_msgs:
            raise DatasetValidationError(
                f"Found {len(mismatched_msgs)} conversations with mismatched message/conversation labels: "
                f"{mismatched_msgs[:5]}"
            )

    return df


def load_dataset(
    data_root: str | Path,
    validate: bool = True,
    required_columns: set[str] | None = None,
) -> DatasetSplits:
    """Discover and load all available dataset splits (train, val, test) from directory.

    Searches for:
    - `<split>_conversations.parquet` or `<split>_conversations.csv`
    - `<split>.parquet` or `<split>.csv`

    Args:
        data_root: Root directory containing dataset files.
        validate: If True, validates individual splits and cross-split overlap.
        required_columns: Optional column override for validation.

    Returns:
        DatasetSplits containing train, val, and test DataFrames.

    Raises:
        FileNotFoundError: If no valid split files are found in data_root.
        DatasetValidationError: If validation fails.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root directory does not exist or is not a directory: {root.resolve()}"
        )

    splits_data: dict[str, pd.DataFrame] = {}

    for split_name in EXPECTED_SPLIT_NAMES:
        candidate_filenames = [
            f"{split_name}_conversations.parquet",
            f"{split_name}_conversations.csv",
            f"{split_name}.parquet",
            f"{split_name}.csv",
        ]

        found_path: Path | None = None
        for fname in candidate_filenames:
            candidate = root / fname
            if candidate.is_file():
                found_path = candidate
                break

        if found_path is not None:
            df = load_split(
                found_path,
                validate=validate,
                required_columns=required_columns,
            )
            splits_data[split_name] = df

    if not splits_data:
        raise FileNotFoundError(
            f"No recognizable dataset splits found in {root.resolve()}. "
            f"Expected files like train_conversations.parquet, val_conversations.parquet, etc."
        )

    if validate and len(splits_data) > 1:
        conv_overlaps = check_split_overlap(splits_data, id_col="conversation_id")
        if conv_overlaps:
            raise DatasetValidationError(
                f"Data leakage detected across splits (conversation_id overlap): {conv_overlaps}"
            )

        msg_overlaps = check_split_overlap(splits_data, id_col="utterance_id")
        if msg_overlaps:
            raise DatasetValidationError(
                f"Data leakage detected across splits (utterance_id overlap): {msg_overlaps}"
            )

    return DatasetSplits(
        train=splits_data.get("train"),
        val=splits_data.get("val"),
        test=splits_data.get("test"),
    )
