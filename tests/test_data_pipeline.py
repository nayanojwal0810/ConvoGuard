"""Automated tests for ConvoGuard data loading, validation, and auditing pipeline.

All tests utilize small synthetic fixtures. Synthetic data is used exclusively
for software verification and is never treated as dataset results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from convoguard.data.audit import audit_dataset, compute_file_sha256
from convoguard.data.loader import (
    REQUIRED_COLUMNS,
    DatasetValidationError,
    check_conversation_label_consistency,
    check_duplicate_messages,
    check_order_consistency,
    check_required_columns,
    check_split_overlap,
    find_attack_turns,
    load_dataset,
    load_split,
)


@pytest.fixture
def valid_conversation_data() -> pd.DataFrame:
    """Create a minimal valid synthetic conversation DataFrame with 2 conversations."""
    return pd.DataFrame(
        [
            # Conversation 1: Clean discussion (has_attack=False, 3 turns)
            {
                "conversation_id": "c1",
                "split": "train",
                "has_attack": False,
                "is_comment_attack": False,
                "utterance_id": "u1_0",
                "text": "Hello, could you explain the revert?",
                "speaker": "Alice",
                "timestamp": 1600000000.0,
                "total_utterances": 3,
                "utterance_index": 0,
            },
            {
                "conversation_id": "c1",
                "split": "train",
                "has_attack": False,
                "is_comment_attack": False,
                "utterance_id": "u1_1",
                "text": "Sure, it was missing citations.",
                "speaker": "Bob",
                "timestamp": 1600000100.0,
                "total_utterances": 3,
                "utterance_index": 1,
            },
            {
                "conversation_id": "c1",
                "split": "train",
                "has_attack": False,
                "is_comment_attack": False,
                "utterance_id": "u1_2",
                "text": "Understood, I will add the reliable sources.",
                "speaker": "Alice",
                "timestamp": 1600000200.0,
                "total_utterances": 3,
                "utterance_index": 2,
            },
            # Conversation 2: Derailed discussion (has_attack=True, 3 turns, attack at index 2)
            {
                "conversation_id": "c2",
                "split": "train",
                "has_attack": True,
                "is_comment_attack": False,
                "utterance_id": "u2_0",
                "text": "Please stop removing content.",
                "speaker": "Charlie",
                "timestamp": 1600000300.0,
                "total_utterances": 3,
                "utterance_index": 0,
            },
            {
                "conversation_id": "c2",
                "split": "train",
                "has_attack": True,
                "is_comment_attack": False,
                "utterance_id": "u2_1",
                "text": "The content violates neutral point of view.",
                "speaker": "Dave",
                "timestamp": 1600000400.0,
                "total_utterances": 3,
                "utterance_index": 1,
            },
            {
                "conversation_id": "c2",
                "split": "train",
                "has_attack": True,
                "is_comment_attack": True,
                "utterance_id": "u2_2",
                "text": "Synthetic attack text for unit testing only.",
                "speaker": "Charlie",
                "timestamp": 1600000500.0,
                "total_utterances": 3,
                "utterance_index": 2,
            },
        ]
    )


# 1. Valid file loading (Parquet and CSV)
def test_valid_file_loading_parquet(tmp_path: Path, valid_conversation_data: pd.DataFrame):
    parquet_path = tmp_path / "train_conversations.parquet"
    valid_conversation_data.to_parquet(parquet_path, index=False)

    df_loaded = load_split(parquet_path, validate=True)
    assert len(df_loaded) == 6
    assert list(df_loaded.columns) == list(valid_conversation_data.columns)
    assert (df_loaded["conversation_id"] == valid_conversation_data["conversation_id"]).all()


def test_valid_file_loading_csv(tmp_path: Path, valid_conversation_data: pd.DataFrame):
    csv_path = tmp_path / "train_conversations.csv"
    valid_conversation_data.to_csv(csv_path, index=False)

    df_loaded = load_split(csv_path, validate=True)
    assert len(df_loaded) == 6
    assert set(REQUIRED_COLUMNS).issubset(set(df_loaded.columns))


# 2. Missing required columns
def test_missing_required_columns(valid_conversation_data: pd.DataFrame):
    df_missing = valid_conversation_data.drop(columns=["is_comment_attack", "timestamp"])
    with pytest.raises(DatasetValidationError, match="missing required columns"):
        check_required_columns(df_missing)


# 3. Duplicate message keys
def test_duplicate_message_keys(valid_conversation_data: pd.DataFrame):
    df_dup = valid_conversation_data.copy()
    # Introduce duplicate utterance_id
    df_dup.loc[1, "utterance_id"] = df_dup.loc[0, "utterance_id"]

    dup_ids = check_duplicate_messages(df_dup)
    assert len(dup_ids) == 1
    assert dup_ids[0] == "u1_0"


# 4. Stable chronological ordering
def test_stable_chronological_ordering_success(valid_conversation_data: pd.DataFrame):
    inv_idx, inv_ts = check_order_consistency(valid_conversation_data)
    assert len(inv_idx) == 0
    assert len(inv_ts) == 0


def test_stable_chronological_ordering_violations(valid_conversation_data: pd.DataFrame):
    df_violating = valid_conversation_data.copy()
    # Invert order of utterance_index for conversation c1
    df_violating.loc[df_violating["conversation_id"] == "c1", "utterance_index"] = [2, 1, 0]
    # Reverse timestamp for conversation c2
    df_violating.loc[df_violating["conversation_id"] == "c2", "timestamp"] = [500.0, 400.0, 300.0]

    inv_idx, inv_ts = check_order_consistency(df_violating)
    assert "c1" in inv_idx
    assert "c2" in inv_ts


# 5. Consistency of conversation-level labels
def test_conversation_level_label_inconsistency(valid_conversation_data: pd.DataFrame):
    df_bad = valid_conversation_data.copy()
    # Set conflicting has_attack values within c1
    df_bad.loc[1, "has_attack"] = True

    inconsistent_conv, _ = check_conversation_label_consistency(df_bad)
    assert "c1" in inconsistent_conv


def test_conversation_message_label_mismatch(valid_conversation_data: pd.DataFrame):
    df_bad = valid_conversation_data.copy()
    # c2 is has_attack=True, but remove all attack comments
    df_bad.loc[df_bad["conversation_id"] == "c2", "is_comment_attack"] = False

    _, mismatched_msgs = check_conversation_label_consistency(df_bad)
    assert "c2" in mismatched_msgs


# 6. Correct identification of the annotated attack turn
def test_correct_identification_of_annotated_attack_turn(valid_conversation_data: pd.DataFrame):
    attack_turns = find_attack_turns(valid_conversation_data)
    assert len(attack_turns) == 1
    row = attack_turns.iloc[0]
    assert row["conversation_id"] == "c2"
    assert row["attack_utterance_id"] == "u2_2"
    assert row["attack_index"] == 2
    assert row["total_conversation_turns"] == 3
    assert bool(row["is_terminal_turn"]) is True
    assert row["prefix_length"] == 2


# 7. Correct detection of split overlap
def test_correct_detection_of_split_overlap(tmp_path: Path, valid_conversation_data: pd.DataFrame):
    train_df = valid_conversation_data.copy()
    # Val dataset with a leaked conversation ID from train
    val_df = pd.DataFrame(
        [
            {
                "conversation_id": "c2",  # Overlap with train!
                "split": "val",
                "has_attack": True,
                "is_comment_attack": True,
                "utterance_id": "u_val_0",
                "text": "Synthetic overlap text.",
                "speaker": "Eve",
                "timestamp": 1600000900.0,
                "total_utterances": 1,
                "utterance_index": 0,
            }
        ]
    )

    splits = {"train": train_df, "val": val_df}
    overlaps = check_split_overlap(splits, id_col="conversation_id")
    assert "train_vs_val" in overlaps
    assert "c2" in overlaps["train_vs_val"]

    # Test load_dataset fails with DatasetValidationError
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_df.to_parquet(data_dir / "train_conversations.parquet", index=False)
    val_df.to_parquet(data_dir / "val_conversations.parquet", index=False)

    with pytest.raises(DatasetValidationError, match="Data leakage detected"):
        load_dataset(data_dir, validate=True)


# 8. Empty or malformed input
def test_empty_or_malformed_input(tmp_path: Path):
    empty_file = tmp_path / "empty.parquet"
    empty_file.touch()

    with pytest.raises(DatasetValidationError, match="empty"):
        load_split(empty_file)

    corrupt_file = tmp_path / "corrupt.parquet"
    corrupt_file.write_bytes(b"not a valid parquet file content")

    with pytest.raises(DatasetValidationError, match="Failed to read Parquet"):
        load_split(corrupt_file)

    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_dataset(non_existent)


# 9. Reproducibility of audit summaries
def test_reproducibility_of_audit_summaries(tmp_path: Path, valid_conversation_data: pd.DataFrame):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    valid_conversation_data.to_parquet(data_dir / "train_conversations.parquet", index=False)

    out1 = tmp_path / "audit1.json"
    out2 = tmp_path / "audit2.json"

    res1 = audit_dataset(data_dir, output_path=out1)
    res2 = audit_dataset(data_dir, output_path=out2)

    # Identical returned dict
    assert res1 == res2

    # Identical on-disk JSON
    with out1.open("r", encoding="utf-8") as f1, out2.open("r", encoding="utf-8") as f2:
        assert f1.read() == f2.read()


# 10. Provenance SHA-256 calculation
def test_provenance_sha256(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_bytes(b"ConvoGuard Provenance Test")
    h = compute_file_sha256(sample_file)
    import hashlib

    expected = hashlib.sha256(b"ConvoGuard Provenance Test").hexdigest()
    assert h == expected


# 11. Verify message text is never exported in audit summary
def test_no_leakage_of_message_text_in_audit(tmp_path: Path, valid_conversation_data: pd.DataFrame):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    valid_conversation_data.to_parquet(data_dir / "train_conversations.parquet", index=False)

    out = tmp_path / "audit_leak_check.json"
    audit_dataset(data_dir, output_path=out)

    with out.open("r", encoding="utf-8") as f:
        content = f.read()

    # None of the private / synthetic comment strings must appear in the summary
    assert "Synthetic attack text for unit testing only" not in content
    assert "Hello, could you explain the revert?" not in content
