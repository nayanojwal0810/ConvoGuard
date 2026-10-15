"""Dataset auditing and provenance verification for ConvoGuard.

Performs comprehensive validation and statistical profiling of raw conversation
datasets without exposing message content or committing private data:
- File inventory, SHA-256 provenance hashes, shapes, and schemas
- Utterance and conversation counts per split
- Distinct conversation-level vs message-level label distributions
- Null value and duplicate key checks
- Monotonic sequence index and timestamp ordering checks
- Cross-split conversation and utterance leakage checks
- Attack message positioning and pre-attack observation prefix lengths
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from convoguard.config import get_config
from convoguard.data.loader import (
    EXPECTED_SPLIT_NAMES,
    REQUIRED_COLUMNS,
    check_conversation_label_consistency,
    check_duplicate_messages,
    check_order_consistency,
    check_split_overlap,
    find_attack_turns,
    load_split,
)

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: str | Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 cryptographic hash of a file for provenance."""
    p = Path(filepath)
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetAuditor:
    """Comprehensive auditor for ConvoGuard conversation datasets."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root).resolve()

    def discover_files(self) -> list[Path]:
        """Find candidate dataset files in data_root."""
        if not self.data_root.exists() or not self.data_root.is_dir():
            return []

        candidates = []
        for ext in ("*.parquet", "*.csv"):
            candidates.extend(self.data_root.glob(ext))
        return sorted(candidates)

    def audit(self) -> dict[str, Any]:
        """Run the complete audit on data_root and return structured dictionary."""
        if not self.data_root.exists() or not self.data_root.is_dir():
            return {
                "status": "DATASET_NOT_FOUND",
                "error": f"Dataset directory not found at: {self.data_root}",
                "data_root": str(self.data_root),
                "checks_evaluated": False,
            }

        files = self.discover_files()
        if not files:
            return {
                "status": "NO_DATA_FILES",
                "error": f"No supported dataset files (.parquet, .csv) found in {self.data_root}",
                "data_root": str(self.data_root),
                "checks_evaluated": False,
            }

        file_inventory: dict[str, Any] = {}
        splits_raw: dict[str, pd.DataFrame] = {}

        for f in files:
            file_hash = compute_file_sha256(f)
            file_stat = f.stat()
            file_info: dict[str, Any] = {
                "filename": f.name,
                "size_bytes": file_stat.st_size,
                "format": f.suffix.lstrip(".").lower(),
                "sha256": file_hash,
            }

            try:
                # Load without strict schema validation first to record real file state
                df_unvalidated = load_split(f, validate=False)
                file_info["shape"] = [int(df_unvalidated.shape[0]), int(df_unvalidated.shape[1])]
                file_info["columns"] = list(df_unvalidated.columns)
                file_info["dtypes"] = {col: str(dtype) for col, dtype in df_unvalidated.dtypes.items()}
                file_info["read_status"] = "SUCCESS"

                # Check if file corresponds to an expected split name
                for split_name in EXPECTED_SPLIT_NAMES:
                    if split_name in f.name.lower():
                        splits_raw[split_name] = df_unvalidated
                        break
            except Exception as e:
                file_info["read_status"] = "ERROR"
                file_info["read_error"] = str(e)

            file_inventory[f.name] = file_info

        # Run split-level and global checks
        split_reports: dict[str, Any] = {}
        all_dfs: list[pd.DataFrame] = []

        for split_name, df in splits_raw.items():
            all_dfs.append(df)
            split_reports[split_name] = self._audit_split(df, split_name)

        # Cross-split overlap checks
        leakage_checks: dict[str, Any] = {}
        if len(splits_raw) > 1:
            conv_overlaps = check_split_overlap(splits_raw, id_col="conversation_id")
            msg_overlaps = check_split_overlap(splits_raw, id_col="utterance_id")
            leakage_checks = {
                "conversation_id_overlap": {
                    pair: len(ids) for pair, ids in conv_overlaps.items()
                },
                "utterance_id_overlap": {
                    pair: len(ids) for pair, ids in msg_overlaps.items()
                },
                "has_conversation_leakage": len(conv_overlaps) > 0,
                "has_utterance_leakage": len(msg_overlaps) > 0,
            }
        else:
            leakage_checks = {
                "status": "UNAVAILABLE",
                "reason": "Fewer than 2 splits available to evaluate cross-split leakage",
            }

        # Overall aggregate summary
        aggregate_summary = self._compute_aggregate_summary(splits_raw, all_dfs)

        return {
            "status": "SUCCESS",
            "data_root": str(self.data_root),
            "file_inventory": file_inventory,
            "split_reports": split_reports,
            "leakage_checks": leakage_checks,
            "aggregate_summary": aggregate_summary,
        }

    def _audit_split(self, df: pd.DataFrame, split_name: str) -> dict[str, Any]:
        """Audit a single split DataFrame."""
        report: dict[str, Any] = {
            "num_utterances": int(len(df)),
            "columns_present": list(df.columns),
        }

        # Required columns check
        missing_required = sorted(list(REQUIRED_COLUMNS - set(df.columns)))
        report["missing_required_columns"] = missing_required
        report["has_canonical_schema"] = len(missing_required) == 0

        # Null values check (never output raw message text)
        report["null_counts"] = {col: int(count) for col, count in df.isnull().sum().items()}
        report["total_null_entries"] = int(df.isnull().sum().sum())

        # Duplicate utterance ID check
        if "utterance_id" in df.columns:
            dup_msgs = check_duplicate_messages(df, message_id_col="utterance_id")
            report["duplicate_utterance_ids_count"] = len(dup_msgs)
        else:
            report["duplicate_utterance_ids_count"] = "UNAVAILABLE (column missing)"

        # Conversation count
        if "conversation_id" in df.columns:
            unique_convs = int(df["conversation_id"].nunique())
            report["num_conversations"] = unique_convs
        else:
            report["num_conversations"] = "UNAVAILABLE (column missing)"
            unique_convs = 0

        # Conversation-level label distribution (has_attack)
        if "conversation_id" in df.columns and "has_attack" in df.columns:
            conv_labels = df.groupby("conversation_id", sort=False)["has_attack"].first()
            true_convs = int((conv_labels == True).sum())  # noqa: E712
            false_convs = int((conv_labels == False).sum())  # noqa: E712
            report["conversation_label_distribution"] = {
                "derailed_conversations (has_attack=True)": true_convs,
                "clean_conversations (has_attack=False)": false_convs,
                "derailment_rate": round(true_convs / unique_convs, 6) if unique_convs else 0.0,
            }
        else:
            report["conversation_label_distribution"] = "UNAVAILABLE"

        # Message-level label distribution (is_comment_attack)
        if "is_comment_attack" in df.columns:
            true_msgs = int((df["is_comment_attack"] == True).sum())  # noqa: E712
            false_msgs = int((df["is_comment_attack"] == False).sum())  # noqa: E712
            report["message_label_distribution"] = {
                "attack_messages (is_comment_attack=True)": true_msgs,
                "non_attack_messages (is_comment_attack=False)": false_msgs,
                "attack_message_rate": round(true_msgs / len(df), 6) if len(df) else 0.0,
            }
        else:
            report["message_label_distribution"] = "UNAVAILABLE"

        # Sequence ordering & timestamp monotonicity
        if (
            "conversation_id" in df.columns
            and "utterance_index" in df.columns
            and "timestamp" in df.columns
        ):
            invalid_idx, invalid_ts = check_order_consistency(df)
            report["ordering_checks"] = {
                "violating_index_order_conversations_count": len(invalid_idx),
                "violating_timestamp_order_conversations_count": len(invalid_ts),
                "is_index_strictly_monotonic": len(invalid_idx) == 0,
                "is_timestamp_non_decreasing": len(invalid_ts) == 0,
            }
        else:
            report["ordering_checks"] = "UNAVAILABLE"

        # Within-conversation label consistency
        if (
            "conversation_id" in df.columns
            and "has_attack" in df.columns
            and "is_comment_attack" in df.columns
        ):
            inconsistent_conv, mismatched_msgs = check_conversation_label_consistency(df)
            report["label_consistency_checks"] = {
                "inconsistent_conv_labels_count": len(inconsistent_conv),
                "mismatched_message_labels_count": len(mismatched_msgs),
                "is_labeling_consistent": len(inconsistent_conv) == 0 and len(mismatched_msgs) == 0,
            }
        else:
            report["label_consistency_checks"] = "UNAVAILABLE"

        # Attack position & prefix length distribution
        if (
            "conversation_id" in df.columns
            and "utterance_index" in df.columns
            and "is_comment_attack" in df.columns
        ):
            attack_turns = find_attack_turns(df)
            if not attack_turns.empty:
                prefix_lens = attack_turns["prefix_length"].to_numpy()
                terminal_count = int(attack_turns["is_terminal_turn"].sum())
                total_attacks = len(attack_turns)
                report["attack_positioning"] = {
                    "total_annotated_attacks": total_attacks,
                    "terminal_turn_attacks_count": terminal_count,
                    "terminal_turn_percentage": round(terminal_count / total_attacks * 100, 2),
                    "prefix_lengths_before_attack": {
                        "min": int(np.min(prefix_lens)),
                        "max": int(np.max(prefix_lens)),
                        "mean": round(float(np.mean(prefix_lens)), 3),
                        "median": float(np.median(prefix_lens)),
                        "p25": float(np.percentile(prefix_lens, 25)),
                        "p75": float(np.percentile(prefix_lens, 75)),
                    },
                }
            else:
                report["attack_positioning"] = {
                    "total_annotated_attacks": 0,
                    "note": "No attacks annotated in this split",
                }
        else:
            report["attack_positioning"] = "UNAVAILABLE"

        # Conversation length profile
        if "conversation_id" in df.columns:
            conv_lens = df.groupby("conversation_id", sort=False).size().to_numpy()
            report["conversation_lengths"] = {
                "min": int(np.min(conv_lens)),
                "max": int(np.max(conv_lens)),
                "mean": round(float(np.mean(conv_lens)), 3),
                "median": float(np.median(conv_lens)),
                "p25": float(np.percentile(conv_lens, 25)),
                "p75": float(np.percentile(conv_lens, 75)),
            }
        else:
            report["conversation_lengths"] = "UNAVAILABLE"

        return report

    def _compute_aggregate_summary(
        self, splits_raw: dict[str, pd.DataFrame], all_dfs: list[pd.DataFrame]
    ) -> dict[str, Any]:
        """Compute dataset-wide aggregate counts and checks across all splits."""
        if not all_dfs:
            return {"status": "UNAVAILABLE"}

        total_messages = sum(len(df) for df in all_dfs)
        total_convs = sum(
            df["conversation_id"].nunique() for df in all_dfs if "conversation_id" in df.columns
        )

        all_derailed = 0
        all_clean = 0
        all_attack_msgs = 0
        all_clean_msgs = 0

        for df in all_dfs:
            if "conversation_id" in df.columns and "has_attack" in df.columns:
                cl = df.groupby("conversation_id", sort=False)["has_attack"].first()
                all_derailed += int((cl == True).sum())  # noqa: E712
                all_clean += int((cl == False).sum())  # noqa: E712
            if "is_comment_attack" in df.columns:
                all_attack_msgs += int((df["is_comment_attack"] == True).sum())  # noqa: E712
                all_clean_msgs += int((df["is_comment_attack"] == False).sum())  # noqa: E712

        return {
            "total_splits": len(splits_raw),
            "split_names": list(splits_raw.keys()),
            "total_messages": total_messages,
            "total_unique_conversations": total_convs,
            "overall_conversation_distribution": {
                "derailed": all_derailed,
                "clean": all_clean,
                "derailment_rate": round(all_derailed / total_convs, 6) if total_convs else 0.0,
            },
            "overall_message_distribution": {
                "attack_messages": all_attack_msgs,
                "clean_messages": all_clean_msgs,
                "attack_message_rate": round(all_attack_msgs / total_messages, 6)
                if total_messages
                else 0.0,
            },
        }


def audit_dataset(
    data_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the dataset at data_root and optionally save output JSON.

    Args:
        data_root: Root directory of dataset.
        output_path: Optional path to save JSON audit summary.

    Returns:
        Structured audit report dict.
    """
    auditor = DatasetAuditor(data_root=data_root)
    summary = auditor.audit()

    if output_path is not None:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved dataset audit summary to %s", out)

    return summary


def main_cli() -> None:
    """CLI entry point for convoguard-audit and python scripts/audit_dataset.py."""
    parser = argparse.ArgumentParser(
        description="Audit ConvoGuard dataset files, schema integrity, and label distributions."
    )
    config = get_config()
    default_data_root = str(config.resolve_data_root())

    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data_root,
        help="Path to the dataset directory containing Parquet or CSV split files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/data_audit/summary.json",
        help="Path to save the JSON audit report.",
    )
    args = parser.parse_args()

    print("[*] Starting ConvoGuard dataset audit...")
    print(f"[*] Data root: {args.data_root}")
    print(f"[*] Target output: {args.output}")

    summary = audit_dataset(data_root=args.data_root, output_path=args.output)

    status = summary.get("status")
    print(f"[*] Audit finished with status: {status}")

    if status == "SUCCESS":
        agg = summary.get("aggregate_summary", {})
        print(f"    Total messages: {agg.get('total_messages')}")
        print(f"    Total conversations: {agg.get('total_unique_conversations')}")
        print(f"    Splits found: {agg.get('split_names')}")
        print(f"    Conversation derailment distribution: {agg.get('overall_conversation_distribution')}")
        print(f"    Message attack distribution: {agg.get('overall_message_distribution')}")
        print(f"[OK] Summary saved successfully to {Path(args.output).resolve()}")
    else:
        print(f"[!] Audit failed or incomplete: {summary.get('error')}")


if __name__ == "__main__":
    main_cli()
