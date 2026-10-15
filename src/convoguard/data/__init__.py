"""Data loading, validation, and auditing modules for ConvoGuard."""

from convoguard.data.audit import DatasetAuditor, audit_dataset
from convoguard.data.forecasting import (
    DynamicExample,
    StaticExample,
    build_dynamic_examples,
    build_static_examples,
    validate_forecasting_conversation,
)
from convoguard.data.loader import (
    DatasetSplits,
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

__all__ = [
    "DatasetSplits",
    "DatasetValidationError",
    "DatasetAuditor",
    "audit_dataset",
    "load_split",
    "load_dataset",
    "check_required_columns",
    "check_duplicate_messages",
    "check_order_consistency",
    "check_conversation_label_consistency",
    "check_split_overlap",
    "find_attack_turns",
    "StaticExample",
    "DynamicExample",
    "validate_forecasting_conversation",
    "build_static_examples",
    "build_dynamic_examples",
]
