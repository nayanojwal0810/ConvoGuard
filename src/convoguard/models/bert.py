"""Implementation of published BERT Sequence Classification baselines (BERT-SC & BERT-SC+).

Reference Paper:
    "Dynamic Forecasting of Conversation Derailment"
    Yova Kementchedjhieva and Anders Søgaard (EMNLP 2021)
    https://aclanthology.org/2021.emnlp-main.624/
Official Implementation Reference:
    https://github.com/YovaKem/forecasting_derailment

Key Features:
- bert-base-uncased encoder for conversational sequence classification.
- Turn-preserving recursive truncation algorithm cropping the longest turn down to 512 tokens.
- Explicit [SEP] turn separators between conversational turns.
- BERT-SC: Static full-prefix training (one sample per conversation).
- BERT-SC+: Dynamic prefix training (successive prefix samples per conversation).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def turn_preserving_truncate(
    turns: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> tuple[list[int], list[int]]:
    """Tokenize and truncate conversation turns preserving all turns according to EMNLP 2021.

    Algorithm (Kementchedjhieva & Søgaard, 2021):
    1. Tokenize each turn individually into subword token IDs (without CLS/SEP).
    2. Account for special tokens: 1 [CLS] at start and 1 [SEP] after each turn.
       Total tokens = 1 (CLS) + sum(len(turn_ids) + 1 (SEP) for each turn).
    3. While total tokens > max_length:
       Identify the turn with the maximum number of tokens, and crop 1 token from its end.
    4. Construct final input_ids: [CLS] + turn_0 + [SEP] + turn_1 + [SEP] + ...

    Args:
        turns: List of string turns in chronological order.
        tokenizer: Hugging Face PreTrainedTokenizer (e.g., BertTokenizer).
        max_length: Maximum allowed sequence length (default: 512).

    Returns:
        tuple of (input_ids, attention_mask) of length <= max_length.
    """
    if not turns:
        cls_id = tokenizer.cls_token_id or 101
        sep_id = tokenizer.sep_token_id or 102
        return [cls_id, sep_id], [1, 1]

    # Tokenize each turn individually (no special tokens)
    tokenized_turns: list[list[int]] = [
        tokenizer.encode(turn, add_special_tokens=False) for turn in turns
    ]

    # Calculate budget: [CLS] + sum(len(t) + [SEP])
    # Total = 1 + sum(len(t) + 1 for t in tokenized_turns) = 1 + sum(len(t)) + len(tokenized_turns)
    def current_total_length() -> int:
        return 1 + sum(len(t) + 1 for t in tokenized_turns)

    # Recursive longest-turn truncation
    while current_total_length() > max_length:
        lengths = [len(t) for t in tokenized_turns]
        max_len = max(lengths)
        if max_len == 0:
            # All turns are completely empty, cannot crop further
            break
        # Pick the first turn matching maximum length and pop the last token
        longest_idx = lengths.index(max_len)
        tokenized_turns[longest_idx].pop()

    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102

    input_ids: list[int] = [cls_id]
    for turn_ids in tokenized_turns:
        input_ids.extend(turn_ids)
        input_ids.append(sep_id)

    # Hard safety guard in the event max_length is smaller than turn count
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length - 1] + [sep_id]

    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask


class BertForecastingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for ConvoGuard BERT conversational forecasting."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        turns = ex.get("turns", [ex["text"]])
        input_ids, attention_mask = turn_preserving_truncate(
            turns=turns,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        return {
            "conversation_id": ex["conversation_id"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(ex["label"], dtype=torch.long),
            "prefix_length": ex.get("prefix_length", len(turns)),
            "total_conversation_turns": ex.get("total_conversation_turns", len(turns) + 1),
        }


def pad_bert_collate(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """Collate function dynamically padding batch to the maximum sequence length in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    labels = []
    conv_ids = []
    prefix_lens = []
    total_turns = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        padding_len = max_len - seq_len

        padded_input_ids = torch.cat(
            [item["input_ids"], torch.full((padding_len,), pad_token_id, dtype=torch.long)]
        )
        padded_attention_mask = torch.cat(
            [item["attention_mask"], torch.zeros(padding_len, dtype=torch.long)]
        )

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)
        labels.append(item["label"])
        conv_ids.append(item["conversation_id"])
        prefix_lens.append(item["prefix_length"])
        total_turns.append(item["total_conversation_turns"])

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(labels),
        "conversation_ids": conv_ids,
        "prefix_lengths": prefix_lens,
        "total_conversation_turns": total_turns,
    }


class BertForecastingClassifier(nn.Module):
    """BERT Sequence Classification model for conversation derailment forecasting."""

    def __init__(
        self,
        pretrained_model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.config = BertConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
        )
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss, outputs.logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
