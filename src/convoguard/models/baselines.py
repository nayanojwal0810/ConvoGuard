"""Fast CPU baselines for conversational derailment forecasting.

Implements:
1. PriorBaseline: Empirical class-prior and majority baseline.
2. StaticTfidfBaseline: Word-and-character TF-IDF + Logistic Regression on full pre-target histories.
3. DynamicTfidfBaseline: Word-and-character TF-IDF + Logistic Regression on successive prefix histories.

All vectorizers and classifiers fit strictly on training text only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


class PriorBaseline:
    """Class-prior baseline predicting the empirical training positive rate."""

    def __init__(self):
        self.prior_prob_: float = 0.5
        self.majority_class_: int = 0

    def fit(self, texts: list[str] | np.ndarray, y: list[int] | np.ndarray) -> PriorBaseline:
        y_arr = np.asarray(y, dtype=int)
        self.prior_prob_ = float(np.mean(y_arr))
        self.majority_class_ = int(self.prior_prob_ >= 0.5)
        return self

    def predict_proba(self, texts: list[str] | np.ndarray) -> np.ndarray:
        n = len(texts)
        prob_1 = np.full(n, self.prior_prob_, dtype=float)
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, texts: list[str] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)


def build_tfidf_pipeline(
    max_word_features: int = 10000,
    max_char_features: int = 15000,
    c_reg: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Build a fixed word-and-character TF-IDF feature pipeline with Logistic Regression."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_word_features,
        sublinear_tf=True,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char",
        max_features=max_char_features,
        sublinear_tf=True,
        lowercase=True,
    )

    union = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_vec),
            ("char_tfidf", char_vec),
        ]
    )

    classifier = LogisticRegression(
        C=c_reg,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )

    return Pipeline([("features", union), ("clf", classifier)])


class StaticTfidfBaseline:
    """Static TF-IDF + Logistic Regression trained on full pre-target conversation prefixes."""

    def __init__(
        self,
        max_word_features: int = 10000,
        max_char_features: int = 15000,
        c_reg: float = 1.0,
        random_state: int = 42,
    ):
        self.pipeline = build_tfidf_pipeline(
            max_word_features=max_word_features,
            max_char_features=max_char_features,
            c_reg=c_reg,
            random_state=random_state,
        )

    def fit(self, texts: list[str] | np.ndarray, y: list[int] | np.ndarray) -> StaticTfidfBaseline:
        y_arr = np.asarray(y, dtype=int)
        self.pipeline.fit(texts, y_arr)
        return self

    def predict_proba(self, texts: list[str] | np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: list[str] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)


class DynamicTfidfBaseline:
    """Dynamic TF-IDF + Logistic Regression trained on successive prefix conversation histories."""

    def __init__(
        self,
        max_word_features: int = 10000,
        max_char_features: int = 15000,
        c_reg: float = 1.0,
        random_state: int = 42,
    ):
        self.pipeline = build_tfidf_pipeline(
            max_word_features=max_word_features,
            max_char_features=max_char_features,
            c_reg=c_reg,
            random_state=random_state,
        )

    def fit(self, texts: list[str] | np.ndarray, y: list[int] | np.ndarray) -> DynamicTfidfBaseline:
        y_arr = np.asarray(y, dtype=int)
        self.pipeline.fit(texts, y_arr)
        return self

    def predict_proba(self, texts: list[str] | np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: list[str] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)
