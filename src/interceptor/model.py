from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from .features import combine_goal_and_trace


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float


class DriftClassifier:
    """Classifier for detecting hijacked logic from goal-trace text pairs."""

    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        if model_type not in {"random_forest", "logistic"}:
            raise ValueError("model_type must be one of: random_forest, logistic")

        if model_type == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            estimator = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            )

        self.model_type = model_type
        self.pipeline: Pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("clf", estimator),
            ]
        )

    def _to_texts(self, goals: Iterable[str], traces: Iterable[str]) -> list[str]:
        return [combine_goal_and_trace(g, t) for g, t in zip(goals, traces)]

    def fit(self, goals: Iterable[str], traces: Iterable[str], labels: Iterable[int]) -> None:
        texts = self._to_texts(goals, traces)
        self.pipeline.fit(texts, list(labels))

    def predict(self, goals: Iterable[str], traces: Iterable[str]) -> list[int]:
        texts = self._to_texts(goals, traces)
        return list(self.pipeline.predict(texts))

    def predict_proba(self, goals: Iterable[str], traces: Iterable[str]) -> list[float]:
        texts = self._to_texts(goals, traces)
        prob = self.pipeline.predict_proba(texts)
        return [float(row[1]) for row in prob]

    def evaluate(self, goals: Iterable[str], traces: Iterable[str], labels: Iterable[int]) -> EvalResult:
        y_true = list(labels)
        y_pred = self.predict(goals, traces)
        return EvalResult(
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
        )

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "model_type": self.model_type,
                "pipeline": self.pipeline,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DriftClassifier":
        payload = joblib.load(path)
        obj = cls(model_type=payload["model_type"])
        obj.pipeline = payload["pipeline"]
        return obj
