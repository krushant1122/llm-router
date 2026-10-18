"""Difficulty classifier routing signal.

Extracts hand-crafted features from a prompt and produces a difficulty
score in [0, 1].  A higher score indicates the prompt is likely too hard
for a local model and should be escalated to the frontier.

When a trained sklearn model is available it is used; otherwise a
weighted-heuristic fallback is applied.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from router.models import SignalResult


class DifficultyClassifier:
    """Scores prompt difficulty as a routing signal.

    Parameters
    ----------
    model_path:
        Optional path to a persisted ``joblib`` model (sklearn
        ``LogisticRegression``).  If supplied and the file exists the
        model is loaded and used for inference; otherwise a hand-tuned
        heuristic is used.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._model: object | None = None
        if model_path and model_path.exists():
            import joblib

            self._model = joblib.load(model_path)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, prompt: str) -> dict[str, float]:
        """Extract hand-crafted features from a prompt.

        Returns a dict whose keys are stable across calls (and sorted
        alphabetically) so the resulting vector is compatible with the
        sklearn model.
        """
        words = prompt.split()
        n_words = len(words)
        n_chars = max(len(prompt), 1)

        return {
            "has_code_fence": float("```" in prompt),
            "has_code_markers": float(
                any(
                    kw in prompt.lower()
                    for kw in ["def ", "function ", "class ", "import ", "return "]
                )
            ),
            "has_math_symbols": float(
                any(s in prompt for s in ["∑", "∫", "∂", "√", "π", "≥", "≤"])
            ),
            "length": float(n_words),
            "length_norm": min(n_words / 500, 1.0),
            "multi_step_cues": float(
                any(
                    kw in prompt.lower()
                    for kw in ["step by step", "first", "then", "finally", "prove", "derive"]
                )
            ),
            "num_questions": float(len(re.findall(r"\?", prompt))),
            "num_sub_questions": float(
                len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[a-z][.)]|[-*])", prompt))
            ),
            "numeric_density": (
                sum(1 for w in words if any(c.isdigit() for c in w)) / max(n_words, 1)
            ),
            "operator_density": sum(1 for c in prompt if c in "+-*/=<>") / n_chars,
            "sentence_count": float(len(re.findall(r"[.!?]+", prompt))),
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, prompt: str) -> SignalResult:
        """Return a ``SignalResult`` with the difficulty score."""
        features = self.extract_features(prompt)

        if self._model is not None:
            # Use trained logistic regression
            feature_vector = np.array([[features[k] for k in sorted(features.keys())]])
            prob = float(self._model.predict_proba(feature_vector)[0, 1])  # type: ignore[union-attr]
        else:
            prob = self._heuristic_score(features)

        return SignalResult(
            name="difficulty",
            score=min(max(prob, 0.0), 1.0),
            metadata=features,
        )

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_score(self, features: dict[str, float]) -> float:
        """Weighted heuristic when no trained model is available."""
        score = 0.0
        score += features["length_norm"] * 0.20
        score += features["numeric_density"] * 0.15
        score += features["has_code_fence"] * 0.25
        score += features["has_code_markers"] * 0.15
        score += features["multi_step_cues"] * 0.15
        score += min(features["num_sub_questions"] / 4.0, 1.0) * 0.10
        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, prompts: list[str], labels: list[int], save_path: Path) -> None:
        """Train a logistic regression difficulty classifier.

        Parameters
        ----------
        prompts:
            Raw prompt strings.
        labels:
            Binary labels — 1 = difficult / should escalate.
        save_path:
            Where to persist the trained model via ``joblib``.
        """
        import joblib
        from sklearn.linear_model import LogisticRegression

        rows = [list(dict(sorted(self.extract_features(p).items())).values()) for p in prompts]
        feature_matrix = np.array(rows)
        target = np.array(labels)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(feature_matrix, target)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        self._model = model
