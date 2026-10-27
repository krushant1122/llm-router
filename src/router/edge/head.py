"""Edge router classification head and training pipeline.

Trains a compact MLP (384 -> 64 -> 1) on frozen sentence embeddings using
actual outcome labels (or proxy labels as secondary) with early stopping
on a held-out validation split. Also retrains the Logistic Regression
baseline on the exact same dataset for a fair benchmark.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

# Ensure HuggingFace loads directly from local cache without HTTP retry delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import joblib
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from router.config import AppConfig
from router.signals.difficulty import DifficultyClassifier

logger = logging.getLogger(__name__)


class EdgeRouterHead(nn.Module):
    """Compact classification head operating on 384-dim frozen embeddings."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output probability of escalation in [0, 1]."""
        return self.net(x).squeeze(-1)


def train_edge_head(
    config: AppConfig,
    data_path: Path | None = None,
    output_dir: Path | None = None,
    seed: int = 42,
    epochs: int = 100,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    patience: int = 10,
) -> dict[str, Any]:
    """Train the MLP head on frozen sentence embeddings with validation early-stopping.

    Also trains a matching LogisticRegression model on hand-crafted features
    using the exact same train/validation outcome labels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_file = data_path or config.edge.training_data_path
    out_dir = output_dir or config.edge.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found at {data_file}. Run `router edge label` first.")

    # 1. Load labeled tasks
    records: list[dict[str, Any]] = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {data_file}")

    prompts = [r["prompt"] for r in records]
    outcome_labels = np.array([r["outcome_label"] for r in records], dtype=np.float32)

    total_samples = len(records)
    n_pos = int(np.sum(outcome_labels))
    n_neg = total_samples - n_pos

    logger.info("Training dataset: %d total samples (Positive/Escalate: %d, Negative/Local: %d)", total_samples, n_pos, n_neg)

    # 2. Extract or load cached embeddings
    emb_cache_path = out_dir / "train_embeddings_cache.pt"
    if emb_cache_path.exists():
        logger.info("Loading cached embeddings from %s", emb_cache_path)
        embeddings = torch.load(emb_cache_path, weights_only=True)
    else:
        logger.info("Encoding %d prompts with %s...", total_samples, config.edge.model_id)
        encoder = SentenceTransformer(config.edge.model_id, local_files_only=True)
        raw_embs = encoder.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = torch.tensor(raw_embs, dtype=torch.float32)
        torch.save(embeddings, emb_cache_path)
        logger.info("Saved embeddings cache to %s", emb_cache_path)

    # 3. Train / Val split (held-out validation for early stopping)
    indices = list(range(total_samples))
    random.shuffle(indices)
    val_size = max(5, int(total_samples * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = embeddings[train_idx]
    y_train = torch.tensor(outcome_labels[train_idx], dtype=torch.float32)
    X_val = embeddings[val_idx]
    y_val = torch.tensor(outcome_labels[val_idx], dtype=torch.float32)

    # 4. Train EdgeRouterHead
    input_dim = embeddings.shape[1]
    hidden_dim = config.edge.head_hidden_dim
    head = EdgeRouterHead(input_dim=input_dim, hidden_dim=hidden_dim)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights: dict[str, Any] | None = None
    best_epoch = 0
    patience_counter = 0

    head.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        preds = head(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        # Validation check
        head.eval()
        with torch.no_grad():
            val_preds = head(X_val)
            val_loss = float(criterion(val_preds, y_val))

        head.train()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d (best epoch %d, val_loss: %.4f)", epoch, best_epoch, best_val_loss)
                break

    assert best_weights is not None
    head.load_state_dict(best_weights)
    head.eval()

    head_path = out_dir / "head.pt"
    torch.save(head.state_dict(), head_path)
    logger.info("Saved trained head to %s", head_path)

    # Evaluate head on validation set
    with torch.no_grad():
        val_probs = head(X_val).numpy()
        val_preds_binary = (val_probs >= 0.5).astype(int)
        val_targets = y_val.numpy().astype(int)

    val_acc = float(accuracy_score(val_targets, val_preds_binary))
    val_f1 = float(f1_score(val_targets, val_preds_binary, zero_division=0))
    try:
        val_auc = float(roc_auc_score(val_targets, val_probs))
    except ValueError:
        val_auc = 0.5

    # 5. Retrain matching LogisticRegression baseline on same outcome labels
    diff_clf = DifficultyClassifier()
    feature_dicts = [diff_clf.extract_features(p) for p in prompts]
    X_handcrafted = np.array([list(dict(sorted(fd.items())).values()) for fd in feature_dicts])

    lr_model = LogisticRegression(max_iter=1000, random_state=seed)
    lr_model.fit(X_handcrafted[train_idx], outcome_labels[train_idx])

    lr_val_probs = lr_model.predict_proba(X_handcrafted[val_idx])[:, 1]
    lr_val_preds = (lr_val_probs >= 0.5).astype(int)
    lr_acc = float(accuracy_score(val_targets, lr_val_preds))
    lr_f1 = float(f1_score(val_targets, lr_val_preds, zero_division=0))
    try:
        lr_auc = float(roc_auc_score(val_targets, lr_val_probs))
    except ValueError:
        lr_auc = 0.5

    baseline_path = config.paths.models_dir / "edge_baseline_logistic.joblib"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_model, baseline_path)
    logger.info("Saved retrained baseline LogisticRegression to %s", baseline_path)

    results_summary: dict[str, Any] = {
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "class_balance": {
            "total": total_samples,
            "escalate_positive": n_pos,
            "local_negative": n_neg,
            "escalate_pct": round(n_pos / total_samples * 100, 2),
        },
        "early_stopping": {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
        },
        "mlp_head_val_metrics": {
            "accuracy": round(val_acc, 4),
            "f1": round(val_f1, 4),
            "auroc": round(val_auc, 4),
        },
        "logistic_baseline_val_metrics": {
            "accuracy": round(lr_acc, 4),
            "f1": round(lr_f1, 4),
            "auroc": round(lr_auc, 4),
        },
        "artifacts": {
            "head_path": str(head_path),
            "baseline_path": str(baseline_path),
        },
    }

    metrics_log_path = out_dir / "train_metrics.json"
    metrics_log_path.write_text(json.dumps(results_summary, indent=2))
    return results_summary
