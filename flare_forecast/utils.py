"""
utils.py
========
Shared utilities for the Aditya-L1 Solar Flare Forecasting pipeline.

Provides:
- Deterministic seed setting (NumPy + TensorFlow)
- Structured logging (file + console)
- Class weight computation
- JSON I/O helpers
- SHAP explainability wrapper (with graceful fallback)
"""

from __future__ import annotations

import os
import json
import logging
import random
import pickle
import numpy as np
from typing import Any, Dict, Optional, Sequence


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  –  Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for full reproducibility across NumPy, Python,
    and TensorFlow (must be called BEFORE importing tensorflow).

    Parameters
    ----------
    seed : int
        Master random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  Logging
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(name: str = "flare_forecast",
               log_file: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to both the console (stdout) and
    optionally a file.

    Parameters
    ----------
    name     : Logger name (usually the module calling this function).
    log_file : Optional path to a .log file. If None, file logging is skipped.
    level    : Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s  %(name)s  |  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3  –  Class Weights
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(y: np.ndarray, n_classes: int) -> Dict[int, float]:
    """
    Compute balanced class weights to counteract class imbalance.

    Uses sklearn's 'balanced' mode:
        weight[c] = n_samples / (n_classes × count[c])

    Parameters
    ----------
    y        : 1-D integer array of class labels (0..n_classes-1).
    n_classes: Total number of classes.

    Returns
    -------
    Dict mapping class index → float weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(n_classes)
    # Only compute weights for classes actually present in y
    present = np.unique(y.astype(int))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=present,
        y=y.astype(int),
    )

    # Build full dict; missing classes get weight=1.0
    weight_dict: Dict[int, float] = {c: 1.0 for c in range(n_classes)}
    for c, w in zip(present, weights):
        weight_dict[int(c)] = float(w)

    return weight_dict


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4  –  JSON / Pickle I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_json(obj: Any, path: str) -> None:
    """Serialise *obj* to a JSON file at *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for numpy scalars."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_pickle(obj: Any, path: str) -> None:
    """Save *obj* to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load and return a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5  –  SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(model,
                        X_sample: np.ndarray,
                        background: np.ndarray,
                        n_background: int = 50,
                        logger: Optional[logging.Logger] = None):
    """
    Compute SHAP values for an LSTM model using DeepExplainer.

    If the `shap` library is not installed, falls back to computing
    permutation importance (feature-level only, not timestep-level).

    Parameters
    ----------
    model       : Trained Keras model.
    X_sample    : Array of shape (n_explain, window, features) to explain.
    background  : Array of shape (n_bg_pool, window, features) for background.
    n_background: How many background samples to use (random subset).
    logger      : Optional logger for status messages.

    Returns
    -------
    shap_values : list of arrays (one per class) of shape
                  (n_explain, window, features), or None on failure.
    method      : str — "shap_deep", "shap_gradient", or "permutation"
    """
    log = logger or get_logger("shap")

    # ── SHAP DeepExplainer ──────────────────────────────────────────────────
    try:
        import shap
        bg_idx  = np.random.choice(len(background),
                                   size=min(n_background, len(background)),
                                   replace=False)
        bg      = background[bg_idx]
        explainer = shap.DeepExplainer(model, bg)
        shap_vals = explainer.shap_values(X_sample[:min(50, len(X_sample))])
        log.info("SHAP DeepExplainer computed successfully.")
        return shap_vals, "shap_deep"
    except ImportError:
        log.warning("'shap' not installed — falling back to permutation importance.")
    except Exception as exc:
        log.warning(f"SHAP DeepExplainer failed ({exc}) — trying GradientExplainer.")

    # ── SHAP GradientExplainer (TF-only fallback) ──────────────────────────
    try:
        import shap
        import tensorflow as tf
        bg_idx  = np.random.choice(len(background),
                                   size=min(n_background, len(background)),
                                   replace=False)
        bg = background[bg_idx]
        explainer = shap.GradientExplainer(model, bg)
        shap_vals = explainer.shap_values(X_sample[:min(50, len(X_sample))])
        log.info("SHAP GradientExplainer computed successfully.")
        return shap_vals, "shap_gradient"
    except Exception as exc:
        log.warning(f"SHAP GradientExplainer also failed ({exc}) — using permutation.")

    # ── Permutation importance (feature-level) ─────────────────────────────
    log.info("Computing permutation feature importance (fallback).")
    return _permutation_importance(model, X_sample), "permutation"


def _permutation_importance(model,
                             X: np.ndarray,
                             n_repeats: int = 5) -> np.ndarray:
    """
    Estimate feature importance by permuting each feature column and
    measuring the drop in mean predicted probability of the top class.

    Returns
    -------
    importance : array of shape (n_features,) — higher = more important.
    """
    baseline_probs = model.predict(X, verbose=0)
    baseline_score = np.mean(np.max(baseline_probs, axis=1))

    n_features  = X.shape[-1]
    importances = np.zeros(n_features)

    for f in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            # Shuffle feature f across the time axis for all samples
            for i in range(len(X_perm)):
                np.random.shuffle(X_perm[i, :, f])
            probs  = model.predict(X_perm, verbose=0)
            score  = np.mean(np.max(probs, axis=1))
            scores.append(score)
        importances[f] = baseline_score - np.mean(scores)

    return importances
