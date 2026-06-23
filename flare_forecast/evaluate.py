"""
evaluate.py
===========
Comprehensive evaluation of the trained solar flare forecasting model.

Metrics computed
----------------
- Accuracy
- Per-class and macro Precision, Recall, F1
- Full Classification Report (sklearn)
- Confusion Matrix (raw + normalized)
- ROC-AUC (one-vs-rest, per class + macro)
- Precision-Recall AUC per class
- True Skill Statistic (TSS)  — key operational space weather metric

True Skill Statistic (TSS)
--------------------------
TSS = Recall(flare) − False Alarm Ratio
    = (TP / (TP + FN)) − (FP / (FP + TN))

For multiclass: TSS is computed per class in a one-vs-rest manner,
then averaged across flare classes (1, 2, 3).  TSS ∈ [−1, 1];
random guessing → 0, perfect → 1.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

from .config import Config
from .utils import save_json, get_logger

_LOG = get_logger("evaluate")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  –  True Skill Statistic
# ══════════════════════════════════════════════════════════════════════════════

def true_skill_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    flare_classes: Tuple[int, ...] = (1, 2, 3),
) -> Dict:
    """
    Compute TSS for each class (one-vs-rest) and average across flare classes.

    TSS = Sensitivity − (1 − Specificity)
         = TP/(TP+FN)  −  FP/(FP+TN)

    Parameters
    ----------
    y_true       : Ground-truth integer labels.
    y_pred       : Predicted integer labels.
    n_classes    : Total number of classes.
    flare_classes: Which classes are "flare" classes (for averaging).

    Returns
    -------
    dict with per-class TSS and 'tss_macro' (mean over flare_classes).
    """
    tss_per_class = {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp                     # row sum - diagonal
        fp = cm[:, c].sum() - tp                     # col sum - diagonal
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn + 1e-12)         # recall
        specificity = tn / (tn + fp + 1e-12)
        tss         = sensitivity - (1 - specificity)
        tss_per_class[c] = float(tss)

    tss_flare = float(np.mean([tss_per_class[c] for c in flare_classes
                                if c in tss_per_class]))
    tss_per_class["tss_macro"] = tss_flare
    return tss_per_class


def heidke_skill_score(cm: np.ndarray) -> float:
    """
    Compute the multiclass Heidke Skill Score (HSS).
    """
    n = cm.sum()
    if n == 0: return 0.0
    trace = np.trace(cm)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected_correct = np.sum((row_sums * col_sums) / n)
    if n == expected_correct: return 0.0
    hss = (trace - expected_correct) / (n - expected_correct)
    return float(hss)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  Full Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    cfg:      Config,
) -> Dict:
    """
    Run full evaluation on the test set and persist all metrics.

    Parameters
    ----------
    model  : Trained Keras model.
    X_test : (N, window, features)
    y_test : (N,) integer ground-truth labels
    cfg    : Config.

    Returns
    -------
    dict with keys: 'accuracy', 'classification_report', 'confusion_matrix',
                    'roc_auc_macro', 'roc_auc_per_class', 'pr_auc_per_class',
                    'tss_per_class', 'tss_macro', 'per_class_metrics'
    """
    _LOG.info("Running evaluation on test set …")

    # ── Predictions ──────────────────────────────────────────────────────────
    proba   = model.predict(X_test, verbose=0)                    # (N, 4)
    y_pred  = np.argmax(proba, axis=1).astype(int)
    classes = list(range(cfg.n_classes))

    # ── Accuracy ──────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    _LOG.info(f"Accuracy: {acc:.4f}")

    # ── Missing class warnings ────────────────────────────────────────────────
    test_counts = np.bincount(y_test, minlength=cfg.n_classes)
    for c, count in enumerate(test_counts):
        if count == 0:
            _LOG.warning(f"  WARNING: Class {c} ({cfg.class_names[c]}) is missing from the test set!")

    # ── Per-class Precision / Recall / F1 ────────────────────────────────────
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )
    per_class = {}
    for i, name in enumerate(cfg.class_names):
        per_class[name] = {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }
        _LOG.info(
            f"  {name:12s}  P={prec[i]:.3f}  R={rec[i]:.3f}  "
            f"F1={f1[i]:.3f}  n={support[i]}"
        )

    # ── Classification report ─────────────────────────────────────────────────
    clf_report = classification_report(
        y_test, y_pred,
        labels=classes,
        target_names=cfg.class_names,
        zero_division=0,
    )
    _LOG.info(f"\nClassification Report:\n{clf_report}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm      = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    # ── ROC-AUC (one-vs-rest) ─────────────────────────────────────────────────
    y_bin = label_binarize(y_test, classes=classes)   # (N, 4)
    roc_auc_per_class = {}
    for i, name in enumerate(cfg.class_names):
        if y_bin[:, i].sum() == 0:
            roc_auc_per_class[name] = None
            continue
        auc = roc_auc_score(y_bin[:, i], proba[:, i])
        roc_auc_per_class[name] = float(auc)

    valid_aucs = [v for v in roc_auc_per_class.values() if v is not None]
    roc_auc_macro = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    _LOG.info(f"ROC-AUC macro: {roc_auc_macro:.4f}")

    # ── Precision-Recall AUC ──────────────────────────────────────────────────
    pr_auc_per_class = {}
    for i, name in enumerate(cfg.class_names):
        if y_bin[:, i].sum() == 0:
            pr_auc_per_class[name] = None
            continue
        prauc = average_precision_score(y_bin[:, i], proba[:, i])
        pr_auc_per_class[name] = float(prauc)

    # ── True Skill Statistic ──────────────────────────────────────────────────
    tss = true_skill_statistic(y_test, y_pred, cfg.n_classes)
    _LOG.info(f"TSS (macro over flare classes): {tss['tss_macro']:.4f}")

    # ── New Metrics: Balanced Acc, MCC, HSS ───────────────────────────────────
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc     = matthews_corrcoef(y_test, y_pred)
    hss     = heidke_skill_score(cm)
    
    _LOG.info(f"Balanced Accuracy: {bal_acc:.4f}")
    _LOG.info(f"MCC: {mcc:.4f}")
    _LOG.info(f"HSS: {hss:.4f}")

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "accuracy":              float(acc),
        "balanced_accuracy":     float(bal_acc),
        "mcc":                   float(mcc),
        "hss":                   float(hss),
        "classification_report": clf_report,
        "confusion_matrix":      cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "roc_auc_macro":         roc_auc_macro,
        "roc_auc_per_class":     roc_auc_per_class,
        "pr_auc_per_class":      pr_auc_per_class,
        "tss_per_class":         {cfg.class_names[k]: v
                                  for k, v in tss.items()
                                  if k != "tss_macro"},
        "tss_macro":             tss["tss_macro"],
        "per_class_metrics":     per_class,
        "y_pred":                y_pred.tolist(),
        "proba":                 proba.tolist(),
    }

    # Save
    save_json(
        {k: v for k, v in results.items() if k not in ("y_pred", "proba")},
        os.path.join(cfg.output_dir, "metrics.json")
    )
    _LOG.info("Metrics saved → metrics.json")

    # Save predictions for plotting
    pred_df = pd.DataFrame({
        "true_label": y_test,
        "pred_label": y_pred,
        **{f"prob_{cfg.class_names[i]}": proba[:, i]
           for i in range(cfg.n_classes)},
    })
    pred_df.to_csv(
        os.path.join(cfg.output_dir, "test_predictions.csv"), index=False
    )

    return results
