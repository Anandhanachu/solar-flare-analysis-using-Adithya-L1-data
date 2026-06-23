"""
visualize.py
============
All visualization functions for the solar flare LSTM pipeline.

Plots generated
---------------
1. training_curves.png       — loss & accuracy vs epoch
2. confusion_matrix.png      — normalized 4×4 heatmap
3. roc_curves.png            — per-class ROC with AUC
4. pr_curves.png             — Precision-Recall per class
5. probability_distribution.png — predicted probability histograms
6. prediction_timeline.png   — predicted vs true class over test period
7. shap_summary.png          — SHAP / permutation feature importance
8. operational_forecast.png  — styled probability bar chart

All plots use a dark-mode, publication-quality style.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

from .config import Config
from .feature_engineering import FEATURE_NAMES
from .utils import get_logger

_LOG = get_logger("visualize")

# ── Dark-mode style ────────────────────────────────────────────────────────
_DARK = {
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#0D1117",
    "text.color":        "#E6EDF3",
    "axes.labelcolor":   "#E6EDF3",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "grid.color":        "#21262D",
    "grid.linestyle":    ":",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "legend.fontsize":   9,
    "figure.dpi":        130,
}

CLASS_COLORS = ["#58A6FF", "#3FB950", "#FFA657", "#F85149"]
CLASS_PALETTES = {
    0: "#58A6FF",  # Quiet — blue
    1: "#3FB950",  # B/C  — green
    2: "#FFA657",  # M    — orange
    3: "#F85149",  # X    — red
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Training Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history_dict: Dict, cfg: Config) -> str:
    """
    Plot training and validation loss + accuracy across epochs.

    Parameters
    ----------
    history_dict : dict from keras History.history.
    cfg          : Config.

    Returns
    -------
    Saved file path.
    """
    with plt.rc_context(_DARK):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LSTM Training Curves  |  Aditya-L1 Flare Forecast",
                     fontsize=13, color="#E6EDF3")

        epochs = range(1, len(history_dict["loss"]) + 1)

        # Loss
        ax = axes[0]
        ax.plot(epochs, history_dict["loss"],     color="#58A6FF", lw=1.8,
                label="Train Loss")
        ax.plot(epochs, history_dict["val_loss"], color="#FFA657", lw=1.8,
                ls="--", label="Val Loss")
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Sparse Categorical Cross-Entropy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1]
        ax.plot(epochs, history_dict.get("accuracy", []),
                color="#58A6FF", lw=1.8, label="Train Acc")
        ax.plot(epochs, history_dict.get("val_accuracy", []),
                color="#FFA657", lw=1.8, ls="--", label="Val Acc")
        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(cfg.output_dir, "training_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    _LOG.info(f"Saved: training_curves.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 2. Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm_norm: List[List[float]], cfg: Config) -> str:
    """
    Plot the normalized 4×4 confusion matrix as a heatmap.

    Parameters
    ----------
    cm_norm : Normalized confusion matrix (rows = true, cols = predicted).
    cfg     : Config.
    """
    cm = np.array(cm_norm)

    with plt.rc_context(_DARK):
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle("Confusion Matrix (Normalized)  |  Flare Forecast",
                     fontsize=12, color="#E6EDF3")

        cmap = LinearSegmentedColormap.from_list(
            "dark_blues", ["#0D1117", "#0D419D", "#58A6FF"]
        )
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Fraction")

        ax.set_xticks(range(cfg.n_classes))
        ax.set_yticks(range(cfg.n_classes))
        ax.set_xticklabels(cfg.class_names, rotation=30, ha="right",
                            color="#E6EDF3")
        ax.set_yticklabels(cfg.class_names, color="#E6EDF3")
        ax.set_xlabel("Predicted Class", color="#E6EDF3")
        ax.set_ylabel("True Class",      color="#E6EDF3")

        for i in range(cfg.n_classes):
            for j in range(cfg.n_classes):
                val   = cm[i, j]
                color = "white" if val < 0.5 else "#0D1117"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(cfg.output_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    _LOG.info("Saved: confusion_matrix.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROC Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(y_true: np.ndarray,
                    proba:  np.ndarray,
                    cfg:    Config) -> str:
    """
    Plot one-vs-rest ROC curves for each class.

    Parameters
    ----------
    y_true : (N,) integer ground-truth labels.
    proba  : (N, 4) predicted probabilities.
    cfg    : Config.
    """
    y_bin   = label_binarize(y_true, classes=list(range(cfg.n_classes)))
    classes = list(range(cfg.n_classes))

    with plt.rc_context(_DARK):
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.suptitle("ROC Curves (One-vs-Rest)  |  Flare Forecast",
                     fontsize=12, color="#E6EDF3")

        for i, name in enumerate(cfg.class_names):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
                    label=f"{name}  (AUC={roc_auc_val:.3f})")

        ax.plot([0, 1], [0, 1], color="#30363D", lw=1, ls="--",
                label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        plt.tight_layout()
        path = os.path.join(cfg.output_dir, "roc_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    _LOG.info("Saved: roc_curves.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 4. Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr_curves(y_true: np.ndarray,
                   proba:  np.ndarray,
                   cfg:    Config) -> str:
    """Plot Precision-Recall curves per class."""
    y_bin = label_binarize(y_true, classes=list(range(cfg.n_classes)))

    with plt.rc_context(_DARK):
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.suptitle("Precision-Recall Curves  |  Flare Forecast",
                     fontsize=12, color="#E6EDF3")

        for i, name in enumerate(cfg.class_names):
            if y_bin[:, i].sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
            pr_auc_val = auc(rec, prec)
            ax.step(rec, prec, color=CLASS_COLORS[i], lw=2, where="post",
                    label=f"{name}  (AP={pr_auc_val:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        path = os.path.join(cfg.output_dir, "pr_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    _LOG.info("Saved: pr_curves.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 5. Probability Distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_probability_distribution(proba: np.ndarray, cfg: Config) -> str:
    """
    Histogram of predicted probability for each class.

    Parameters
    ----------
    proba : (N, 4) predicted probabilities.
    cfg   : Config.
    """
    with plt.rc_context(_DARK):
        fig, axes = plt.subplots(1, cfg.n_classes,
                                 figsize=(14, 4), sharey=True)
        fig.suptitle("Predicted Probability Distributions  |  Flare Forecast",
                     fontsize=12, color="#E6EDF3")

        for i, (ax, name) in enumerate(zip(axes, cfg.class_names)):
            ax.hist(proba[:, i], bins=40, color=CLASS_COLORS[i],
                    alpha=0.80, edgecolor="none")
            ax.set_title(name)
            ax.set_xlabel("Probability")
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_ylabel("Count")

        plt.tight_layout()
        path = os.path.join(cfg.output_dir, "probability_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    _LOG.info("Saved: probability_distribution.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 6. Prediction Timeline
# ══════════════════════════════════════════════════════════════════════════════

def plot_prediction_timeline(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              proba:  np.ndarray,
                              cfg:    Config) -> str:
    """
    Show predicted vs true class over the test period as a time-ordered
    categorical scatter plot with probability envelope.

    Parameters
    ----------
    y_true : (N,) true labels.
    y_pred : (N,) predicted labels.
    proba  : (N, 4) predicted probabilities.
    cfg    : Config.
    """
    N    = len(y_true)
    x    = np.arange(N)

    with plt.rc_context(_DARK):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8),
                                        gridspec_kw={"height_ratios": [1.5, 1]})
        fig.suptitle("Prediction Timeline  |  Test Set  |  Flare Forecast",
                     fontsize=12, color="#E6EDF3")

        # Class probability stacked area
        ax1.stackplot(
            x,
            proba[:, 0], proba[:, 1], proba[:, 2], proba[:, 3],
            colors   = [c + "AA" for c in CLASS_COLORS],
            labels   = cfg.class_names,
            baseline = "zero",
        )
        ax1.set_ylabel("Probability")
        ax1.set_title("Stacked Class Probabilities")
        ax1.legend(loc="upper right", ncol=4, fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.2)

        # True vs predicted class
        ax2.scatter(x, y_true,  color="#8B949E", s=5, alpha=0.6,
                    label="True Class")
        ax2.scatter(x, y_pred,  color="#FFA657", s=5, alpha=0.6,
                    label="Predicted Class", marker="x")
        ax2.set_yticks(range(cfg.n_classes))
        ax2.set_yticklabels(cfg.class_names, fontsize=8, color="#E6EDF3")
        ax2.set_xlabel("Test Sample Index")
        ax2.set_ylabel("Class")
        ax2.set_title("True vs Predicted Class")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        path = os.path.join(cfg.output_dir, "prediction_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    _LOG.info("Saved: prediction_timeline.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 7. SHAP / Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(importance: np.ndarray,
                             method:     str,
                             cfg:        Config) -> str:
    """
    Bar chart of feature importance scores.

    Parameters
    ----------
    importance : 1-D array of length n_features (SHAP mean|abs or permutation).
    method     : String label ("shap_deep", "shap_gradient", "permutation").
    cfg        : Config.
    """
    feat_names = FEATURE_NAMES
    n_feat     = len(feat_names)

    # Mean absolute SHAP if shape is (n_explain, window, n_features, n_class)
    if importance.ndim > 1:
        # For list-of-arrays (one per class) take mean of class 2+3 (flare)
        importance = np.abs(importance).mean(axis=(0, 1)) \
            if importance.ndim == 3 else np.abs(importance).mean(axis=0)

    importance = np.abs(importance).flatten()[:n_feat]
    order      = np.argsort(importance)[::-1]

    with plt.rc_context(_DARK):
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle(f"Feature Importance ({method})  |  Flare Forecast",
                     fontsize=12, color="#E6EDF3")

        bars = ax.barh(
            [feat_names[i] for i in order],
            importance[order],
            color="#58A6FF", alpha=0.85
        )
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(cfg.output_dir, "shap_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    _LOG.info("Saved: shap_summary.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 8. Operational Forecast Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def plot_operational_forecast(proba_vec: np.ndarray,
                               risk_level: str,
                               cfg:        Config,
                               timestamp:  str = "") -> str:
    """
    Publication-quality operational forecast probability bar chart.

    Parameters
    ----------
    proba_vec  : (4,) probability vector [Quiet, B/C, M, X].
    risk_level : "LOW" | "MODERATE" | "HIGH" | "EXTREME".
    cfg        : Config.
    timestamp  : Optional observation timestamp string.

    Returns
    -------
    Saved file path.
    """
    risk_color = {
        "LOW":      "#3FB950",
        "MODERATE": "#FFA657",
        "HIGH":     "#F85149",
        "EXTREME":  "#DA3633",
    }.get(risk_level, "#8B949E")

    with plt.rc_context(_DARK):
        fig = plt.figure(figsize=(10, 6))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
        ax_bar  = fig.add_subplot(gs[0])
        ax_risk = fig.add_subplot(gs[1])

        # Bar chart
        bars = ax_bar.bar(
            cfg.class_names,
            proba_vec * 100,
            color    = CLASS_COLORS,
            edgecolor= "none",
            width    = 0.55,
        )
        for bar, p in zip(bars, proba_vec):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{p*100:.1f}%",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#E6EDF3",
            )

        ax_bar.set_ylim(0, 115)
        ax_bar.set_ylabel("Probability (%)", fontsize=11)
        ax_bar.set_title(
            f"Aditya-L1  Solar Flare Operational Forecast"
            + (f"   [{timestamp}]" if timestamp else ""),
            fontsize=13, fontweight="bold", color="#E6EDF3",
        )
        ax_bar.grid(True, axis="y", alpha=0.2)
        ax_bar.tick_params(axis="x", labelsize=12, colors="#E6EDF3")

        # Risk ribbon
        ax_risk.set_facecolor(risk_color + "33")
        ax_risk.text(
            0.5, 0.5,
            f"⚡  Risk Level:  {risk_level}",
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            color=risk_color,
            transform=ax_risk.transAxes,
        )
        ax_risk.set_axis_off()

        # Border
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#21262D")

        path = os.path.join(cfg.output_dir, "operational_forecast.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    _LOG.info("Saved: operational_forecast.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: run all evaluation plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_evaluation(results: Dict,
                         history_dict: Dict,
                         cfg:          Config) -> None:
    """
    Generate all evaluation plots from the results dict returned by
    evaluate.evaluate_model().

    Parameters
    ----------
    results      : dict from evaluate_model().
    history_dict : dict from trainer training_history.json.
    cfg          : Config.
    """
    y_true = np.array(results.get("y_pred_true",
                      # fallback: reconstruct from confusion matrix diagonal
                      []))
    y_pred = np.array(results["y_pred"])
    proba  = np.array(results["proba"])

    # 1. Training curves
    if history_dict:
        plot_training_curves(history_dict, cfg)

    # 2. Confusion matrix
    plot_confusion_matrix(results["confusion_matrix_norm"], cfg)

    # 3. ROC curves — need y_true; reconstruct from predictions if needed
    if len(y_true) == len(y_pred):
        plot_roc_curves(y_true, proba, cfg)
        plot_pr_curves(y_true, proba, cfg)
        plot_prediction_timeline(y_true, y_pred, proba, cfg)

    # 4. Probability distribution
    plot_probability_distribution(proba, cfg)
