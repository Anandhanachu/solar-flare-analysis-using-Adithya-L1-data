"""
calibration.py
==============
Probability calibration module for the solar flare LSTM.
Implements Temperature Scaling (default) and Isotonic Regression.
Computes Expected Calibration Error (ECE), Brier Score, and generates
reliability diagrams.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from .config import Config
from .utils import get_logger

_LOG = get_logger("calibration")

def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE) for multi-class using max probability."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences >= bins[i]) & (confidences <= bins[i+1])
        if np.sum(in_bin) > 0:
            acc_bin = np.mean(accuracies[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_bin - conf_bin) * np.sum(in_bin) / len(y_true)
    return ece

def compute_brier_score(y_true, y_prob, n_classes):
    """Computes Multi-class Brier Score."""
    # Convert y_true to one-hot
    y_true_oh = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_prob - y_true_oh)**2, axis=1))


class TemperatureScaling:
    def __init__(self):
        self.temperature = 1.0
        
    def fit(self, logits, y_true):
        def nll(t):
            t_logits = logits / t[0]
            max_l = np.max(t_logits, axis=1, keepdims=True)
            exp_l = np.exp(t_logits - max_l)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.sum(np.log(probs[np.arange(len(y_true)), y_true]))
            
        res = minimize(nll, [1.0], bounds=[(0.01, 10.0)])
        self.temperature = res.x[0]
        _LOG.info(f"Temperature Scaling fitted. T = {self.temperature:.4f}")
        
    def predict_proba(self, logits):
        t_logits = logits / self.temperature
        max_l = np.max(t_logits, axis=1, keepdims=True)
        exp_l = np.exp(t_logits - max_l)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)


class MultiClassIsotonicRegression:
    def __init__(self):
        self.ir_models = []
        self.n_classes = 0
        
    def fit(self, probs, y_true, n_classes):
        self.n_classes = n_classes
        for c in range(n_classes):
            ir = IsotonicRegression(out_of_bounds='clip')
            y_c = (y_true == c).astype(float)
            ir.fit(probs[:, c], y_c)
            self.ir_models.append(ir)
        _LOG.info("Isotonic Regression fitted for all classes.")
            
    def predict_proba(self, probs):
        calibrated = np.zeros_like(probs)
        for c in range(self.n_classes):
            calibrated[:, c] = self.ir_models[c].predict(probs[:, c])
        s = np.sum(calibrated, axis=1, keepdims=True)
        s[s == 0] = 1.0
        return calibrated / s


class Calibrator:
    """Wrapper that handles logits extraction and model saving/loading."""
    def __init__(self, method="temperature", n_classes=4):
        self.method = method
        self.n_classes = n_classes
        if method == "temperature":
            self.model = TemperatureScaling()
        elif method == "isotonic":
            self.model = MultiClassIsotonicRegression()
        else:
            self.model = None

    def _probs_to_logits(self, probs):
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return np.log(probs)

    def fit(self, y_prob, y_true):
        if self.method == "none" or self.model is None:
            return
        
        if self.method == "temperature":
            logits = self._probs_to_logits(y_prob)
            self.model.fit(logits, y_true)
        elif self.method == "isotonic":
            self.model.fit(y_prob, y_true, self.n_classes)

    def predict_proba(self, y_prob):
        if self.method == "none" or self.model is None:
            return y_prob
            
        if self.method == "temperature":
            logits = self._probs_to_logits(y_prob)
            return self.model.predict_proba(logits)
        elif self.method == "isotonic":
            return self.model.predict_proba(y_prob)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        if not os.path.exists(filepath):
            return Calibrator(method="none")
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def plot_reliability_diagram(y_true, y_prob_raw, y_prob_calibrated, cfg: Config):
    """Plots the reliability diagram and confidence histogram."""
    n_bins = 10
    
    # We plot the diagram for the highest confidence predictions
    def _get_bin_stats(y_p):
        confidences = np.max(y_p, axis=1)
        predictions = np.argmax(y_p, axis=1)
        accuracies = (predictions == y_true)
        
        bins = np.linspace(0, 1, n_bins + 1)
        acc_b = []
        conf_b = []
        for i in range(n_bins):
            in_bin = (confidences >= bins[i]) & (confidences <= bins[i+1])
            if np.sum(in_bin) > 0:
                acc_b.append(np.mean(accuracies[in_bin]))
                conf_b.append(np.mean(confidences[in_bin]))
        return confidences, np.array(conf_b), np.array(acc_b)

    raw_confs, raw_c, raw_a = _get_bin_stats(y_prob_raw)
    cal_confs, cal_c, cal_a = _get_bin_stats(y_prob_calibrated)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Reliability Diagram
    ax1.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax1.plot(raw_c, raw_a, "s-", label="Raw Softmax")
    ax1.plot(cal_c, cal_a, "o-", label="Calibrated")
    ax1.set_ylabel("Empirical Accuracy")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence Histogram
    ax2.hist(raw_confs, bins=10, range=(0, 1), alpha=0.5, label="Raw Softmax")
    ax2.hist(cal_confs, bins=10, range=(0, 1), alpha=0.5, label="Calibrated")
    ax2.set_xlabel("Predicted Confidence")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Confidence Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "reliability_diagram.png"), dpi=150)
    plt.close()


def run_calibration(model, X_val, y_val, cfg: Config):
    """
    Runs the calibration phase on the validation dataset, computes metrics,
    and saves the calibrator model and plots.
    """
    _LOG.info(f"Running Probability Calibration (method={cfg.calibration_method})...")
    
    # Get raw predictions on validation set
    y_prob_raw = model.predict(X_val, batch_size=cfg.batch_size, verbose=0)
    
    # Initialize and fit calibrator
    calibrator = Calibrator(method=cfg.calibration_method, n_classes=cfg.n_classes)
    calibrator.fit(y_prob_raw, y_val)
    
    # Save calibrator
    calibrator_path = os.path.join(cfg.output_dir, "calibrator.pkl")
    calibrator.save(calibrator_path)
    
    # Apply calibration
    y_prob_cal = calibrator.predict_proba(y_prob_raw)
    
    # Compute metrics
    ece_raw = compute_ece(y_val, y_prob_raw)
    ece_cal = compute_ece(y_val, y_prob_cal)
    
    brier_raw = compute_brier_score(y_val, y_prob_raw, cfg.n_classes)
    brier_cal = compute_brier_score(y_val, y_prob_cal, cfg.n_classes)
    
    _LOG.info(f"  Raw Softmax -> ECE: {ece_raw:.4f}  |  Brier: {brier_raw:.4f}")
    _LOG.info(f"  Calibrated  -> ECE: {ece_cal:.4f}  |  Brier: {brier_cal:.4f}")
    
    # Plot diagrams
    plot_reliability_diagram(y_val, y_prob_raw, y_prob_cal, cfg)
    _LOG.info("Calibration completed and plots saved.")
