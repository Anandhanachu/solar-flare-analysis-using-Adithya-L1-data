"""
explain.py
==========
Explainability module using Integrated Gradients for the solar flare LSTM.

Computes feature attribution and temporal attribution scores.
Generates Integrated Gradient heatmaps and plots.
Saves plots automatically to outputs directory.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

from .config import Config
from .feature_engineering import FEATURE_NAMES
from .utils import get_logger

_LOG = get_logger("explain")


class IntegratedGradients:
    """
    Computes Integrated Gradients for a TensorFlow/Keras model.
    Integrated Gradients assigns an importance score to each input feature
    by approximating the integral of gradients along a straight path from a
    baseline input to the actual input.
    """
    def __init__(self, model, m_steps=50):
        self.model = model
        self.m_steps = m_steps

    @tf.function
    def compute_gradients(self, inputs, target_class_idx):
        """Computes gradients of the target class probability wrt inputs."""
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            preds = self.model(inputs)
            # Gather the predicted probability for the target class
            target_probs = preds[:, target_class_idx]
        return tape.gradient(target_probs, inputs)

    def compute_integrated_gradients(self, inputs, baseline, target_class_idx):
        """
        Computes integrated gradients for a single input or batch of inputs.
        
        inputs: shape (batch, window, features)
        baseline: shape (batch, window, features) or broadcastable
        """
        # 1. Generate interpolated inputs
        alphas = tf.linspace(start=0.0, stop=1.0, num=self.m_steps + 1)
        alphas = tf.reshape(alphas, [self.m_steps + 1, 1, 1, 1])
        
        # Expand dims for broadcasting: (m_steps+1, batch, window, features)
        inputs_exp = tf.expand_dims(inputs, 0)
        baseline_exp = tf.expand_dims(baseline, 0)
        
        path_inputs = baseline_exp + alphas * (inputs_exp - baseline_exp)
        
        # 2. Compute gradients along the path
        # Flatten the batch dimension to feed into compute_gradients
        path_shape = tf.shape(path_inputs)
        flat_path = tf.reshape(path_inputs, [-1, path_shape[2], path_shape[3]])
        
        grads = self.compute_gradients(flat_path, target_class_idx)
        
        # Reshape gradients back to (m_steps+1, batch, window, features)
        grads = tf.reshape(grads, path_shape)
        
        # 3. Average the gradients (excluding the 0-th step to match trapezoidal rule roughly)
        avg_grads = tf.reduce_mean(grads[:-1], axis=0)
        
        # 4. Multiply by (inputs - baseline)
        integrated_gradients = (inputs - baseline) * avg_grads
        return integrated_gradients.numpy()


def generate_ig_explanations(model, X_train: np.ndarray, X_test: np.ndarray, cfg: Config):
    """
    Generate Integrated Gradients plots for the trained model.
    """
    _LOG.info("Generating Integrated Gradients explanations...")
    
    try:
        ig_explainer = IntegratedGradients(model, m_steps=50)
        
        # We compute IG for a subset of the test set to save time
        test_size = min(50, len(X_test))
        # Select highest flare events if possible, or just random
        test_sample = X_test[:test_size]
        
        # Convert to tensor
        test_sample_tf = tf.convert_to_tensor(test_sample, dtype=tf.float32)
        
        # Baseline: zero baseline by default, or mean training sequence
        # We will use the mean of the training data as the baseline
        mean_baseline = np.mean(X_train, axis=0, keepdims=True)
        baseline_tf = tf.convert_to_tensor(np.repeat(mean_baseline, test_size, axis=0), dtype=tf.float32)
        
        # We will explain the most critical class (Class 3 / X Class if available, else M Class)
        target_class = min(3, cfg.n_classes - 1)
        title_suffix = f"(Class {cfg.class_names[target_class]})"
        
        # Compute IG
        ig_values = ig_explainer.compute_integrated_gradients(test_sample_tf, baseline_tf, target_class)
        # ig_values shape: (test_size, window_size, n_features)
        
        # 1. Global Feature Importance (Aggregated over time and samples)
        # Sum absolute IG values across the time dimension, then mean over test samples
        ig_feature = np.mean(np.sum(np.abs(ig_values), axis=1), axis=0)
        
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(FEATURE_NAMES))
        # Sort features by importance
        sorted_idx = np.argsort(ig_feature)
        plt.barh(y_pos, ig_feature[sorted_idx], align='center', color='coral')
        plt.yticks(y_pos, [FEATURE_NAMES[i] for i in sorted_idx])
        plt.xlabel('Mean Absolute Integrated Gradient')
        plt.title(f'IG Global Feature Importance {title_suffix}')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, "ig_feature_importance.png"), dpi=150)
        plt.close()
        
        # 2. Time Importance (Aggregated over features and samples)
        ig_time = np.mean(np.sum(np.abs(ig_values), axis=2), axis=0)
        
        plt.figure(figsize=(10, 4))
        plt.plot(range(-cfg.window_size, 0), ig_time, marker='o', markersize=3, color='steelblue')
        plt.title(f"IG Time-Step Importance {title_suffix}")
        plt.xlabel("Time step relative to prediction point (minutes)")
        plt.ylabel("Mean Absolute IG Value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, "ig_time_importance.png"), dpi=150)
        plt.close()
        
        # 3. IG Heatmap (for a single prominent test sample)
        # Take the sample with the highest sum of IG values
        best_sample_idx = np.argmax(np.sum(np.abs(ig_values), axis=(1, 2)))
        heatmap_data = ig_values[best_sample_idx].T # Shape: (n_features, window_size)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm', origin='lower')
        plt.colorbar(label='Integrated Gradient')
        plt.yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES)
        plt.xlabel("Time step relative to prediction point")
        plt.title(f"IG Heatmap for Sample #{best_sample_idx} {title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, "ig_heatmap.png"), dpi=150)
        plt.close()
        
        _LOG.info("Integrated Gradients plots saved to outputs directory.")
        
    except Exception as e:
        _LOG.warning(f"Integrated Gradients computation failed: {e}")
