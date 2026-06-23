"""
trainer.py
==========
Training loop for the solar flare LSTM model using TensorFlow / Keras.

Callbacks used
--------------
- EarlyStopping        : stops when val_loss does not improve for `patience` epochs,
                         restores best weights automatically.
- ReduceLROnPlateau    : halves the learning rate when val_loss plateaus.
- ModelCheckpoint      : saves the best model (lowest val_loss) to disk.
- CSVLogger            : writes per-epoch metrics to a CSV for auditing.

Class imbalance
---------------
Balanced class weights are computed from the training label distribution and
passed to model.fit() via the class_weight argument, ensuring that the rare
M- and X-class flare samples contribute proportionally more to the loss.
"""

from __future__ import annotations

import os
import time
import numpy as np
from typing import Dict

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from ._tf_check import require_tensorflow
require_tensorflow()

from tensorflow import keras

from .config import Config
from .utils import compute_class_weights, save_json, get_logger

_LOG = get_logger("trainer")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  –  Callback Factory
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCallback(keras.callbacks.Callback):
    """
    Computes macro F1 and TSS on the validation set at the end of each epoch.
    Allows EarlyStopping and ModelCheckpoint to monitor these metrics.
    """
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, n_classes: int):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.n_classes = n_classes

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Predict on validation data
        y_pred_prob = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Macro F1
        _, _, f1, _ = precision_recall_fscore_support(
            self.y_val, y_pred, labels=list(range(self.n_classes)), 
            zero_division=0, average="macro"
        )
        logs["val_f1_macro"] = float(f1)
        
        # TSS (macro over flare classes 1, 2, 3)
        cm = confusion_matrix(self.y_val, y_pred, labels=list(range(self.n_classes)))
        tss_list = []
        for c in range(1, self.n_classes): # typically flare classes
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            tss = sens - (1 - spec)
            tss_list.append(tss)
            
        logs["val_tss"] = float(np.mean(tss_list)) if tss_list else 0.0
        
        # Format the log output
        print(f" — val_f1_macro: {logs['val_f1_macro']:.4f} — val_tss: {logs['val_tss']:.4f}")


def build_callbacks(cfg: Config, X_val: np.ndarray, y_val: np.ndarray) -> list:
    """
    Construct the standard set of Keras training callbacks.

    1. EarlyStopping
       Monitors cfg.checkpoint_monitor. Stops training when it has not improved for
       cfg.early_stop_patience epochs. Restores best weights on stop.

    2. ReduceLROnPlateau
       Reduces the learning rate when val_loss plateaus.

    3. ModelCheckpoint
       Saves the full Keras model whenever cfg.checkpoint_monitor is the best seen.

    4. CSVLogger
       Appends per-epoch metrics to training_log.csv.
       
    5. MetricsCallback
       Computes val_f1_macro and val_tss.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir, "best_model.keras")

    monitor = getattr(cfg, 'checkpoint_monitor', 'val_loss')
    mode = "min" if monitor == "val_loss" else "max"

    early_stop = keras.callbacks.EarlyStopping(
        monitor              = monitor,
        mode                 = mode,
        patience             = cfg.early_stop_patience,
        restore_best_weights = True,
        verbose              = 1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor   = "val_loss",
        factor    = cfg.lr_reduce_factor,
        patience  = cfg.lr_reduce_patience,
        min_lr    = cfg.min_lr,
        verbose   = 1,
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath          = ckpt_path,
        monitor           = monitor,
        mode              = mode,
        save_best_only    = True,
        save_weights_only = False,
        verbose           = 1,
    )

    csv_log = keras.callbacks.CSVLogger(
        os.path.join(cfg.output_dir, "training_log.csv"),
        append=False,
    )

    metrics_cb = MetricsCallback(X_val, y_val, cfg.n_classes)

    _LOG.info(f"Best model checkpoint → {ckpt_path} (monitoring: {monitor})")
    return [metrics_cb, early_stop, reduce_lr, checkpoint, csv_log]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    cfg:     Config,
) -> keras.callbacks.History:
    """
    Train the compiled Keras model with balanced class weighting.

    Parameters
    ----------
    model   : Compiled Keras model from model.build_model().
    X_train : np.ndarray of shape (N_train, window_size, n_features)
    y_train : np.ndarray of shape (N_train,)  — integer class labels
    X_val   : np.ndarray of shape (N_val, window_size, n_features)
    y_val   : np.ndarray of shape (N_val,)    — integer class labels
    cfg     : Config

    Returns
    -------
    keras.callbacks.History
        Contains per-epoch 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
        Also saved to cfg.output_dir/training_history.json.
    """
    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weights(y_train, cfg.n_classes)
    _LOG.info(
        f"Class weights: "
        + "  ".join(f"C{k}={v:.2f}" for k, v in class_weights.items())
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = build_callbacks(cfg, X_val, y_val)

    # Count parameters
    trainable_count = np.sum([keras.backend.count_params(w) for w in model.trainable_weights])
    
    _LOG.info("="*60)
    _LOG.info("              TRAINING CONFIGURATION")
    _LOG.info("="*60)
    _LOG.info(f"Model       : {cfg.model_type.upper()}")
    _LOG.info(f"Loss        : {cfg.loss_function}")
    _LOG.info(f"Monitor     : {getattr(cfg, 'checkpoint_monitor', 'val_loss')}")
    _LOG.info(f"Epochs      : {cfg.epochs}")
    _LOG.info(f"Batch size  : {cfg.batch_size}")
    _LOG.info(f"Base LR     : {cfg.learning_rate}")
    _LOG.info(f"Train size  : {len(X_train)} sequences")
    _LOG.info(f"Val size    : {len(X_val)} sequences")
    _LOG.info(f"Parameters  : {trainable_count:,} trainable")
    _LOG.info("="*60)

    # ── model.fit ─────────────────────────────────────────────────────────────
    history = model.fit(
        X_train,
        y_train,
        validation_data = (X_val, y_val),
        epochs          = cfg.epochs,
        batch_size      = cfg.batch_size,
        class_weight    = class_weights,
        callbacks       = callbacks,
        shuffle         = False,   # preserve temporal ordering within batches
        verbose         = 1,
    )

    # ── Persist history ───────────────────────────────────────────────────────
    hist_path    = os.path.join(cfg.output_dir, "training_history.json")
    history_dict = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    save_json(history_dict, hist_path)
    _LOG.info(f"Training history saved → {hist_path}")

    monitor = getattr(cfg, 'checkpoint_monitor', 'val_loss')
    best_epoch = int(np.argmin(history.history[monitor])) + 1 if monitor == "val_loss" else int(np.argmax(history.history[monitor])) + 1
    best_val   = float(min(history.history[monitor])) if monitor == "val_loss" else float(max(history.history[monitor]))
    _LOG.info(
        f"Training complete  |  best_epoch={best_epoch}  "
        f"best_{monitor}={best_val:.4f}"
    )

    return history
