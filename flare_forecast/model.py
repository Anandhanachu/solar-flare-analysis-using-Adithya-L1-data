"""
model.py
========
LSTM + Bahdanau Attention neural network for multiclass solar flare
probability forecasting using TensorFlow / Keras.

Architecture
------------
    Input  (batch, window_size, n_features)
      |
      +-- LSTM(128, return_sequences=True)
      |
      +-- Dropout(0.3)
      |
      +-- LSTM(64, return_sequences=True)
      |
      +-- BahdanauAttention  <- custom single-head additive attention
      |
      +-- Dense(64, relu)
      |
      +-- Dropout(0.3)
      |
      +-- Dense(4, softmax)   <- class probabilities [Quiet, B/C, M, X]

Dependency
----------
TensorFlow >= 2.16 is REQUIRED.  If it is not installed, this module
raises an ImportError with clear installation instructions.
It does NOT substitute a different algorithm.

Python version
--------------
TensorFlow supports Python 3.9 - 3.12.
Python 3.14 (current venv) is NOT yet supported.
Python 3.12 is already installed on this machine -- use it.

Extensibility
-------------
`build_model(cfg)` dispatches by cfg.model_type ('lstm' | 'gru').
To add a Transformer: implement `_build_transformer_model()` and
register it in `_BUILDERS`.
"""

from __future__ import annotations

import numpy as np

from ._tf_check import require_tensorflow
require_tensorflow()   # raises ImportError with install instructions if TF absent

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers

from .config import Config
from .feature_engineering import N_FEATURES
from .utils import get_logger

_LOG = get_logger("model")
_LOG.info(f"TensorFlow {tf.__version__} loaded successfully.")

# ============================================================================
# SECTION 0 -- Custom Focal Loss
# ============================================================================

class SparseCategoricalFocalLoss(keras.losses.Loss):
    """
    Focal loss for sparse integer labels.
    """
    def __init__(self, gamma=2.0, alpha=0.25, name="sparse_categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1], dtype=y_pred.dtype)
        pt = tf.reduce_sum(y_onehot * y_pred, axis=-1)
        loss = - tf.math.pow(1. - pt, self.gamma) * tf.math.log(pt)
        return loss



# ============================================================================
# SECTION 1 -- Custom Bahdanau (Additive) Attention Layer
# ============================================================================

class BahdanauAttention(layers.Layer):
    """
    Single-head additive (Bahdanau) attention over a sequence.

    Given encoder hidden states h of shape (batch, T, d_model), computes
    a context vector:
        c = sum_t  alpha_t * h_t
    where:
        alpha_t = softmax(score_t)
        score_t = V^T * tanh(W * h_t)

    The output context vector c has shape (batch, d_model).

    Parameters
    ----------
    units : int
        Internal projection dimensionality.
    """

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W     = layers.Dense(units, use_bias=False)
        self.V     = layers.Dense(1,     use_bias=False)

    def call(self, encoder_output: tf.Tensor):
        """
        Parameters
        ----------
        encoder_output : tf.Tensor  shape (batch, T, d_model)

        Returns
        -------
        context : tf.Tensor  shape (batch, d_model)
        weights : tf.Tensor  shape (batch, T, 1)
        """
        score   = self.V(tf.nn.tanh(self.W(encoder_output)))  # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)                 # (batch, T, 1)
        context = tf.reduce_sum(weights * encoder_output, axis=1)  # (batch, d_model)
        return context, weights

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ============================================================================
# SECTION 2 -- LSTM Model Builder
# ============================================================================

def _build_lstm_model(cfg: Config) -> Model:
    """
    Build the primary LSTM + Bahdanau Attention + Dense Keras model.

    Layer stack
    -----------
    Input -> Bidirectional(LSTM(64)) -> LayerNormalization -> Dropout -> [Attention] ->
    Dense(32, relu) -> Dropout -> Dense(4, softmax)

    Parameters
    ----------
    cfg : Config
        Uses window_size, lstm_units_2 (for LSTM units), dense_units (override to 32),
        dropout_rate, use_attention, n_classes.

    Returns
    -------
    keras.Model  (uncompiled)
    """
    inp = keras.Input(
        shape=(cfg.window_size, N_FEATURES),
        name="input_sequence",
    )

    # Bidirectional LSTM block
    x = layers.Bidirectional(
        layers.LSTM(
            cfg.lstm_units_2, # Use 64 from config
            return_sequences=cfg.use_attention,
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        name="bilstm_1"
    )(inp)
    x = layers.LayerNormalization(name="ln_1")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_1")(x)

    # Bahdanau Attention
    if cfg.use_attention:
        context, _ = BahdanauAttention(
            units=cfg.lstm_units_2 // 2,
            name="bahdanau_attention",
        )(x)
        x = context   # (batch, lstm_units_2 * 2)

    # Dense classification head
    x = layers.Dense(
        32, # Configurable dense_units bypassed for Phase 2 strict requirement
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_2")(x)
    out = layers.Dense(cfg.n_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="SolarFlareLSTM")


# ============================================================================
# SECTION 3 -- GRU Model Builder (drop-in alternative)
# ============================================================================

def _build_gru_model(cfg: Config) -> Model:
    """
    GRU-based alternative to the LSTM model.
    Select via cfg.model_type = 'gru'.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    keras.Model  (uncompiled)
    """
    inp = keras.Input(shape=(cfg.window_size, N_FEATURES), name="input_sequence")
    
    # Bidirectional GRU block
    x = layers.Bidirectional(
        layers.GRU(
            cfg.lstm_units_2,
            return_sequences=cfg.use_attention,
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        name="bigru_1"
    )(inp)
    x = layers.LayerNormalization(name="ln_1")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_1")(x)

    if cfg.use_attention:
        context, _ = BahdanauAttention(
            units=cfg.lstm_units_2 // 2,
            name="bahdanau_attention",
        )(x)
        x = context
        
    x = layers.Dense(
        32, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_2")(x)
    out = layers.Dense(cfg.n_classes, activation="softmax", name="output")(x)
    return Model(inputs=inp, outputs=out, name="SolarFlareGRU")


# ============================================================================
# SECTION 4 -- Public Factory Function
# ============================================================================

# Registry: add new architectures here only.
_BUILDERS = {
    "lstm": _build_lstm_model,
    "gru":  _build_gru_model,
    # "transformer": _build_transformer_model,   # register here when ready
}


def build_model(cfg: Config) -> Model:
    """
    Build and compile a solar flare forecasting Keras model.

    Architecture is selected by cfg.model_type:
        'lstm'  -> Bidirectional(LSTM(64)) -> LayerNorm -> [Attention] -> Dense(32) -> Dense(4, softmax)
        'gru'   -> Bidirectional(GRU(64))  -> LayerNorm -> [Attention] -> Dense(32) -> Dense(4, softmax)

    Loss      : Sparse categorical cross-entropy.
    Optimizer : Adam (learning rate from cfg.learning_rate).
    Metrics   : accuracy.
    Note      : Class weights are applied in trainer.py, not here.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    keras.Model -- compiled, ready for model.fit().

    Raises
    ------
    ValueError
        If cfg.model_type is not registered in _BUILDERS.
    """
    builder = _BUILDERS.get(cfg.model_type.lower())
    if builder is None:
        raise ValueError(
            f"Unknown model_type='{cfg.model_type}'. "
            f"Available: {list(_BUILDERS.keys())}"
        )

    model     = builder(cfg)
    optimizer = keras.optimizers.Adam(
        learning_rate=cfg.learning_rate,
        clipnorm=cfg.clipnorm
    )
    
    if cfg.loss_function == "focal_loss":
        loss = SparseCategoricalFocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
    else:
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer = optimizer,
        loss      = loss,
        metrics   = ["accuracy"],
    )

    _LOG.info(f"Model '{cfg.model_type.upper()}' built and compiled.")
    _LOG.info(
        f"  Parameters: {model.count_params():,}  |  "
        f"Window: {cfg.window_size}  |  Features: {N_FEATURES}  |  "
        f"Classes: {cfg.n_classes}  |  Attention: {cfg.use_attention}"
    )
    return model
