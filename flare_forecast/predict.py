"""
predict.py
==========
Operational forecasting interface for the trained solar flare LSTM model.

Requires TensorFlow / Keras.  If TensorFlow is not installed this module
raises an ImportError with clear installation instructions — it does NOT
substitute a different algorithm.

Provides
--------
OperationalForecaster — loads model + scaler and generates class-probability
                        forecasts with a risk-level classification.

Risk levels (based on P(M) + P(X))
-----------------------------------
LOW      : P(M+X) < 0.15   — routine monitoring
MODERATE : 0.15 ≤ P(M+X) < 0.40  — heightened awareness
HIGH     : 0.40 ≤ P(M+X) < 0.70  — alert
EXTREME  : P(M+X) ≥ 0.70  — emergency

Usage
-----
    from flare_forecast.predict import OperationalForecaster
    fc = OperationalForecaster(cfg)
    result = fc.forecast_latest()
    fc.print_forecast(result)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from ._tf_check import require_tensorflow
require_tensorflow()

from tensorflow import keras

from .config import Config
from .dataset import load_dataset
from .feature_engineering import build_features, FEATURE_NAMES
from .preprocessing import clean_features, apply_scaler
from .model import BahdanauAttention
from .calibration import Calibrator
from .utils import load_pickle, get_logger
from .visualize import plot_operational_forecast

_LOG = get_logger("predict")


class OperationalForecaster:
    """
    End-to-end operational solar flare forecaster using the trained LSTM model.

    Loads a saved Keras model (.keras) and fitted StandardScaler (.pkl)
    from cfg.output_dir, then produces per-class probability forecasts
    and risk-level assessments.

    Parameters
    ----------
    cfg         : Config — contains output paths, thresholds, class names.
    model_path  : Optional override path to the .keras model file.
    scaler_path : Optional override path to the scaler.pkl file.

    Raises
    ------
    FileNotFoundError
        If the model or scaler files are not found.  Run
        `python -m flare_forecast.main train` first.
    """

    def __init__(
        self,
        cfg:         Config,
        model_path:  Optional[str] = None,
        scaler_path: Optional[str] = None,
    ):
        self.cfg = cfg

        # ── Load Keras model ───────────────────────────────────────────────────
        mp = model_path or os.path.join(cfg.output_dir, "best_model.keras")
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"Trained LSTM model not found at:\n  {mp}\n\n"
                "Run training first:\n"
                "  python -m flare_forecast.main train"
            )

        _LOG.info(f"Loading LSTM model from {mp} …")
        self.model = keras.models.load_model(
            mp,
            custom_objects={"BahdanauAttention": BahdanauAttention},
        )
        _LOG.info("LSTM model loaded successfully.")

        # ── Load StandardScaler ────────────────────────────────────────────────
        sp = scaler_path or os.path.join(cfg.output_dir, "scaler.pkl")
        if not os.path.exists(sp):
            raise FileNotFoundError(
                f"Fitted scaler not found at:\n  {sp}\n\n"
                "Run training first:\n"
                "  python -m flare_forecast.main train"
            )

        self.scaler: StandardScaler = load_pickle(sp)
        _LOG.info("StandardScaler loaded.")

        # ── Load Calibrator ────────────────────────────────────────────────────
        cp = os.path.join(cfg.output_dir, "calibrator.pkl")
        self.calibrator = Calibrator.load(cp)
        _LOG.info(f"Calibrator loaded (method: {self.calibrator.method}).")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def forecast_latest(self) -> Dict:
        """
        Load the most recent FITS data, extract features, select the last
        window_size-minute window, and run inference.

        Returns
        -------
        dict with keys:
            'probabilities'   – {class_name: float probability} (calibrated)
            'raw_probabilities' - {class_name: float probability} (uncalibrated)
            'predicted_class' – str name of the highest-probability class
            'risk_level'      – str  "LOW" | "MODERATE" | "HIGH" | "EXTREME"
            'confidence_score' - float
            'prediction_entropy' - float
            'proba_vector'    – np.ndarray of shape (4,)
            'timestamp'       – UTC timestamp string of the last data point
        """
        _LOG.info("Loading latest FITS data for operational forecast …")
        raw_df  = load_dataset(self.cfg)
        feat_df = build_features(raw_df, self.cfg)
        feat_df = clean_features(feat_df)
        feat_sc = apply_scaler(feat_df, self.scaler)

        win = self.cfg.window_size
        if len(feat_sc) < win:
            raise ValueError(
                f"Not enough data for a {win}-step window "
                f"(got {len(feat_sc)} rows)."
            )

        last_window = feat_sc[FEATURE_NAMES].values[-win:].astype(np.float32)
        X           = last_window[np.newaxis, ...]   # shape: (1, win, n_features)
        timestamp   = str(feat_sc.index[-1])

        return self._predict(X, timestamp)

    def forecast_from_array(
        self,
        X:         np.ndarray,
        timestamp: str = "",
    ) -> Dict:
        """
        Run inference on a pre-built, pre-scaled feature window.

        Parameters
        ----------
        X         : np.ndarray of shape (window_size, n_features) or
                    (1, window_size, n_features).  Must already be scaled.
        timestamp : Optional display timestamp string.

        Returns
        -------
        Same dict as forecast_latest().
        """
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        return self._predict(X.astype(np.float32), timestamp)

    def print_forecast(self, result: Dict) -> None:
        """Print a formatted operational forecast to stdout."""
        bar_width = 30
        sep       = "═" * 54

        print(f"\n{sep}")
        print("  ADITYA-L1   SOLAR FLARE OPERATIONAL FORECAST")
        if result.get("timestamp"):
            print(f"  Timestamp : {result['timestamp']}")
        print(sep)
        print()

        for name, prob in result["probabilities"].items():
            filled  = int(prob * bar_width)
            bar     = "█" * filled + "░" * (bar_width - filled)
            raw_p   = result["raw_probabilities"][name]
            print(f"  {name:12s} : {bar}  {prob * 100:5.1f}%  (Raw: {raw_p * 100:5.1f}%)")

        print()
        print(f"  Predicted class   : {result['predicted_class']}")
        print(f"  Confidence Score  : {result['confidence_score'] * 100:.1f}%")
        print(f"  Prediction Entropy: {result['prediction_entropy']:.3f} nats")

        risk      = result["risk_level"]
        risk_icon = {
            "LOW": "🟢", "MODERATE": "🟡",
            "HIGH": "🔴", "EXTREME": "🚨",
        }.get(risk, "⚪")
        print(f"  Risk Level        : {risk_icon}  {risk}")
        print(sep + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _predict(self, X: np.ndarray, timestamp: str) -> Dict:
        """
        Core inference: run the LSTM model and package the result.

        Parameters
        ----------
        X         : (1, window_size, n_features) float32 array.
        timestamp : Display timestamp string.

        Returns
        -------
        Forecast result dict.
        """
        raw_proba_arr  = self.model.predict(X, verbose=0)
        cal_proba_arr  = self.calibrator.predict_proba(raw_proba_arr)[0]
        raw_proba_arr  = raw_proba_arr[0]
        
        pred_class = int(np.argmax(cal_proba_arr))
        risk_level = self._risk_level(cal_proba_arr)
        
        # Calculate entropy (uncertainty score)
        entropy = -np.sum(cal_proba_arr * np.log(cal_proba_arr + 1e-7))

        probabilities = {
            name: float(cal_proba_arr[i])
            for i, name in enumerate(self.cfg.class_names)
        }
        raw_probabilities = {
            name: float(raw_proba_arr[i])
            for i, name in enumerate(self.cfg.class_names)
        }

        result = {
            "probabilities":   probabilities,
            "raw_probabilities": raw_probabilities,
            "predicted_class": self.cfg.class_names[pred_class],
            "confidence_score": float(np.max(cal_proba_arr)),
            "prediction_entropy": float(entropy),
            "risk_level":      risk_level,
            "proba_vector":    cal_proba_arr,
            "timestamp":       timestamp,
        }

        # Save the forecast dashboard image
        plot_operational_forecast(cal_proba_arr, risk_level, self.cfg, timestamp)
        return result

    def _risk_level(self, proba_vec: np.ndarray) -> str:
        """
        Classify operational risk from P(M-class) + P(X-class).

        Thresholds (from cfg.risk_thresholds):
            EXTREME  ≥ 0.70
            HIGH     ≥ 0.40
            MODERATE ≥ 0.15
            LOW      <  0.15

        Parameters
        ----------
        proba_vec : (4,) probability array [Quiet, B/C, M, X].

        Returns
        -------
        str — one of "LOW", "MODERATE", "HIGH", "EXTREME".
        """
        p_significant = float(proba_vec[2] + proba_vec[3])   # P(M) + P(X)
        thr           = self.cfg.risk_thresholds

        if p_significant >= thr["EXTREME"]:
            return "EXTREME"
        if p_significant >= thr["HIGH"]:
            return "HIGH"
        if p_significant >= thr["MODERATE"]:
            return "MODERATE"
        return "LOW"
