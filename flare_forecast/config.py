"""
config.py
=========
Central configuration for the Aditya-L1 Solar Flare LSTM Forecasting System.

All hyperparameters, file paths, model settings, and threshold values are
defined here as a single dataclass, making the entire pipeline configurable
from one place. Future models (Transformer, GRU, XGBoost) can be selected
by changing `model_type` without touching any other file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict


# ── Base filesystem paths ──────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))   # flare_forecast/
_BASE   = os.path.dirname(_HERE)                        # ISRO,BHA/


@dataclass
class Config:
    """
    Master configuration dataclass.

    Every field has a sensible default. Override individual fields after
    instantiation:
        cfg = Config()
        cfg.window_size = 360   # 6-hour window
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    base_dir: str = _BASE

    solexs_gz: str = os.path.join(
        _BASE,
        r"SLX\AL1_SLX_L1_20260620_v1.0\SDD2\AL1_SOLEXS_20260620_SDD2_L1.lc.gz"
    )
    helios_files: List[str] = field(default_factory=lambda: [
        os.path.join(
            _BASE,
            r"HLS\2026\06\20\HLS_20260620_000008_43177sec_lev1_V111\czt\lightcurve_czt1.fits"
        ),
        os.path.join(
            _BASE,
            r"HLS\2026\06\20\HLS_20260620_121027_42563sec_lev1_V111\czt\lightcurve_czt1.fits"
        ),
    ])
    helios_bands: List[str] = field(default_factory=lambda: [
        "CZT1_LC_BAND_18.00KEV_TO_160.00KEV",   # broadband (primary)
        "CZT1_LC_BAND_20.00KEV_TO_40.00KEV",    # low-hard
        "CZT1_LC_BAND_80.00KEV_TO_150.00KEV",   # high-hard (for hardness ratio)
    ])

    output_dir: str = os.path.join(_BASE, "flare_forecast", "outputs")

    # ── Observation date ───────────────────────────────────────────────────
    obs_date: str = "2026-06-20"

    # ── Resampling ─────────────────────────────────────────────────────────
    resample_freq: str  = "1min"    # '1s' or '1min' — use 1min for practical LSTM
    # Number of resampled bins per day: 1440 for 1-min, 86400 for 1-sec

    # ── GOES-proxy class thresholds (multiple of rolling baseline) ─────────
    # Based on SoLEXS soft X-ray count-rate calibration (Jain et al. 2021 proxy)
    # class_thresholds[i] = lower bound for class i in units of baseline multiples
    class_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "quiet":  0.0,    # [0.0,  2.0) × baseline  → Class 0 (Quiet)
        "bc":     2.0,    # [2.0, 10.0) × baseline  → Class 1 (B/C)
        "m":     10.0,    # [10.0, 50.0) × baseline → Class 2 (M)
        "x":     50.0,    # ≥ 50.0 × baseline       → Class 3 (X)
    })
    class_names: List[str] = field(
        default_factory=lambda: ["Quiet Sun", "B/C Class", "M Class", "X Class"]
    )
    n_classes: int = 4

    # ── Baseline rolling window (minutes) ─────────────────────────────────
    baseline_win_min: int = 30   # 30-minute rolling 25th-pct background
    baseline_pct:     int = 25

    # ── Sequence / windowing ───────────────────────────────────────────────
    window_size:       int = 120   # 120 minutes (2 h) — lookback window
    prediction_horizon: int = 60  # 60 minutes ahead — label is for t+horizon

    # ── Feature engineering rolling windows (minutes) ─────────────────────
    roll_short_min: int = 5    # short-term rolling stats
    roll_long_min:  int = 15   # long-term rolling stats

    # ── Data split (chronological) ─────────────────────────────────────────
    train_frac: float = 0.70
    val_frac:   float = 0.15
    # test_frac = 1 - train_frac - val_frac = 0.15

    # ── Model & Loss ────────────────────────────────────────────────────────
    model_type:   str   = "lstm"    # "lstm" | "gru" | "transformer"
    loss_function: str  = "categorical_crossentropy"  # "categorical_crossentropy" | "focal_loss"
    focal_gamma:  float = 2.0       # Focusing parameter for focal loss
    focal_alpha:  float = 0.25      # Balancing parameter for focal loss
    
    lstm_units_1: int   = 128
    lstm_units_2: int   = 64
    dense_units:  int   = 64
    dropout_rate: float = 0.3
    use_attention: bool = True      # Bahdanau attention between LSTM layers

    # ── Training ───────────────────────────────────────────────────────────
    epochs:        int   = 100
    batch_size:    int   = 16       # Default batch size
    auto_tune_batch_size: bool = False
    candidate_batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    
    learning_rate: float = 3e-4     # Optimized learning rate
    clipnorm:      float = 1.0      # Gradient clipping threshold
    
    early_stop_patience: int   = 15
    lr_reduce_patience:  int   = 7
    lr_reduce_factor:    float = 0.5
    min_lr:              float = 1e-6
    
    checkpoint_monitor:  str   = "val_tss"  # "val_tss" | "val_f1_macro" | "val_loss"

    # ── Calibration ───────────────────────────────────────────────────────
    calibration_method: str = "temperature" # "temperature" | "isotonic" | "none"


    # ── Risk thresholds (P(M) + P(X)) ─────────────────────────────────────
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "LOW":      0.00,
        "MODERATE": 0.15,
        "HIGH":     0.40,
        "EXTREME":  0.70,
    })

    # ── Reproducibility ────────────────────────────────────────────────────
    random_seed: int = 42

    def __post_init__(self):
        """Create output directory if it does not exist."""
        os.makedirs(self.output_dir, exist_ok=True)


# Default singleton — import and use directly or override fields
CFG = Config()
