"""
feature_engineering.py
=======================
Computes all engineered features for each 1-minute time bin from the
synchronized SoLEXS + HEL1OS data.

Features produced per timestep  (22 total)
------------------------------------------
SoLEXS (10):
    slx_count, slx_peak_5m, slx_mean_5m, slx_std_5m,
    slx_mean_15m, slx_std_15m, slx_d1, slx_d2,
    slx_rise_flag, slx_bg_ratio

HEL1OS (7):
    hls_bb, hls_mean_5m, hls_std_5m, hls_hardness,
    hls_d1, hls_peak_5m, hls_bg_ratio

Temporal (5):
    hour_sin, hour_cos, prev_flare_count,
    minutes_since_flare, day_norm

Design
------
All features are computed in pure pandas/numpy — no scikit-learn here
(scaling is in preprocessing.py).  The function returns a DataFrame
whose column names become the canonical feature list throughout the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger

_LOG = get_logger("feature_engineering")

# Canonical ordered feature list — must match what build_features() returns
FEATURE_NAMES = [
    # SoLEXS
    "slx_count",
    "slx_peak_5m",
    "slx_mean_5m",
    "slx_std_5m",
    "slx_mean_15m",
    "slx_std_15m",
    "slx_d1",
    "slx_d2",
    "slx_rise_flag",
    "slx_bg_ratio",
    # HEL1OS
    "hls_bb",
    "hls_mean_5m",
    "hls_std_5m",
    "hls_hardness",
    "hls_d1",
    "hls_peak_5m",
    "hls_bg_ratio",
    # Temporal
    "hour_sin",
    "hour_cos",
    "prev_flare_count",
    "minutes_since_flare",
    "day_norm",
]

N_FEATURES = len(FEATURE_NAMES)   # 22


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute all engineered features from the labeled synchronized DataFrame.

    Parameters
    ----------
    df  : Output of dataset.assign_labels() — must have columns:
          'slx_count', 'hls_bb', 'hls_lo', 'hls_hi',
          'baseline', 'bg_ratio', 'label'
          Index: DatetimeIndex UTC (1-min).
    cfg : Config — uses roll_short_min, roll_long_min.

    Returns
    -------
    pd.DataFrame with columns = FEATURE_NAMES + ['label'].
    Rows with all-NaN features are dropped.
    """
    _LOG.info("Computing engineered features …")
    out = pd.DataFrame(index=df.index)

    ws = cfg.roll_short_min   # 5
    wl = cfg.roll_long_min    # 15

    # ── SoLEXS features ─────────────────────────────────────────────────────
    slx = df["slx_count"].fillna(0.0)   # treat gaps as 0

    out["slx_count"]   = slx
    out["slx_peak_5m"] = slx.rolling(ws, min_periods=1).max()
    out["slx_mean_5m"] = slx.rolling(ws, min_periods=1).mean()
    out["slx_std_5m"]  = slx.rolling(ws, min_periods=1).std().fillna(0.0)
    out["slx_mean_15m"]= slx.rolling(wl, min_periods=1).mean()
    out["slx_std_15m"] = slx.rolling(wl, min_periods=1).std().fillna(0.0)

    # Flux derivatives (finite differences, cts/s per minute)
    d1 = slx.diff().fillna(0.0)
    d2 = d1.diff().fillna(0.0)
    out["slx_d1"]        = d1
    out["slx_d2"]        = d2
    out["slx_rise_flag"] = (d1 > 0).astype(float)
    out["slx_bg_ratio"]  = df["bg_ratio"].fillna(0.0)

    # ── HEL1OS features ──────────────────────────────────────────────────────
    hls = df.get("hls_bb", pd.Series(0.0, index=df.index)).fillna(0.0)
    hls_lo = df.get("hls_lo", pd.Series(0.0, index=df.index)).fillna(0.0)
    hls_hi = df.get("hls_hi", pd.Series(0.0, index=df.index)).fillna(0.0)

    out["hls_bb"]      = hls
    out["hls_mean_5m"] = hls.rolling(ws, min_periods=1).mean()
    out["hls_std_5m"]  = hls.rolling(ws, min_periods=1).std().fillna(0.0)

    # Spectral hardness: (80-150 keV) / (20-40 keV)  — high ratio → harder spectrum
    denom = hls_lo.replace(0, np.nan)
    out["hls_hardness"] = (hls_hi / denom).fillna(0.0).clip(upper=50.0)

    hls_d1 = hls.diff().fillna(0.0)
    out["hls_d1"]      = hls_d1
    out["hls_peak_5m"] = hls.rolling(ws, min_periods=1).max()

    hls_baseline = (
        hls.rolling(cfg.baseline_win_min, min_periods=5, center=True)
           .quantile(cfg.baseline_pct / 100)
           .ffill().bfill().clip(lower=1.0)
    )
    out["hls_bg_ratio"] = (hls / hls_baseline).fillna(0.0)

    # ── Temporal features ────────────────────────────────────────────────────
    hours = df.index.hour + df.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Cumulative flare count (number of flare bins so far in the day)
    # A bin is a "flare" if its instantaneous class ≥ 1
    is_flare = (df["bg_ratio"].fillna(0) >= cfg.class_thresholds["bc"]).astype(int)
    out["prev_flare_count"] = is_flare.cumsum().shift(1).fillna(0.0)

    # Minutes since the last flare-class bin
    last_flare_min = _time_since_event(is_flare)
    out["minutes_since_flare"] = last_flare_min

    # Day of year normalized 0–1
    doy = df.index[0].day_of_year if len(df) > 0 else 1
    out["day_norm"] = float(doy) / 365.0

    # ── Attach label ─────────────────────────────────────────────────────────
    out["label"] = df["label"].values

    # ── Ensure all FEATURE_NAMES columns exist (fill missing with 0) ─────────
    for col in FEATURE_NAMES:
        if col not in out.columns:
            out[col] = 0.0

    # Reorder to canonical order
    out = out[FEATURE_NAMES + ["label"]]

    # Log feature stats
    _LOG.info(f"  Feature matrix shape: {out.shape}")
    nan_counts = out[FEATURE_NAMES].isna().sum()
    if nan_counts.any():
        _LOG.warning(f"  NaN counts per feature:\n{nan_counts[nan_counts > 0]}")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _time_since_event(binary_series: pd.Series) -> pd.Series:
    """
    For each timestep t, compute the number of bins since the last
    event (binary_series == 1).  Returns 0 for the first event and
    increments by 1 for each subsequent non-event bin.

    Example: [0,0,1,0,0,1,0] → [0,0,0,1,2,0,1]
    """
    result = np.zeros(len(binary_series), dtype=float)
    last_event = 0
    for i, val in enumerate(binary_series.values):
        if val == 1:
            result[i] = 0
            last_event = i
        else:
            result[i] = i - last_event
    return pd.Series(result, index=binary_series.index)
