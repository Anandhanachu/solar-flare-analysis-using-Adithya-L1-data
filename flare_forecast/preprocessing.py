"""
preprocessing.py
================
Data cleaning, normalization, sliding-window generation, chronological
splitting, and SMOTE oversampling for the solar flare LSTM pipeline.

Pipeline order
--------------
1. Impute missing values (forward fill → backward fill → zero fill)
2. Clip outliers (±5σ per feature)
3. Fit StandardScaler on training split only
4. Apply scaler to val and test
5. Generate sliding windows (X, y)
6. Optionally apply SMOTE on flattened training windows

Key design choices
------------------
- Never shuffle time-series data (chronological split only)
- Scaler fitted exclusively on training data to prevent data leakage
- SMOTE is applied only to (flattened) training windows
- Window targets use the label at the END of the window (t + window_size → t + window_size + horizon)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.preprocessing import StandardScaler

from .config import Config
from .feature_engineering import FEATURE_NAMES
from .utils import get_logger, save_pickle

_LOG = get_logger("preprocessing")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  –  Data Quality Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_data_quality(df: pd.DataFrame):
    """
    Check the feature DataFrame for data quality issues before processing.
    """
    _LOG.info("Running data quality checks...")
    
    # Check for NaN and Inf
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    nan_count = df[feature_cols].isna().sum().sum()
    inf_count = np.isinf(df[feature_cols].values).sum()
    
    if nan_count > 0:
        _LOG.warning(f"Found {nan_count} missing values in feature columns.")
    if inf_count > 0:
        _LOG.warning(f"Found {inf_count} infinite values in feature columns.")
        
    # Check for duplicate timestamps
    if df.index.duplicated().any():
        _LOG.warning("Found duplicate timestamps in the DatetimeIndex.")
        
    # Check timestamp continuity
    dt = df.index.to_series().diff().dt.total_seconds().dropna()
    if not (dt == dt.iloc[0]).all():
        _LOG.warning("Timestamp intervals are not perfectly continuous. Gaps exist.")

    _LOG.info("Data quality checks complete.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  Missing Value Imputation & Outlier Clipping
# ══════════════════════════════════════════════════════════════════════════════


def clean_features(feat_df: pd.DataFrame,
                   sigma_clip: float = 5.0) -> pd.DataFrame:
    """
    Impute missing values and clip outliers in the feature matrix.

    Steps
    -----
    1. Forward fill → backward fill → zero fill remaining NaN.
    2. Clip values beyond ±sigma_clip standard deviations from the mean.

    Parameters
    ----------
    feat_df    : DataFrame with columns = FEATURE_NAMES + ['label'].
    sigma_clip : Clip threshold in standard deviations (default 5.0).

    Returns
    -------
    Cleaned DataFrame (same shape, same columns).
    """
    df = feat_df.copy()
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

    # Impute
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)

    # Clip outliers
    for col in feature_cols:
        mu  = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(lower=mu - sigma_clip * std,
                                   upper=mu + sigma_clip * std)

    nan_remaining = df[feature_cols].isna().sum().sum()
    _LOG.info(f"Cleaning done. NaN remaining: {nan_remaining}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  Chronological Train / Val / Test Split
# ══════════════════════════════════════════════════════════════════════════════

def chronological_split(
    feat_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the feature DataFrame into train / val / test sets
    in chronological order.  NO random shuffling.

    Parameters
    ----------
    feat_df : Cleaned feature DataFrame.
    cfg     : Config with train_frac, val_frac.

    Returns
    -------
    Tuple of (train_df, val_df, test_df).
    """
    n      = len(feat_df)
    n_trn  = int(n * cfg.train_frac)
    n_val  = int(n * cfg.val_frac)

    train_df = feat_df.iloc[:n_trn]
    val_df   = feat_df.iloc[n_trn : n_trn + n_val]
    test_df  = feat_df.iloc[n_trn + n_val :]

    _LOG.info(f"Split  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Validation: Check class distribution
    for name, df_split in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
        if "label" in df_split.columns:
            dist = df_split["label"].value_counts().sort_index().to_dict()
            dist_str = ", ".join([f"C{k}: {v}" for k, v in dist.items()])
            _LOG.info(f"  {name} class distribution: {dist_str}")
            if len(dist) < cfg.n_classes:
                missing = [c for c in range(cfg.n_classes) if c not in dist]
                _LOG.warning(f"  {name} split is missing classes: {missing}!")
                _LOG.warning(f"  Consider adjusting the split fraction or using rolling-origin validation.")

    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3  –  Normalization
# ══════════════════════════════════════════════════════════════════════════════

def fit_scaler(train_df: pd.DataFrame,
               cfg: Config) -> StandardScaler:
    """
    Fit a StandardScaler on the training feature columns only.

    Saves the fitted scaler to cfg.output_dir/scaler.pkl for
    inference-time reuse.

    Parameters
    ----------
    train_df : Training split DataFrame.
    cfg      : Config.

    Returns
    -------
    Fitted StandardScaler.
    """
    feature_cols = [c for c in FEATURE_NAMES if c in train_df.columns]
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    scaler_path = os.path.join(cfg.output_dir, "scaler.pkl")
    save_pickle(scaler, scaler_path)
    _LOG.info(f"Scaler fitted and saved → {scaler_path}")
    return scaler


def apply_scaler(df: pd.DataFrame,
                 scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply a pre-fitted StandardScaler to the feature columns of *df*.

    The 'label' column is not transformed.

    Parameters
    ----------
    df     : Feature DataFrame (any split).
    scaler : Fitted StandardScaler from fit_scaler().

    Returns
    -------
    Scaled DataFrame (same index and column names).
    """
    df = df.copy()
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4  –  Sliding Window Generation
# ══════════════════════════════════════════════════════════════════════════════

def make_sequences(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate overlapping sliding windows from a scaled feature DataFrame.

    Each window:
        X[i] = features for bins [i, i + window_size)   shape: (window_size, n_features)
        y[i] = label at bin i + window_size - 1          scalar int

    The label at the end of the window already encodes the maximum class
    within the next prediction_horizon bins (set in dataset.assign_labels).

    Parameters
    ----------
    df  : Scaled feature DataFrame with 'label' column.
    cfg : Config with window_size.

    Returns
    -------
    X : np.ndarray  shape (N, window_size, n_features)
    y : np.ndarray  shape (N,)  dtype int
    """
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X_arr        = df[feature_cols].values.astype(np.float32)
    y_arr        = df["label"].values.astype(np.int32)

    win   = cfg.window_size
    N     = len(X_arr) - win + 1

    if N <= 0:
        raise ValueError(
            f"Not enough data for window_size={win}. "
            f"Got {len(X_arr)} rows."
        )

    X = np.stack([X_arr[i : i + win] for i in range(N)],  axis=0)
    y = np.array([y_arr[i + win - 1]  for i in range(N)], dtype=np.int32)

    _LOG.info(f"  Sequences: X={X.shape}  y={y.shape}  "
              f"class dist={np.bincount(y, minlength=cfg.n_classes).tolist()}")
              
    # Window Validation: Print first window shape and target to verify alignment
    if len(X) > 0:
        _LOG.info(f"  Sample Window [0]: length={len(X[0])}, shape={X[0].shape}, label={y[0]}")
              
    return X, y


# (SMOTE functions completely removed per refactoring requirements)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6  –  High-Level Preprocessing Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    feat_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[
    np.ndarray, np.ndarray,   # X_train, y_train  (after SMOTE)
    np.ndarray, np.ndarray,   # X_val,   y_val
    np.ndarray, np.ndarray,   # X_test,  y_test
    StandardScaler,            # fitted scaler
]:
    """
    Full preprocessing pipeline:
    clean → split → scale → window → (SMOTE on train).

    Parameters
    ----------
    feat_df : Feature DataFrame from feature_engineering.build_features().
    cfg     : Config.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    # 0. Data Quality Check
    verify_data_quality(feat_df)
    
    # 1. Clean
    feat_df = clean_features(feat_df)

    # 2. Chronological split
    train_df, val_df, test_df = chronological_split(feat_df, cfg)

    # 3. Fit scaler on train, apply everywhere
    scaler   = fit_scaler(train_df, cfg)
    train_sc = apply_scaler(train_df, scaler)
    val_sc   = apply_scaler(val_df,   scaler)
    test_sc  = apply_scaler(test_df,  scaler)

    # 4. Sliding windows
    _LOG.info("Generating training sequences …")
    X_train, y_train = make_sequences(train_sc, cfg)
    _LOG.info("Generating validation sequences …")
    X_val,   y_val   = make_sequences(val_sc,   cfg)
    _LOG.info("Generating test sequences …")
    X_test,  y_test  = make_sequences(test_sc,  cfg)

    _LOG.info(
        f"Final shapes — "
        f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
