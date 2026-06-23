"""
dataset.py
==========
Raw data loading for the Aditya-L1 Solar Flare Forecasting pipeline.

Responsibilities
----------------
- Load SoLEXS SDD2 light curve from gzip-compressed FITS file.
- Load HEL1OS CZT1 light curves (multiple energy bands) from two
  orbital observation blocks.
- Build a synchronized DataFrame resampled to a configurable frequency
  (default: 1-minute bins).
- Assign GOES-proxy flare class labels based on configurable thresholds.

All timing is done in UTC.  No global state — every function takes a
Config object and returns plain DataFrames.
"""

from __future__ import annotations

import os
import gzip
import shutil
import tempfile
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from .config import Config
from .utils import get_logger

warnings.filterwarnings("ignore")

_LOG = get_logger("dataset")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  –  SoLEXS SDD2 Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_solexs(cfg: Config) -> pd.DataFrame:
    """
    Load the SoLEXS SDD2 broadband light curve from a gzip-compressed FITS.

    The FITS HDU 'RATE' contains:
        TIME   (float64) – Unix seconds  (MJDREFI=40587 → 1970-01-01 UTC)
        COUNTS (float64) – Count rate in cts/s

    Parameters
    ----------
    cfg : Config object containing the path to SoLEXS FITS (.lc.gz).

    Returns
    -------
    pd.DataFrame with columns ['utc', 'slx_count'] sorted by time,
    NaN-free (NaT / negative counts removed).
    """
    _LOG.info(f"Loading SoLEXS: {os.path.basename(cfg.solexs_gz)}")

    # Decompress to a temporary FITS file — astropy cannot read gzip directly
    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    with gzip.open(cfg.solexs_gz, "rb") as fi:
        shutil.copyfileobj(fi, tmp)
    tmp.close()

    try:
        with fits.open(tmp.name) as hdul:
            hdu        = hdul["RATE"]
            time_unix  = hdu.data["TIME"].astype(float)
            counts     = hdu.data["COUNTS"].astype(float)
            date_obs   = hdu.header.get("DATE-OBS", "unknown")
            date_end   = hdu.header.get("DATE-END", "unknown")
    finally:
        os.unlink(tmp.name)

    _LOG.info(f"  DATE-OBS={date_obs}  DATE-END={date_end}  rows={len(time_unix):,}")

    utc = pd.to_datetime(time_unix, unit="s", utc=True, errors="coerce")
    df  = pd.DataFrame({"utc": utc, "slx_count": counts})
    df  = df.dropna(subset=["utc", "slx_count"])
    df  = df[df["slx_count"] >= 0]
    df  = df.sort_values("utc").reset_index(drop=True)

    _LOG.info(f"  SoLEXS clean rows: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  HEL1OS CZT1 Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_helios(cfg: Config) -> pd.DataFrame:
    """
    Load HEL1OS CZT1 light curves for multiple energy bands from two
    orbital observation blocks.

    HDU layout (all bands):
        MJD      (float64) – Modified Julian Date
        CTR      (float64) – Count rate (cts/s)
        STAT_ERR (float64) – Statistical error = sqrt(CTR)

    Parameters
    ----------
    cfg : Config object with helios_files and helios_bands lists.

    Returns
    -------
    pd.DataFrame with columns:
        'utc'      – UTC timestamp
        'hls_bb'   – broadband (18-160 keV) count rate
        'hls_lo'   – low-hard (20-40 keV) count rate
        'hls_hi'   – high-hard (80-150 keV) count rate
    Rows with negative counts or NaT timestamps are removed.
    """
    _LOG.info("Loading HEL1OS CZT1 (2 observation blocks) …")

    band_cols = {
        cfg.helios_bands[0]: "hls_bb",   # 18-160 keV broadband
        cfg.helios_bands[1]: "hls_lo",   # 20-40 keV
        cfg.helios_bands[2]: "hls_hi",   # 80-150 keV
    }

    # Load each file × band combination, keyed by (file_idx, band)
    all_parts: List[pd.DataFrame] = []

    for obs_idx, fpath in enumerate(cfg.helios_files, 1):
        if not os.path.exists(fpath):
            _LOG.warning(f"  OBS-{obs_idx}: file not found → {fpath}")
            continue

        with fits.open(fpath) as hdul:
            available_names = [h.name for h in hdul]
            obs_frames: dict = {}

            for band_key, col_name in band_cols.items():
                if band_key not in available_names:
                    _LOG.warning(f"  Band '{band_key}' not in OBS-{obs_idx}")
                    continue
                hdu     = hdul[band_key]
                mjd_arr = hdu.data["MJD"].astype(float)
                ctr_arr = hdu.data["CTR"].astype(float)
                unix_s  = Time(mjd_arr, format="mjd", scale="utc").unix
                obs_frames[col_name] = pd.Series(ctr_arr, name=col_name,
                                                  index=unix_s)

        if not obs_frames:
            continue

        # Align all bands to the broadband index (they share the same time axis)
        base_col = "hls_bb" if "hls_bb" in obs_frames else list(obs_frames.keys())[0]
        df_obs   = pd.DataFrame(obs_frames).reset_index()
        df_obs.rename(columns={"index": "unix_s"}, inplace=True)
        all_parts.append(df_obs)
        _LOG.info(f"  OBS-{obs_idx}: {len(df_obs):,} rows loaded")

    if not all_parts:
        raise FileNotFoundError("No HEL1OS files could be loaded.")

    hls_raw = pd.concat(all_parts, ignore_index=True)
    utc     = pd.to_datetime(hls_raw["unix_s"], unit="s", utc=True, errors="coerce")
    hls_raw["utc"] = utc

    # Keep only non-negative, non-NaT rows
    valid_cols = [c for c in ["hls_bb", "hls_lo", "hls_hi"] if c in hls_raw.columns]
    hls_raw    = hls_raw.dropna(subset=["utc"] + valid_cols)
    for col in valid_cols:
        hls_raw = hls_raw[hls_raw[col] >= 0]

    hls_raw = hls_raw.sort_values("utc").reset_index(drop=True)
    _LOG.info(f"  HEL1OS total clean rows: {len(hls_raw):,}")

    zero_pct = (hls_raw["hls_bb"] == 0).mean() * 100 if "hls_bb" in hls_raw.columns else 0
    _LOG.info(f"  HEL1OS broadband zero-count fraction: {zero_pct:.1f}% (duty cycle gaps)")

    return hls_raw[["utc"] + valid_cols]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3  –  Synchronized Grid Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_sync_grid(slx_df: pd.DataFrame,
                    hls_df: pd.DataFrame,
                    cfg: Config) -> pd.DataFrame:
    """
    Resample both instruments onto a common uniform UTC grid.

    Grid frequency is cfg.resample_freq (default '1min').
    Bins with no data are filled with NaN — honest data gap representation.
    HEL1OS zero-count bins (detector gaps) are kept as 0.0.

    Parameters
    ----------
    slx_df : SoLEXS DataFrame from load_solexs().
    hls_df : HEL1OS DataFrame from load_helios().
    cfg    : Config object.

    Returns
    -------
    pd.DataFrame with a DatetimeIndex (UTC) and columns:
        'slx_count', 'hls_bb', 'hls_lo', 'hls_hi'
    """
    day_start = pd.Timestamp(f"{cfg.obs_date} 00:00:00", tz="UTC")
    day_end   = pd.Timestamp(f"{cfg.obs_date} 23:59:00", tz="UTC")
    grid      = pd.date_range(start=day_start, end=day_end, freq=cfg.resample_freq)

    _LOG.info(f"Building {cfg.resample_freq} grid: {day_start} → {day_end} "
              f"({len(grid):,} bins)")

    def _resample(df: pd.DataFrame, time_col: str, value_cols: List[str]) -> pd.DataFrame:
        """Floor timestamps to grid resolution, mean-aggregate, reindex."""
        df = df.copy()
        df["_ts"] = df[time_col].dt.floor(cfg.resample_freq)
        agg = df.groupby("_ts")[value_cols].mean()
        return agg.reindex(grid)

    slx_grid = _resample(slx_df, "utc", ["slx_count"])
    hls_cols  = [c for c in ["hls_bb", "hls_lo", "hls_hi"] if c in hls_df.columns]
    hls_grid  = _resample(hls_df, "utc", hls_cols)

    sync = pd.concat([slx_grid, hls_grid], axis=1)
    sync.index.name = "utc"

    slx_cov = sync["slx_count"].notna().mean() * 100
    _LOG.info(f"  SoLEXS coverage : {slx_cov:.1f}%")
    if "hls_bb" in sync.columns:
        hls_cov = sync["hls_bb"].notna().mean() * 100
        _LOG.info(f"  HEL1OS coverage : {hls_cov:.1f}%")

    return sync


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4  –  GOES-Proxy Label Assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_labels(sync_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Assign a GOES-proxy flare class label (0–3) to each time bin based on
    the SoLEXS count rate relative to a rolling baseline.

    Label rules (configurable via cfg.class_thresholds):
        0 – Quiet Sun : count < bc_thresh × baseline
        1 – B/C Class : bc_thresh ≤ count < m_thresh × baseline
        2 – M Class   : m_thresh  ≤ count < x_thresh × baseline
        3 – X Class   : count ≥ x_thresh × baseline

    The label at time t represents the highest class seen within
    cfg.prediction_horizon bins AFTER t (i.e., the forecast target).

    Parameters
    ----------
    sync_df : Synchronized grid DataFrame.
    cfg     : Config with class thresholds and baseline settings.

    Returns
    -------
    sync_df copy with additional columns:
        'baseline'    – rolling background estimate
        'bg_ratio'    – slx_count / baseline
        'label'       – integer class 0–3 (forecast target)
    """
    df = sync_df.copy()

    # ── Rolling baseline (low-percentile of SoLEXS) ─────────────────────────
    win_bins = cfg.baseline_win_min   # already in minutes if resample_freq='1min'
    baseline = (
        df["slx_count"]
          .rolling(win_bins, min_periods=max(5, win_bins // 5), center=True)
          .quantile(cfg.baseline_pct / 100)
          .ffill()
          .bfill()
    )
    # Clamp baseline to a minimum of 1.0 cts/s to avoid division by zero
    baseline = baseline.clip(lower=1.0)
    df["baseline"] = baseline
    df["bg_ratio"] = df["slx_count"] / baseline

    # ── Instantaneous class per bin ─────────────────────────────────────────
    thr = cfg.class_thresholds
    def _classify(ratio: float) -> int:
        if ratio >= thr["x"]:
            return 3
        if ratio >= thr["m"]:
            return 2
        if ratio >= thr["bc"]:
            return 1
        return 0

    inst_class = df["bg_ratio"].apply(
        lambda r: _classify(r) if pd.notna(r) else 0
    ).astype(int)

    # ── Forecast label: max class in [t, t + horizon] ─────────────────────
    # Rolling max of inst_class shifted backwards in time (future window)
    horizon = cfg.prediction_horizon
    # shift(-horizon) looks at values horizon bins ahead
    future_max = (
        pd.Series(inst_class.values, index=df.index)
          .shift(-horizon)                          # align so t sees t+horizon
          .rolling(horizon, min_periods=1).max()   # max over the horizon
    )
    # For the last `horizon` bins we fall back to instantaneous class
    labels = future_max.fillna(inst_class).astype(int)
    df["label"] = labels.values

    counts = np.bincount(df["label"].values, minlength=cfg.n_classes)
    for i, (name, cnt) in enumerate(zip(cfg.class_names, counts)):
        _LOG.info(f"  Class {i} ({name:12s}): {cnt:5d} bins  ({100*cnt/len(df):.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5  –  High-Level Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(cfg: Config) -> pd.DataFrame:
    """
    Full data loading pipeline: load → sync → label.

    Returns a DataFrame ready for feature engineering, with columns:
        'slx_count', 'hls_bb', 'hls_lo', 'hls_hi',
        'baseline', 'bg_ratio', 'label'
    Index: DatetimeIndex (UTC, 1-min frequency).
    """
    slx_df  = load_solexs(cfg)
    hls_df  = load_helios(cfg)
    sync_df = build_sync_grid(slx_df, hls_df, cfg)
    labeled = assign_labels(sync_df, cfg)

    _LOG.info(f"Dataset ready: {len(labeled):,} bins × "
              f"{labeled.shape[1]} columns")
    return labeled
