"""
=============================================================================
  ADITYA-L1 SOLAR FLARE EXPLORATORY DATA ANALYSIS (EDA)
  =============================================================================
  Instruments : SoLEXS SDD2 (Soft X-ray, 1-15 keV)
                HEL1OS CZT1  (Hard X-ray, 18-160 keV)
  Date        : 2026-06-20 (UTC)
  Author      : Aditya-L1 Analysis Pipeline
  =============================================================================

  Pipeline Sections
  -----------------
  0.  Imports & Configuration
  1.  Load SoLEXS SDD2 Light Curve (gzip FITS)
  2.  Load HEL1OS CZT1 18-160 keV (2 observations, regular FITS)
  3.  Timestamp Conversion -> UTC
  4.  Data Cleaning (NaN, negatives, duplicates)
  5.  Synchronize on Common 1-Second UTC Timeline
  6.  Descriptive Statistics
  7.  Smoothing (Savitzky-Golay)
  8.  Individual Light Curve Plots
  9.  Combined Dual-Axis Plot
  10. Statistics Plots (histograms, rolling mean/std)
  11. Flare Detection (automatic peak finding, baseline subtraction)
  12. Flare Annotation Plot
  13. Cross-Correlation & Soft-Hard Time Lag
  14. Flare Event Catalog -> CSV
  15. Zoomed Flare Profile Plots
  16. HEL1OS Multi-Band Light Curves
  17. Summary Report
"""

# ============================================================
# SECTION 0 -- IMPORTS & CONFIGURATION
# ============================================================
import sys
import os
import gzip
import shutil
import tempfile
import warnings

# Force UTF-8 output on Windows (avoids cp1252 encode errors)
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend -- renders to file
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.io import fits
from astropy.time import Time
from scipy.signal import find_peaks, savgol_filter, correlate, correlation_lags
from scipy.stats import pearsonr

# ── Paths ─────────────────────────────────────────────────────────────────
BASE    = r'c:\Users\anand\ISRO,BHA'
OUTDIR  = os.path.join(BASE, 'eda_output')
os.makedirs(OUTDIR, exist_ok=True)

# ── Publication-quality plot style ────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'        : 150,
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 11,
    'axes.labelsize'    : 12,
    'axes.titlesize'    : 13,
    'axes.titleweight'  : 'bold',
    'legend.fontsize'   : 10,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : ':',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : '#f8f9fa',
})

# Colour palette
C_SOFT   = '#E8722A'   # SoLEXS -- orange
C_SOFT2  = '#C0430A'   # darker orange for smoothed line
C_HARD   = '#3A6EBF'   # HEL1OS -- blue
C_HARD2  = '#1A3E8F'   # darker blue for smoothed line
C_PEAK   = '#E63946'   # peak markers
C_BASE   = '#6C757D'   # baseline

print("=" * 70)
print("  ADITYA-L1 Solar Flare EDA")
print("  SoLEXS SDD2 (Soft)  +  HEL1OS CZT1 18-160 keV (Hard)")
print("  Observation date: 2026-06-20")
print("=" * 70)


# ============================================================
# SECTION 1 -- LOAD SoLEXS SDD2 LIGHT CURVE
# ============================================================
# File: AL1_SOLEXS_20260620_SDD2_L1.lc.gz  (gzip-compressed FITS)
# HDU layout:
#   HDU #0  PRIMARY  -- metadata, no data
#   HDU #1  RATE     -- BinTable: TIME (float64, Unix seconds), COUNTS (float64)
#
# Timing reference:
#   MJDREFI = 40587  ->  MJD 40587 = 1970-01-01 00:00:00 UTC (Unix epoch)
#   MJDREFF = 0
#   ∴ TIME column is in UNIX seconds (seconds since 1970-01-01 00:00:00 UTC)

print("\n[1] Loading SoLEXS SDD2 light curve ...")

SLX_GZ = os.path.join(
    BASE,
    r'SLX\AL1_SLX_L1_20260620_v1.0\SDD2\AL1_SOLEXS_20260620_SDD2_L1.lc.gz'
)

# Decompress to a temporary FITS file so astropy can open it normally
_tmp = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
with gzip.open(SLX_GZ, 'rb') as _fi:
    shutil.copyfileobj(_fi, _tmp)
_tmp.close()

with fits.open(_tmp.name) as hdul:
    hdu_rate  = hdul['RATE']
    # .astype(float) converts FITS big-endian arrays to native little-endian
    # required because pandas does not support big-endian buffers
    time_raw  = hdu_rate.data['TIME'].astype(float)    # Unix seconds
    count_raw = hdu_rate.data['COUNTS'].astype(float)  # Broadband counts/sec
    # Keep header metadata for reference
    mjdrefi   = hdu_rate.header.get('MJDREFI', 40587)
    timedel   = hdu_rate.header.get('TIMEDEL', 1)    # seconds per bin
    date_obs  = hdu_rate.header.get('DATE-OBS', 'unknown')
    date_end  = hdu_rate.header.get('DATE-END', 'unknown')

os.unlink(_tmp.name)   # Clean up temp file

print(f"    Header DATE-OBS  : {date_obs}")
print(f"    Header DATE-END  : {date_end}")
print(f"    Time bin size    : {timedel} second(s)")
print(f"    Raw rows loaded  : {len(time_raw):,}")


# ============================================================
# SECTION 2 -- LOAD HEL1OS CZT1 LIGHT CURVE (18-160 keV)
# ============================================================
# HEL1OS data splits the day into two orbital observation blocks.
# Each file has 6 HDUs:
#   HDU #0  PRIMARY                             -- metadata
#   HDU #1  CZT1_LC_BAND_20.00KEV_TO_40.00KEV  -- 20-40 keV band
#   HDU #2  CZT1_LC_BAND_40.00KEV_TO_60.00KEV  -- 40-60 keV band
#   HDU #3  CZT1_LC_BAND_60.00KEV_TO_80.00KEV  -- 60-80 keV band
#   HDU #4  CZT1_LC_BAND_80.00KEV_TO_150.00KEV -- 80-150 keV band
#   HDU #5  CZT1_LC_BAND_18.00KEV_TO_160.00KEV -- Full broadband (THIS ONE)
#
# Columns (all HDUs): MJD (float64), ISOT (30-char string),
#                     CTR (cts/sec), STAT_ERR (cts/sec = sqrt(CTR))

print("\n[2] Loading HEL1OS CZT1 18-160 keV (two observation blocks) ...")

HLS_BASE  = os.path.join(BASE, r'HLS\2026\06\20')
HLS_BAND  = 'CZT1_LC_BAND_18.00KEV_TO_160.00KEV'

HLS_FILES = [
    os.path.join(
        HLS_BASE,
        r'HLS_20260620_000008_43177sec_lev1_V111\czt\lightcurve_czt1.fits'
    ),
    os.path.join(
        HLS_BASE,
        r'HLS_20260620_121027_42563sec_lev1_V111\czt\lightcurve_czt1.fits'
    ),
]

hls_parts = []
for obs_num, fpath in enumerate(HLS_FILES, start=1):
    with fits.open(fpath) as hdul:
        hdu      = hdul[HLS_BAND]
        # .astype(float) forces native byte order (FITS is big-endian)
        mjd_arr  = hdu.data['MJD'].astype(float)       # Modified Julian Date
        ctr_arr  = hdu.data['CTR'].astype(float)       # Count rate (cts/sec)
        err_arr  = hdu.data['STAT_ERR'].astype(float)  # Statistical error

    # Convert MJD -> Unix seconds via astropy
    # Time(..., format='mjd', scale='utc') handles leap seconds correctly
    astro_t = Time(mjd_arr, format='mjd', scale='utc')
    unix_s  = astro_t.unix     # float64 array of Unix timestamps

    df_part = pd.DataFrame({
        'unix_s'   : unix_s,
        'ctr_hls'  : ctr_arr,
        'stat_err' : err_arr,
        'obs_block': obs_num
    })
    hls_parts.append(df_part)
    print(f"    OBS-{obs_num}: {len(df_part):,} rows  "
          f"MJD {mjd_arr.min():.5f} -> {mjd_arr.max():.5f}")

hls_raw = pd.concat(hls_parts, ignore_index=True)
print(f"    Total rows concatenated: {len(hls_raw):,}")


# ============================================================
# SECTION 3 -- TIMESTAMP CONVERSION -> UTC
# ============================================================
# SoLEXS: TIME is Unix seconds -> pandas.to_datetime(unit='s', utc=True)
# HEL1OS: already converted to unix_s via astropy above

print("\n[3] Converting timestamps to UTC pandas DatetimeTZDtype ...")

# SoLEXS timestamps
slx_utc = pd.to_datetime(time_raw, unit='s', utc=True, errors='coerce')
slx_df  = pd.DataFrame({'utc': slx_utc, 'counts': count_raw})

# HEL1OS timestamps
hls_utc = pd.to_datetime(hls_raw['unix_s'], unit='s', utc=True, errors='coerce')
hls_df  = hls_raw.copy()
hls_df['utc'] = hls_utc

print(f"    SoLEXS range : {slx_df['utc'].min()}  ->  {slx_df['utc'].max()}")
print(f"    HEL1OS range : {hls_df['utc'].min()}  ->  {hls_df['utc'].max()}")


# ============================================================
# SECTION 4 -- DATA CLEANING
# ============================================================
# Remove:
#   • Rows where timestamp conversion failed (NaT)
#   • NaN count/rate values
#   • Negative count values (detector artefacts)
#   • Duplicate timestamps (take mean if any exist)

print("\n[4] Cleaning data ...")

# -- SoLEXS --
n_before = len(slx_df)
slx_df = slx_df.dropna(subset=['utc', 'counts'])
slx_df = slx_df[slx_df['counts'] >= 0]   # remove unphysical negatives
slx_df = slx_df.sort_values('utc').reset_index(drop=True)
print(f"    SoLEXS: {n_before:,} -> {len(slx_df):,} rows  "
      f"(removed {n_before - len(slx_df):,} bad rows)")

# -- HEL1OS --
n_before = len(hls_df)
hls_df = hls_df.dropna(subset=['utc', 'ctr_hls'])
hls_df = hls_df[hls_df['ctr_hls'] >= 0]
hls_df = hls_df.sort_values('utc').reset_index(drop=True)
print(f"    HEL1OS: {n_before:,} -> {len(hls_df):,} rows  "
      f"(removed {n_before - len(hls_df):,} bad rows)")

# Note on HEL1OS zero-count rows:
# HEL1OS CZT has a detector duty cycle: it integrates for ~1 sec then
# has a readout dead-time gap.  Zero-count seconds are REAL DETECTOR GAPS,
# NOT missing data.  We keep them as zeros.
zero_pct = (hls_df['ctr_hls'] == 0).mean() * 100
print(f"    HEL1OS zero-count rows: {zero_pct:.1f}%  (detector duty cycle -- kept)")


# ============================================================
# SECTION 5 -- SYNCHRONIZE ON COMMON 1-SECOND UTC TIMELINE
# ============================================================
# Create a uniform 1-second grid for the full day (86 400 bins).
# Both instruments are re-indexed on this grid.
# Bins with no data are filled with NaN -- honest representation of gaps.

print("\n[5] Synchronizing on common 1-second UTC timeline ...")

DAY_START  = pd.Timestamp('2026-06-20 00:00:00', tz='UTC')
DAY_END    = pd.Timestamp('2026-06-20 23:59:59', tz='UTC')
COMMON_IDX = pd.date_range(start=DAY_START, end=DAY_END, freq='1s')

print(f"    Grid: {DAY_START}  ->  {DAY_END}  ({len(COMMON_IDX):,} bins)")

def resample_to_grid(df, time_col, value_col, grid):
    """Floor timestamps to second, group-mean duplicates, reindex to grid."""
    df = df.copy()
    df['_ts'] = df[time_col].dt.floor('s')
    return df.groupby('_ts')[value_col].mean().reindex(grid)

slx_series = resample_to_grid(slx_df, 'utc', 'counts',  COMMON_IDX)
hls_series = resample_to_grid(hls_df, 'utc', 'ctr_hls', COMMON_IDX)

# Master synchronized DataFrame
sync_df = pd.DataFrame({
    'utc' : COMMON_IDX,
    'slx' : slx_series.values,   # SoLEXS soft X-ray counts/sec
    'hls' : hls_series.values,   # HEL1OS hard X-ray counts/sec
}, index=COMMON_IDX)

cov_slx  = sync_df['slx'].notna().sum()
cov_hls  = sync_df['hls'].notna().sum()
cov_both = (sync_df['slx'].notna() & sync_df['hls'].notna()).sum()

print(f"    SoLEXS coverage  : {cov_slx:,} / {len(COMMON_IDX):,} bins "
      f"({100*cov_slx/len(COMMON_IDX):.1f}%)")
print(f"    HEL1OS coverage  : {cov_hls:,} / {len(COMMON_IDX):,} bins "
      f"({100*cov_hls/len(COMMON_IDX):.1f}%)")
print(f"    Both valid       : {cov_both:,} bins "
      f"({100*cov_both/len(COMMON_IDX):.1f}%)")


# ============================================================
# SECTION 6 -- DESCRIPTIVE STATISTICS
# ============================================================

print("\n[6] Descriptive Statistics")
print("=" * 65)

STATS_CONFIG = [
    ('SoLEXS SDD2  (cts/s, broadband soft X-ray)', 'slx'),
    ('HEL1OS CZT1  (cts/s, 18-160 keV hard X-ray)', 'hls'),
]

stats_records = []
for label, col in STATS_CONFIG:
    data = sync_df[col].dropna()
    q25, q50, q75, q99 = data.quantile([0.25, 0.50, 0.75, 0.99])
    print(f"\n  {label}")
    print(f"    N        : {len(data):,}")
    print(f"    Mean     : {data.mean():.3f}")
    print(f"    Median   : {q50:.3f}")
    print(f"    Std Dev  : {data.std():.3f}")
    print(f"    Min      : {data.min():.3f}")
    print(f"    Max      : {data.max():.3f}")
    print(f"    P25      : {q25:.3f}")
    print(f"    P75      : {q75:.3f}")
    print(f"    P99      : {q99:.3f}")
    print(f"    Skewness : {data.skew():.3f}")
    print(f"    Kurtosis : {data.kurtosis():.3f}")
    stats_records.append({
        'instrument': label, 'N': len(data),
        'mean': round(data.mean(), 3), 'median': round(q50, 3),
        'std': round(data.std(), 3), 'min': round(data.min(), 3),
        'max': round(data.max(), 3), 'p25': round(q25, 3),
        'p75': round(q75, 3), 'p99': round(q99, 3),
        'skew': round(data.skew(), 3), 'kurtosis': round(data.kurtosis(), 3)
    })

stats_df = pd.DataFrame(stats_records)
stats_df.to_csv(os.path.join(OUTDIR, 'descriptive_stats.csv'), index=False)
print(f"\n  Saved: descriptive_stats.csv")


# ============================================================
# SECTION 7 -- SMOOTHING (SAVITZKY-GOLAY FILTER)
# ============================================================
# Savitzky-Golay polynomial smoothing preserves peak shapes and widths
# better than a simple moving average, making it ideal for flare detection.
# Window = 61 sec (~1 min),  Polynomial order = 3.
# We apply the filter only to contiguous valid (non-NaN) segments.

SMOOTH_WIN  = 61   # seconds (must be odd)
SMOOTH_POLY = 3

def savgol_gapped(arr, window=SMOOTH_WIN, poly=SMOOTH_POLY):
    """
    Apply SavGol filter segment-by-segment across NaN gaps.
    Each contiguous valid run is smoothed independently.
    """
    arr    = np.array(arr, dtype=float)
    result = np.full_like(arr, np.nan)
    valid  = np.isfinite(arr)

    # Find contiguous valid segments
    dv   = np.diff(valid.astype(int), prepend=0, append=0)
    starts = np.where(dv == 1)[0]
    ends   = np.where(dv == -1)[0]

    for s, e in zip(starts, ends):
        seg = arr[s:e]
        if len(seg) >= window:
            result[s:e] = savgol_filter(seg, window_length=window, polyorder=poly)
        else:
            result[s:e] = seg   # Too short to filter -- keep raw
    return result

slx_smooth = savgol_gapped(sync_df['slx'].values)
hls_smooth = savgol_gapped(sync_df['hls'].values)

sync_df['slx_smooth'] = slx_smooth
sync_df['hls_smooth'] = hls_smooth

print(f"\n[7] Savitzky-Golay smoothing applied  (window={SMOOTH_WIN}s, poly={SMOOTH_POLY})")


# ============================================================
# SECTION 8 -- INDIVIDUAL LIGHT CURVE PLOTS
# ============================================================

print("\n[8] Plotting individual light curves ...")

fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                         gridspec_kw={'hspace': 0.08})
fig.suptitle('Aditya-L1 Light Curves -- 2026-06-20 (UTC)',
             fontsize=15, fontweight='bold', y=1.01)

# Panel 1 -- SoLEXS SDD2
ax = axes[0]
ax.fill_between(sync_df['utc'], sync_df['slx'], alpha=0.25, color=C_SOFT,
                label='Raw (1s bins)')
ax.plot(sync_df['utc'], sync_df['slx_smooth'], color=C_SOFT2,
        linewidth=0.9, label='Smoothed (SavGol 61s)', alpha=0.9)
ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
ax.set_title('SoLEXS SDD2 -- Soft X-ray  (broadband, ~1-15 keV)', fontsize=12)
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlim(DAY_START, DAY_END)

# Panel 2 -- HEL1OS CZT1
ax = axes[1]
ax.fill_between(sync_df['utc'], sync_df['hls'], alpha=0.20, color=C_HARD,
                label='Raw (1s bins)')
ax.plot(sync_df['utc'], sync_df['hls_smooth'], color=C_HARD2,
        linewidth=0.9, label='Smoothed (SavGol 61s)', alpha=0.9)
ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
ax.set_xlabel('Time (UTC)', fontsize=11)
ax.set_title('HEL1OS CZT1 -- Hard X-ray  (18-160 keV)', fontsize=12)
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

plt.tight_layout()
out_indiv = os.path.join(OUTDIR, '01_individual_lightcurves.png')
fig.savefig(out_indiv, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 01_individual_lightcurves.png")


# ============================================================
# SECTION 9 -- COMBINED DUAL-AXIS OVERLAY PLOT
# ============================================================
# Dual y-axis allows comparing signals on very different scales
# (SoLEXS ~0-900 cts/s  vs  HEL1OS ~0-500 cts/s).

print("[9] Combined dual-axis overlay ...")

fig, ax1 = plt.subplots(figsize=(18, 6))
fig.suptitle(
    'Aditya-L1  --  SoLEXS (Soft) & HEL1OS (Hard) X-ray  |  2026-06-20',
    fontsize=14, fontweight='bold'
)

ax1.set_xlabel('Time (UTC)', fontsize=12)
ax1.set_ylabel('SoLEXS Count Rate (cts/s)', color=C_SOFT, fontsize=12)
ln1, = ax1.plot(sync_df['utc'], sync_df['slx_smooth'], color=C_SOFT,
                linewidth=1.2, label='SoLEXS SDD2 (Soft, 1-15 keV)')
ax1.fill_between(sync_df['utc'], sync_df['slx_smooth'],
                 alpha=0.10, color=C_SOFT)
ax1.tick_params(axis='y', labelcolor=C_SOFT)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax1.set_xlim(DAY_START, DAY_END)

ax2 = ax1.twinx()
ax2.set_ylabel('HEL1OS Count Rate (cts/s)', color=C_HARD, fontsize=12)
ln2, = ax2.plot(sync_df['utc'], sync_df['hls_smooth'], color=C_HARD,
                linewidth=1.2, label='HEL1OS CZT1 (Hard, 18-160 keV)',
                alpha=0.85)
ax2.fill_between(sync_df['utc'], sync_df['hls_smooth'],
                 alpha=0.08, color=C_HARD)
ax2.tick_params(axis='y', labelcolor=C_HARD)

# Unified legend
ax1.legend(handles=[ln1, ln2], loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
out_comb = os.path.join(OUTDIR, '02_combined_dual_axis.png')
fig.savefig(out_comb, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 02_combined_dual_axis.png")


# ============================================================
# SECTION 10 -- STATISTICS PLOTS
# ============================================================

print("[10] Statistics plots (histograms & rolling statistics) ...")

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle('Aditya-L1 -- Light Curve Statistics  |  2026-06-20',
             fontsize=14, fontweight='bold')

slx_valid = sync_df['slx'].dropna()
hls_valid = sync_df['hls'].dropna()

# Top-left: SoLEXS count distribution
ax = axes[0, 0]
ax.hist(slx_valid, bins=120, color=C_SOFT, alpha=0.75,
        edgecolor='white', linewidth=0.4)
ax.axvline(slx_valid.mean(),   color='darkred', ls='--', lw=1.8,
           label=f'Mean = {slx_valid.mean():.1f}')
ax.axvline(slx_valid.median(), color='orange',  ls=':',  lw=1.8,
           label=f'Median = {slx_valid.median():.1f}')
ax.set_title('SoLEXS SDD2 -- Count Distribution')
ax.set_xlabel('Count Rate (cts/s)')
ax.set_ylabel('Frequency (bins)')
ax.legend()

# Top-right: HEL1OS count distribution
ax = axes[0, 1]
ax.hist(hls_valid, bins=120, color=C_HARD, alpha=0.75,
        edgecolor='white', linewidth=0.4)
ax.axvline(hls_valid.mean(),   color='darkblue', ls='--', lw=1.8,
           label=f'Mean = {hls_valid.mean():.1f}')
ax.axvline(hls_valid.median(), color='skyblue',  ls=':',  lw=1.8,
           label=f'Median = {hls_valid.median():.1f}')
ax.set_title('HEL1OS CZT1 18-160 keV -- Rate Distribution')
ax.set_xlabel('Count Rate (cts/s)')
ax.set_ylabel('Frequency (bins)')
ax.legend()

# Bottom-left: SoLEXS 5-minute rolling mean +/- 1sigma
ROLL_WIN = 300   # 5-minute window
ax = axes[1, 0]
slx_rm  = sync_df['slx'].rolling(ROLL_WIN, min_periods=30, center=True).mean()
slx_rs  = sync_df['slx'].rolling(ROLL_WIN, min_periods=30, center=True).std()
ax.plot(sync_df['utc'], slx_rm, color=C_SOFT2, lw=1.4, label='5-min rolling mean')
ax.fill_between(sync_df['utc'], slx_rm - slx_rs, slx_rm + slx_rs,
                alpha=0.30, color=C_SOFT, label='+/-1sigma')
ax.set_title('SoLEXS -- 5-min Rolling Mean +/- sigma')
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Count Rate (cts/s)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))

# Bottom-right: HEL1OS 5-minute rolling mean +/- 1sigma
ax = axes[1, 1]
hls_rm  = sync_df['hls'].rolling(ROLL_WIN, min_periods=30, center=True).mean()
hls_rs  = sync_df['hls'].rolling(ROLL_WIN, min_periods=30, center=True).std()
ax.plot(sync_df['utc'], hls_rm, color=C_HARD2, lw=1.4, label='5-min rolling mean')
ax.fill_between(sync_df['utc'], hls_rm - hls_rs, hls_rm + hls_rs,
                alpha=0.30, color=C_HARD, label='+/-1sigma')
ax.set_title('HEL1OS CZT1 -- 5-min Rolling Mean +/- sigma')
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Count Rate (cts/s)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))

plt.tight_layout()
out_stats = os.path.join(OUTDIR, '03_statistics.png')
fig.savefig(out_stats, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 03_statistics.png")


# ============================================================
# SECTION 11 -- FLARE DETECTION (AUTOMATIC PEAK FINDING)
# ============================================================
# Algorithm:
# ─────────────────────────────────────────────────────────────
#  1. Estimate BASELINE using a wide rolling low-percentile
#     (25th pct over 10-minute window) to track quiescent emission.
#  2. Compute BACKGROUND-SUBTRACTED signal = raw - baseline.
#  3. Estimate NOISE as rolling std of the background-subtracted signal.
#  4. THRESHOLD = N x noise  (N-sigma above baseline).
#  5. scipy.signal.find_peaks locates candidate peaks above threshold.
#  6. For each peak:
#       • Walk BACKWARDS until signal drops below threshold  -> FLARE START
#       • Walk FORWARDS  until signal drops below threshold  -> FLARE END
#       • Require minimum duration (30 s) to reject noise spikes.
#  7. Compute: rise_time, decay_time, duration, peak_excess, SNR,
#              asymmetry (decay/rise), GOES-class estimate.
# ─────────────────────────────────────────────────────────────

print("\n[11] Automatic flare detection ...")

BASELINE_WIN   = 600   # seconds -- rolling background window
BASELINE_PCT   = 25    # percentile for background estimate
NOISE_WIN      = 600   # seconds -- rolling noise window
SIGMA_THRESH   = 3.5   # detection threshold (N-sigma above background)
MIN_PROMINENCE = 15    # minimum count-rate prominence above background
MIN_DISTANCE   = 180   # minimum seconds between two separate flares
MIN_DURATION   = 30    # minimum flare duration in seconds
MAX_SEARCH     = 3600  # maximum seconds to look for start/end from peak


def compute_background(arr, window=BASELINE_WIN, pct=BASELINE_PCT):
    """
    Estimate the quiescent background using a rolling low-percentile.
    Low percentile (25th) ensures bright flares don't bias the background.
    """
    s = pd.Series(arr)
    bg = s.rolling(window, min_periods=max(30, window//10), center=True)\
          .quantile(pct / 100)
    return bg.ffill().bfill().values


def detect_flares(arr, timestamps, label='',
                  baseline_win=BASELINE_WIN, baseline_pct=BASELINE_PCT,
                  noise_win=NOISE_WIN, sigma=SIGMA_THRESH,
                  min_prom=MIN_PROMINENCE, min_dist=MIN_DISTANCE,
                  min_dur=MIN_DURATION, max_search=MAX_SEARCH):
    """
    Full pipeline: background subtraction -> peak finding -> flare boundaries.

    Parameters
    ----------
    arr        : 1-D float array of count rates (NaN allowed)
    timestamps : pandas DatetimeIndex aligned with arr
    Returns
    -------
    peak_indices : ndarray of peak positions in arr
    flares       : list of dicts with all flare properties
    baseline     : background array (for plotting)
    noise        : noise array (for plotting)
    """
    arr      = np.array(arr, dtype=float)
    valid    = np.isfinite(arr)

    # 1. Background
    baseline = compute_background(arr, window=baseline_win, pct=baseline_pct)

    # 2. Background-subtracted signal
    bkg_sub  = arr - baseline
    bkg_sub[~valid] = 0.0   # treat gaps as zero for peak finding

    # 3. Noise level
    noise = (pd.Series(bkg_sub)
              .rolling(noise_win, min_periods=30, center=True)
              .std()
              .ffill().bfill()
              .values)
    noise = np.where(noise <= 0, 1e-3, noise)   # avoid division by zero

    threshold = sigma * noise

    # 4. Find peaks
    peak_idx, peak_props = find_peaks(
        bkg_sub,
        height      = threshold,    # must exceed N-sigma
        prominence  = min_prom,     # must stand out by this much
        distance    = min_dist,     # minimum seconds between peaks
        width       = 5             # minimum peak width (5 sec)
    )

    print(f"  {label}: {len(peak_idx)} candidate peaks found")

    flares = []
    for pk in peak_idx:
        if not valid[pk]:
            continue

        peak_val  = arr[pk]
        bg_val    = baseline[pk]
        noise_val = noise[pk]

        # 5a. Walk backwards to find flare start
        start_i = pk
        for j in range(pk, max(0, pk - max_search), -1):
            if not valid[j]:
                start_i = j + 1
                break
            if bkg_sub[j] <= threshold[j]:
                start_i = j
                break

        # 5b. Walk forwards to find flare end
        end_i = pk
        for j in range(pk, min(len(arr) - 1, pk + max_search)):
            if not valid[j]:
                end_i = j - 1
                break
            if bkg_sub[j] <= threshold[j]:
                end_i = j
                break

        duration_s  = int(end_i  - start_i)
        rise_s      = int(pk     - start_i)
        decay_s     = int(end_i  - pk)
        peak_excess = float(peak_val - bg_val)

        # Reject very short spikes
        if duration_s < min_dur:
            continue

        flares.append({
            'start_idx'  : start_i,
            'peak_idx'   : pk,
            'end_idx'    : end_i,
            'start_utc'  : timestamps[start_i],
            'peak_utc'   : timestamps[pk],
            'end_utc'    : timestamps[end_i],
            'duration_s' : duration_s,
            'rise_s'     : rise_s,
            'decay_s'    : decay_s,
            'peak_ctr'   : float(peak_val),
            'baseline'   : float(bg_val),
            'peak_excess': peak_excess,
            'snr'        : float(peak_excess / noise_val),
            'asymmetry'  : float(decay_s / max(rise_s, 1)),
            # Asymmetry > 1 -> gradual decay (classic thermal flare)
            # Asymmetry < 1 -> rapid decay (impulsive hard X-ray burst)
        })

    print(f"  {label}: {len(flares)} confirmed flares (>={min_dur}s duration)")
    return np.array([f['peak_idx'] for f in flares]), flares, baseline, noise

# Run detection on smoothed light curves
slx_peak_idx, slx_flares, slx_bg, slx_noise = detect_flares(
    slx_smooth, COMMON_IDX, label='SoLEXS SDD2',
    sigma=3.5, min_prom=20, min_dist=180, min_dur=30
)
hls_peak_idx, hls_flares, hls_bg, hls_noise = detect_flares(
    hls_smooth, COMMON_IDX, label='HEL1OS CZT1',
    sigma=3.5, min_prom=25, min_dist=180, min_dur=30
)

print(f"\n  Confirmed SoLEXS flares : {len(slx_flares)}")
print(f"  Confirmed HEL1OS flares : {len(hls_flares)}")


# ============================================================
# SECTION 12 -- FLARE DETECTION ANNOTATION PLOT
# ============================================================

print("[12] Flare detection annotation plot ...")

def annotate_flares_on_ax(ax, flares, arr, color_span, color_peak):
    """Shade flare spans and mark peaks + start/end on an axis."""
    for k, f in enumerate(flares):
        t_start = pd.Timestamp(f['start_utc'])
        t_peak  = pd.Timestamp(f['peak_utc'])
        t_end   = pd.Timestamp(f['end_utc'])
        ax.axvspan(t_start, t_end,   alpha=0.15, color=color_span, zorder=0)
        ax.axvline(t_peak, color=color_peak, lw=0.9, ls='--', alpha=0.8)
        ax.axvline(t_start, color='green', lw=0.6, ls=':', alpha=0.6)
        ax.axvline(t_end,   color='blue',  lw=0.6, ls=':', alpha=0.6)
        # Label the peak
        ax.annotate(
            f"F{k+1}",
            xy=(t_peak, f['peak_ctr']),
            xytext=(2, 6), textcoords='offset points',
            fontsize=7.5, color=color_peak, fontweight='bold', ha='left'
        )

fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True,
                         gridspec_kw={'hspace': 0.06})
fig.suptitle('Aditya-L1 -- Automatic Flare Detection  |  2026-06-20',
             fontsize=14, fontweight='bold', y=1.01)

# SoLEXS panel
ax = axes[0]
ax.fill_between(sync_df['utc'], sync_df['slx'], alpha=0.15, color=C_SOFT)
ax.plot(sync_df['utc'], slx_smooth, color=C_SOFT2, lw=0.8,
        alpha=0.75, label='Smoothed')
ax.plot(sync_df['utc'], slx_bg, color=C_BASE, lw=1.0, ls='--',
        alpha=0.8, label='Background (25th pct, 10-min window)')
# 3-sigma envelope
ax.fill_between(sync_df['utc'],
                slx_bg + SIGMA_THRESH * slx_noise,
                slx_bg, alpha=0.07, color=C_SOFT,
                label=f'{SIGMA_THRESH}sigma threshold')
if slx_peak_idx.size > 0:
    ax.scatter(sync_df['utc'].iloc[slx_peak_idx], slx_smooth[slx_peak_idx],
               color=C_PEAK, s=40, zorder=6, label=f'Peaks ({len(slx_peak_idx)})')
annotate_flares_on_ax(ax, slx_flares, slx_smooth, C_SOFT, '#CC2200')
ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
ax.set_title('SoLEXS SDD2 -- Soft X-ray  |  Green dashes = flare start, Blue = end', fontsize=11)
ax.legend(loc='upper right', fontsize=8.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# HEL1OS panel
ax = axes[1]
ax.fill_between(sync_df['utc'], sync_df['hls'], alpha=0.12, color=C_HARD)
ax.plot(sync_df['utc'], hls_smooth, color=C_HARD2, lw=0.8,
        alpha=0.75, label='Smoothed')
ax.plot(sync_df['utc'], hls_bg, color=C_BASE, lw=1.0, ls='--',
        alpha=0.8, label='Background')
ax.fill_between(sync_df['utc'],
                hls_bg + SIGMA_THRESH * hls_noise,
                hls_bg, alpha=0.07, color=C_HARD,
                label=f'{SIGMA_THRESH}sigma threshold')
if hls_peak_idx.size > 0:
    ax.scatter(sync_df['utc'].iloc[hls_peak_idx], hls_smooth[hls_peak_idx],
               color='#0055CC', s=40, zorder=6, label=f'Peaks ({len(hls_peak_idx)})')
annotate_flares_on_ax(ax, hls_flares, hls_smooth, C_HARD, '#0022CC')
ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
ax.set_xlabel('Time (UTC)', fontsize=11)
ax.set_title('HEL1OS CZT1 -- Hard X-ray 18-160 keV', fontsize=11)
ax.legend(loc='upper right', fontsize=8.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.set_xlim(DAY_START, DAY_END)

plt.tight_layout()
out_det = os.path.join(OUTDIR, '04_flare_detection.png')
fig.savefig(out_det, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 04_flare_detection.png")


# ============================================================
# SECTION 13 -- CROSS-CORRELATION & SOFT-HARD TIME LAG
# ============================================================
# The Neupert effect predicts:
#   Hard X-rays (non-thermal electrons) arrive FIRST (impulsive phase)
#   Soft X-rays (thermal plasma) peak LATER (gradual phase)
# -> Expected soft-hard lag: positive (soft LEADS in time-reversed sense)
# -> In standard cross-correlation: negative lag = hard leads soft
#
# We compute the full normalized cross-correlation and find the peak lag.
# Also report the Pearson correlation at zero lag.

print("\n[13] Cross-correlation and soft-hard time lag ...")

# Only use bins where BOTH instruments have valid data
mask = np.isfinite(slx_smooth) & np.isfinite(hls_smooth)
slx_cc = slx_smooth[mask]
hls_cc = hls_smooth[mask]

n_cc = len(slx_cc)
print(f"    Samples for cross-correlation: {n_cc:,}")

# Zero-mean, unit-variance normalization
slx_norm = (slx_cc - np.mean(slx_cc)) / (np.std(slx_cc) + 1e-12)
hls_norm = (hls_cc - np.mean(hls_cc)) / (np.std(hls_cc) + 1e-12)

# Full cross-correlation
xcorr = correlate(slx_norm, hls_norm, mode='full')
lags  = correlation_lags(n_cc, n_cc, mode='full')
xcorr = xcorr / n_cc   # normalize by sample count

# Restrict to +/-30 min (1800 sec)
MAX_LAG_S   = 1800
lag_mask    = np.abs(lags) <= MAX_LAG_S
xcorr_sub   = xcorr[lag_mask]
lags_sub    = lags[lag_mask]

# Peak lag
best_i      = np.argmax(xcorr_sub)
best_lag    = int(lags_sub[best_i])
best_xcorr  = float(xcorr_sub[best_i])

# Pearson at zero lag
r_pearson, p_pearson = pearsonr(slx_norm, hls_norm)

print(f"    Best lag (soft - hard)  : {best_lag:+d} seconds")
print(f"    Max cross-correlation   : {best_xcorr:.4f}")
print(f"    Pearson r (zero lag)    : {r_pearson:.4f}   p = {p_pearson:.3e}")

if best_lag > 0:
    interp = (f"Soft X-rays (SoLEXS) LEAD hard X-rays (HEL1OS) by {best_lag}s "
              f"-> Neupert-like behaviour (hard impulsive first, soft peaks later)")
elif best_lag < 0:
    interp = (f"Hard X-rays (HEL1OS) LEAD soft X-rays (SoLEXS) by {abs(best_lag)}s "
              f"-> Hard emission precedes thermal response")
else:
    interp = "No measurable time lag between soft and hard channels"

print(f"    Interpretation          : {interp}")

# Cross-correlation plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Soft-Hard X-ray Cross-Correlation  |  Aditya-L1  2026-06-20',
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(lags_sub, xcorr_sub, color='#5A3E8F', lw=1.0)
ax.axvline(best_lag, color=C_PEAK, lw=1.8, ls='--',
           label=f'Peak lag = {best_lag:+d} s')
ax.axvline(0, color=C_BASE, lw=0.8, ls=':')
ax.axhline(0, color='k', lw=0.4)
ax.set_xlabel('Lag (seconds)  [positive -> soft leads hard]')
ax.set_ylabel('Normalized Cross-Correlation')
ax.set_title(f'Cross-Correlation  (+/-{MAX_LAG_S//60} min)  |  r₀ = {r_pearson:.3f}')
ax.legend()

ax = axes[1]
# Down-sample scatter for speed
step = max(1, n_cc // 5000)
ax.scatter(slx_norm[::step], hls_norm[::step],
           alpha=0.15, s=3, color='#5A3E8F', label='1-sec bins')
m_fit = np.polyfit(slx_norm, hls_norm, 1)
xline = np.linspace(slx_norm.min(), slx_norm.max(), 200)
ax.plot(xline, np.polyval(m_fit, xline), color=C_PEAK, lw=1.8,
        label=f'OLS  r={r_pearson:.3f}  p={p_pearson:.2e}')
ax.set_xlabel('SoLEXS (Soft X-ray, normalised)')
ax.set_ylabel('HEL1OS (Hard X-ray, normalised)')
ax.set_title('Soft vs Hard Scatter (normalised)')
ax.legend(fontsize=9)

plt.tight_layout()
out_xcorr = os.path.join(OUTDIR, '05_cross_correlation.png')
fig.savefig(out_xcorr, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 05_cross_correlation.png")


# ============================================================
# SECTION 14 -- FLARE EVENT CATALOG -> CSV
# ============================================================
# Build two per-instrument catalogs and one matched soft-hard pair catalog.
# Columns exported per flare:
#   flare_id, instrument, band,
#   start_utc, peak_utc, end_utc,
#   duration_s, rise_time_s, decay_time_s,
#   peak_ctr_cts_s, baseline_ctr_cts_s, peak_excess_cts_s,
#   snr, asymmetry (decay/rise ratio)

print("\n[14] Building flare catalogs ...")

def make_catalog(flares, instrument, band):
    rows = []
    for k, f in enumerate(flares):
        rows.append({
            'flare_id'              : f"{instrument}_F{k+1:02d}",
            'instrument'            : instrument,
            'band'                  : band,
            'start_utc'             : str(f['start_utc'])[:19],
            'peak_utc'              : str(f['peak_utc'])[:19],
            'end_utc'               : str(f['end_utc'])[:19],
            'duration_s'            : f['duration_s'],
            'rise_time_s'           : f['rise_s'],
            'decay_time_s'          : f['decay_s'],
            'peak_ctr_cts_s'        : round(f['peak_ctr'], 3),
            'baseline_ctr_cts_s'    : round(f['baseline'], 3),
            'peak_excess_cts_s'     : round(f['peak_excess'], 3),
            'snr'                   : round(f['snr'], 2),
            'asymmetry_decay_rise'  : round(f['asymmetry'], 3),
            # Asymmetry note:
            #   > 1 -> gradual decay  (typical thermal soft-X flare)
            #   < 1 -> rapid decay    (impulsive hard-X burst)
        })
    return pd.DataFrame(rows)

slx_cat = make_catalog(slx_flares, 'SoLEXS_SDD2', 'Soft_~1-15keV')
hls_cat = make_catalog(hls_flares, 'HEL1OS_CZT1', 'Hard_18-160keV')

# Matched soft-hard pairs (within +/-15 min of each other)
matched_pairs = []
MATCH_WINDOW = pd.Timedelta(minutes=15)

for sf in slx_flares:
    st_pk = pd.Timestamp(sf['peak_utc'])
    best_hf, best_dt = None, MATCH_WINDOW
    for hf in hls_flares:
        ht_pk = pd.Timestamp(hf['peak_utc'])
        dt    = abs(st_pk - ht_pk)
        if dt < best_dt:
            best_dt = dt
            best_hf = hf
    if best_hf is not None:
        ht_pk     = pd.Timestamp(best_hf['peak_utc'])
        lag_s     = (st_pk - ht_pk).total_seconds()
        matched_pairs.append({
            'slx_peak_utc'         : str(st_pk)[:19],
            'hls_peak_utc'         : str(ht_pk)[:19],
            'soft_minus_hard_lag_s': round(lag_s, 1),
            # positive -> soft peaks AFTER hard (Neupert-consistent)
            'slx_peak_ctr'         : round(sf['peak_ctr'], 2),
            'hls_peak_ctr'         : round(best_hf['peak_ctr'], 2),
            'slx_duration_s'       : sf['duration_s'],
            'hls_duration_s'       : best_hf['duration_s'],
            'slx_rise_s'           : sf['rise_s'],
            'hls_rise_s'           : best_hf['rise_s'],
            'slx_snr'              : round(sf['snr'], 2),
            'hls_snr'              : round(best_hf['snr'], 2),
        })

lag_cat = pd.DataFrame(matched_pairs)

# Save to CSV
slx_cat.to_csv(os.path.join(OUTDIR, 'solexs_flare_catalog.csv'), index=False)
hls_cat.to_csv(os.path.join(OUTDIR, 'helios_flare_catalog.csv'), index=False)
if not lag_cat.empty:
    lag_cat.to_csv(os.path.join(OUTDIR, 'soft_hard_lag_catalog.csv'), index=False)

print(f"    SoLEXS catalog    : {len(slx_cat)} events -> solexs_flare_catalog.csv")
print(f"    HEL1OS catalog    : {len(hls_cat)} events -> helios_flare_catalog.csv")
print(f"    Matched pairs     : {len(lag_cat)}          -> soft_hard_lag_catalog.csv")

print("\n  --- SoLEXS Flare Catalog ---")
if not slx_cat.empty:
    print(slx_cat.to_string(index=False))
else:
    print("  (no confirmed flares above threshold)")

print("\n  --- HEL1OS Flare Catalog ---")
if not hls_cat.empty:
    print(hls_cat.to_string(index=False))
else:
    print("  (no confirmed flares above threshold)")

if not lag_cat.empty:
    print("\n  --- Soft-Hard Peak Lag ---")
    print(lag_cat.to_string(index=False))


# ============================================================
# SECTION 15 -- ZOOMED FLARE PROFILE PLOTS
# ============================================================
# For each confirmed flare, show +/-5 min around the event with
# start, peak, end markers and a Gaussian background envelope.

print("\n[15] Zoomed flare profile plots ...")

PAD_S = 300   # +/-5 minutes padding around each flare

def plot_flare_profiles(flares, smooth_arr, raw_series, color_fill,
                        color_line, instrument_label, filename):
    if not flares:
        print(f"    No flares to plot for {instrument_label}")
        return
    n = min(len(flares), 8)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle(f'{instrument_label} -- Individual Flare Profiles  |  2026-06-20',
                 fontsize=13, fontweight='bold', y=1.01)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= n:
            ax.set_visible(False)
            continue
        f   = flares[idx]
        i0  = max(0, f['start_idx'] - PAD_S)
        i1  = min(len(sync_df) - 1, f['end_idx'] + PAD_S)
        utc_win = sync_df['utc'].iloc[i0:i1]
        raw_win = raw_series.iloc[i0:i1]
        sm_win  = smooth_arr[i0:i1]

        ax.fill_between(utc_win, raw_win, alpha=0.20, color=color_fill)
        ax.plot(utc_win, sm_win, color=color_line, lw=1.3)
        ax.axvline(pd.Timestamp(f['start_utc']), color='#2DC653',
                   lw=1.2, ls='--', label='Start')
        ax.axvline(pd.Timestamp(f['peak_utc']),  color=C_PEAK,
                   lw=1.4, ls='-',  label='Peak')
        ax.axvline(pd.Timestamp(f['end_utc']),   color='#3E8FCC',
                   lw=1.2, ls='--', label='End')

        pk_ts = pd.Timestamp(f['peak_utc'])
        ax.set_title(
            f"Flare F{idx+1} | Peak: {pk_ts.strftime('%H:%M:%S')}\n"
            f"Dur={f['duration_s']}s  Rise={f['rise_s']}s  "
            f"Decay={f['decay_s']}s  SNR={f['snr']:.1f}",
            fontsize=8
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelsize=8, rotation=25)
        ax.set_ylabel('cts/s', fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    out = os.path.join(OUTDIR, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {filename}")

plot_flare_profiles(slx_flares, slx_smooth, sync_df['slx'],
                    C_SOFT, C_SOFT2, 'SoLEXS SDD2',
                    '06_slx_flare_profiles.png')
plot_flare_profiles(hls_flares, hls_smooth, sync_df['hls'],
                    C_HARD, C_HARD2, 'HEL1OS CZT1 (18-160 keV)',
                    '07_hls_flare_profiles.png')


# ============================================================
# SECTION 16 -- HEL1OS MULTI-BAND LIGHT CURVES
# ============================================================
# Load all 5 CZT1 energy bands from both observations and plot them
# as stacked panels for energy-resolved flare spectral analysis.

print("[16] HEL1OS multi-band light curves ...")

BANDS = [
    ('CZT1_LC_BAND_20.00KEV_TO_40.00KEV',  '20-40 keV',   '#440154'),
    ('CZT1_LC_BAND_40.00KEV_TO_60.00KEV',  '40-60 keV',   '#3B528B'),
    ('CZT1_LC_BAND_60.00KEV_TO_80.00KEV',  '60-80 keV',   '#21918C'),
    ('CZT1_LC_BAND_80.00KEV_TO_150.00KEV', '80-150 keV',  '#5EC962'),
    ('CZT1_LC_BAND_18.00KEV_TO_160.00KEV', '18-160 keV\n(Broadband)', '#FDE725'),
]

fig, axes = plt.subplots(len(BANDS), 1, figsize=(18, 14), sharex=True,
                         gridspec_kw={'hspace': 0.05})
fig.suptitle('HEL1OS CZT1 -- All Energy Bands  |  2026-06-20 (UTC)',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (band_key, band_label, color) in zip(axes, BANDS):
    band_parts = []
    for fpath in HLS_FILES:
        with fits.open(fpath) as hdul:
            names = [h.name for h in hdul]
            if band_key not in names:
                continue
            hdu_b    = hdul[band_key]
            mjd_b    = hdu_b.data['MJD'].copy()
            ctr_b    = hdu_b.data['CTR'].copy()
        astro_b = Time(mjd_b, format='mjd', scale='utc')
        unix_b  = astro_b.unix
        utc_b   = pd.to_datetime(unix_b, unit='s', utc=True)
        df_b    = pd.DataFrame({'ts': utc_b.floor('s'), 'ctr': ctr_b})
        band_parts.append(df_b)

    band_all = pd.concat(band_parts).dropna()
    band_all = band_all[band_all['ctr'] >= 0]
    band_res = band_all.groupby('ts')['ctr'].mean().reindex(COMMON_IDX)
    band_sm  = savgol_gapped(band_res.values)

    ax.fill_between(COMMON_IDX, band_res.values, alpha=0.20, color=color)
    ax.plot(COMMON_IDX, band_sm, color=color, lw=0.9, label=band_label)
    ax.set_ylabel('cts/s', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlim(DAY_START, DAY_END)

axes[-1].set_xlabel('Time (UTC)', fontsize=11)
axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))

plt.tight_layout()
out_mb = os.path.join(OUTDIR, '08_hls_multiband.png')
fig.savefig(out_mb, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 08_hls_multiband.png")


# ============================================================
# SECTION 17 -- SUMMARY REPORT
# ============================================================

print("\n" + "=" * 70)
print("  EDA COMPLETE  --  Output Summary")
print("=" * 70)
print(f"\n  Output directory  :  {OUTDIR}")
print("""
  Plots:
    01_individual_lightcurves.png   Individual SoLEXS & HEL1OS panels
    02_combined_dual_axis.png       Dual-y-axis overlay (soft + hard)
    03_statistics.png               Histograms & 5-min rolling statistics
    04_flare_detection.png          Auto-detected flares with annotations
    05_cross_correlation.png        Soft-hard XCorr & scatter plot
    06_slx_flare_profiles.png       Zoomed SoLEXS flare profiles
    07_hls_flare_profiles.png       Zoomed HEL1OS flare profiles
    08_hls_multiband.png            All 5 CZT1 energy bands stacked

  CSVs:
    descriptive_stats.csv           Per-instrument descriptive statistics
    solexs_flare_catalog.csv        SoLEXS detected flare events + features
    helios_flare_catalog.csv        HEL1OS detected flare events + features
    soft_hard_lag_catalog.csv       Matched soft-hard flare pairs + lags
""")
print(f"  Cross-correlation result:")
print(f"    Best soft-hard peak lag   : {best_lag:+d} seconds")
print(f"    {interp}")
print(f"    Pearson r (0-lag)         : {r_pearson:.4f}")
print(f"\n  SoLEXS flares detected    : {len(slx_flares)}")
print(f"  HEL1OS flares detected    : {len(hls_flares)}")
print(f"  Matched soft-hard pairs   : {len(lag_cat)}")
print("=" * 70)
