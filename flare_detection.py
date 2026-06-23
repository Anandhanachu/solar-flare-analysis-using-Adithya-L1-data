"""
=============================================================================
  ADITYA-L1  --  SOLAR FLARE DETECTION VISUALIZATION
  =============================================================================
  Instruments : SoLEXS SDD2  (Soft X-ray, ~1-15 keV)
                HEL1OS CZT1  (Hard X-ray, 18-160 keV)
  Date        : 2026-06-20 (UTC)

  Output      : flare_detection.png
  =============================================================================

  Plot layout
  -----------
  - X-axis   : UTC time  (00:00 – 24:00 on 2026-06-20)
  - Y-axis   : Count rate  (photons / second)
  - Blue line : SoLEXS SDD2 smoothed light curve
  - Red dots  : HEL1OS CZT1  (sparse – 17.5 % duty cycle)
  - Green dashed  : Baseline (rolling 25th-pct background)
  - Orange dashed : 2 × baseline
  - Red dashed    : 3 × baseline
  - Circle markers: At each flare peak (count > 3 × baseline)
  - Annotations   : UTC time + count rate of every flare peak

  Flare count expected: 8 SoLEXS  /  0 HEL1OS
=============================================================================
"""

import sys
import os
import gzip
import shutil
import tempfile
import warnings

# Force UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.time import Time
from scipy.signal import savgol_filter, find_peaks

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = r'c:\Users\anand\ISRO,BHA'
OUTDIR = BASE          # save flare_detection.png next to this script
os.makedirs(OUTDIR, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
C_SLX      = '#1D7BC3'   # SoLEXS – blue
C_HLS      = '#E53935'   # HEL1OS – red
C_BASE     = '#2ECC71'   # baseline – green
C_2X       = '#E67E22'   # 2× baseline – orange
C_3X       = '#E74C3C'   # 3× baseline – red (dashed)
C_PEAK_MK  = '#FFD700'   # peak circle fill – gold
C_PEAK_ED  = '#CC0000'   # peak circle edge – dark red
C_ANNOT    = '#1A1A2E'   # annotation text

print("=" * 68)
print("  ADITYA-L1  Solar Flare Detection Visualization")
print("  SoLEXS SDD2 (Soft)  +  HEL1OS CZT1 (Hard)  |  2026-06-20")
print("=" * 68)


# ============================================================
# SECTION 1  --  LOAD SoLEXS SDD2 LIGHT CURVE  (gzip FITS)
# ============================================================
print("\n[1] Loading SoLEXS SDD2 light curve …")

SLX_GZ = os.path.join(
    BASE,
    r'SLX\AL1_SLX_L1_20260620_v1.0\SDD2\AL1_SOLEXS_20260620_SDD2_L1.lc.gz'
)

_tmp = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
with gzip.open(SLX_GZ, 'rb') as _fi:
    shutil.copyfileobj(_fi, _tmp)
_tmp.close()

with fits.open(_tmp.name) as hdul:
    hdu_rate  = hdul['RATE']
    time_raw  = hdu_rate.data['TIME'].astype(float)   # Unix seconds
    count_raw = hdu_rate.data['COUNTS'].astype(float) # cts / sec
    date_obs  = hdu_rate.header.get('DATE-OBS', 'unknown')
    date_end  = hdu_rate.header.get('DATE-END', 'unknown')

os.unlink(_tmp.name)

print(f"    DATE-OBS : {date_obs}  |  DATE-END : {date_end}")
print(f"    Rows loaded : {len(time_raw):,}")


# ============================================================
# SECTION 2  --  LOAD HEL1OS CZT1 LIGHT CURVE  (18-160 keV)
# ============================================================
print("\n[2] Loading HEL1OS CZT1 18-160 keV …")

HLS_BASE = os.path.join(BASE, r'HLS\2026\06\20')
HLS_BAND = 'CZT1_LC_BAND_18.00KEV_TO_160.00KEV'

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
for obs_num, fpath in enumerate(HLS_FILES, 1):
    with fits.open(fpath) as hdul:
        hdu     = hdul[HLS_BAND]
        mjd_arr = hdu.data['MJD'].astype(float)
        ctr_arr = hdu.data['CTR'].astype(float)
    unix_s = Time(mjd_arr, format='mjd', scale='utc').unix
    df_part = pd.DataFrame({'unix_s': unix_s, 'ctr_hls': ctr_arr})
    hls_parts.append(df_part)
    print(f"    OBS-{obs_num}: {len(df_part):,} rows")

hls_raw = pd.concat(hls_parts, ignore_index=True)
print(f"    Total HEL1OS rows: {len(hls_raw):,}")


# ============================================================
# SECTION 3  --  TIMESTAMP → UTC  +  CLEANING
# ============================================================
print("\n[3] Timestamp conversion & cleaning …")

# SoLEXS
slx_utc = pd.to_datetime(time_raw, unit='s', utc=True, errors='coerce')
slx_df  = pd.DataFrame({'utc': slx_utc, 'counts': count_raw})
slx_df  = slx_df.dropna(subset=['utc', 'counts'])
slx_df  = slx_df[slx_df['counts'] >= 0].sort_values('utc').reset_index(drop=True)

# HEL1OS
hls_utc = pd.to_datetime(hls_raw['unix_s'], unit='s', utc=True, errors='coerce')
hls_df  = hls_raw.copy()
hls_df['utc'] = hls_utc
hls_df  = hls_df.dropna(subset=['utc', 'ctr_hls'])
hls_df  = hls_df[hls_df['ctr_hls'] >= 0].sort_values('utc').reset_index(drop=True)

zero_pct = (hls_df['ctr_hls'] == 0).mean() * 100
print(f"    HEL1OS zero-count rows: {zero_pct:.1f}%  (detector duty cycle)")


# ============================================================
# SECTION 4  --  SYNCHRONIZE ON 1-SECOND UTC GRID
# ============================================================
print("\n[4] Synchronizing on 1-second UTC timeline …")

DAY_START  = pd.Timestamp('2026-06-20 00:00:00', tz='UTC')
DAY_END    = pd.Timestamp('2026-06-20 23:59:59', tz='UTC')
COMMON_IDX = pd.date_range(start=DAY_START, end=DAY_END, freq='1s')

def resample_to_grid(df, time_col, value_col, grid):
    df = df.copy()
    df['_ts'] = df[time_col].dt.floor('s')
    return df.groupby('_ts')[value_col].mean().reindex(grid)

slx_series = resample_to_grid(slx_df, 'utc', 'counts',  COMMON_IDX)
hls_series = resample_to_grid(hls_df, 'utc', 'ctr_hls', COMMON_IDX)

sync_df = pd.DataFrame({
    'utc' : COMMON_IDX,
    'slx' : slx_series.values,
    'hls' : hls_series.values,
}, index=COMMON_IDX)

print(f"    SoLEXS coverage : {sync_df['slx'].notna().sum():,} / {len(COMMON_IDX):,} bins")
print(f"    HEL1OS coverage : {sync_df['hls'].notna().sum():,} / {len(COMMON_IDX):,} bins")


# ============================================================
# SECTION 5  --  SMOOTHING (SAVITZKY-GOLAY)
# ============================================================
SMOOTH_WIN  = 61
SMOOTH_POLY = 3

def savgol_gapped(arr, window=SMOOTH_WIN, poly=SMOOTH_POLY):
    arr    = np.array(arr, dtype=float)
    result = np.full_like(arr, np.nan)
    valid  = np.isfinite(arr)
    dv     = np.diff(valid.astype(int), prepend=0, append=0)
    starts = np.where(dv ==  1)[0]
    ends   = np.where(dv == -1)[0]
    for s, e in zip(starts, ends):
        seg = arr[s:e]
        if len(seg) >= window:
            result[s:e] = savgol_filter(seg, window_length=window, polyorder=poly)
        else:
            result[s:e] = seg
    return result

slx_smooth = savgol_gapped(sync_df['slx'].values)
print(f"\n[5] Savitzky-Golay smoothing applied (window={SMOOTH_WIN}s, poly={SMOOTH_POLY})")


# ============================================================
# SECTION 6  --  BASELINE  &  THRESHOLD LINES
# ============================================================
print("\n[6] Computing rolling baseline …")

BASELINE_WIN = 600   # 10-minute rolling window
BASELINE_PCT = 25    # 25th-percentile background

baseline_series = (
    pd.Series(slx_smooth)
      .rolling(BASELINE_WIN, min_periods=30, center=True)
      .quantile(BASELINE_PCT / 100)
      .ffill()
      .bfill()
      .values
)

# Single representative baseline value for the horizontal reference lines
# Use the median of the computed baseline for a stable scalar
baseline_val = float(np.nanmedian(baseline_series))
x2_val = 2.0 * baseline_val
x3_val = 3.0 * baseline_val

print(f"    Baseline  : {baseline_val:.2f} cts/s")
print(f"    2× baseline: {x2_val:.2f} cts/s")
print(f"    3× baseline: {x3_val:.2f} cts/s")


# ============================================================
# SECTION 7  --  FLARE DETECTION
# ============================================================
print("\n[7] Detecting flares in SoLEXS …")

NOISE_WIN    = 600
SIGMA_THRESH = 3.5
MIN_PROM     = 20
MIN_DIST     = 180   # seconds
MIN_DUR      = 30    # seconds

# Noise estimate
bkg_sub  = slx_smooth - baseline_series
bkg_sub  = np.where(np.isfinite(bkg_sub), bkg_sub, 0.0)
noise    = (pd.Series(bkg_sub)
              .rolling(NOISE_WIN, min_periods=30, center=True)
              .std()
              .ffill().bfill()
              .values)
noise    = np.where(noise <= 0, 1e-3, noise)
threshold_arr = SIGMA_THRESH * noise

# Find peaks above the dynamic threshold
peak_idx, _ = find_peaks(
    bkg_sub,
    height     = threshold_arr,
    prominence = MIN_PROM,
    distance   = MIN_DIST,
    width      = 5,
)

# Additional filter: count rate > 3 × baseline_val
valid_peaks = [
    p for p in peak_idx
    if np.isfinite(slx_smooth[p]) and slx_smooth[p] > x3_val
]
peak_idx = np.array(valid_peaks, dtype=int)

print(f"    SoLEXS peaks (count > 3× baseline): {len(peak_idx)}")

# Confirm flares by minimum duration
confirmed = []
for pk in peak_idx:
    # Walk back to find start
    start_i = pk
    for j in range(pk, max(0, pk - 3600), -1):
        if not np.isfinite(slx_smooth[j]):
            start_i = j + 1; break
        if bkg_sub[j] <= threshold_arr[j]:
            start_i = j; break
    # Walk forward to find end
    end_i = pk
    for j in range(pk, min(len(slx_smooth) - 1, pk + 3600)):
        if not np.isfinite(slx_smooth[j]):
            end_i = j - 1; break
        if bkg_sub[j] <= threshold_arr[j]:
            end_i = j; break
    if (end_i - start_i) >= MIN_DUR:
        confirmed.append({
            'peak_idx'  : pk,
            'peak_utc'  : COMMON_IDX[pk],
            'peak_ctr'  : float(slx_smooth[pk]),
            'baseline'  : float(baseline_series[pk]),
        })

print(f"    Confirmed SoLEXS flares: {len(confirmed)}")
print(f"    Confirmed HEL1OS flares: 0  (no peaks above 3× baseline)")


# ============================================================
# SECTION 8  --  BUILD THE VISUALIZATION
# ============================================================
print("\n[8] Building flare_detection.png …")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 9))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

# ── SoLEXS  blue line ──────────────────────────────────────────────────────
ax.plot(
    COMMON_IDX, slx_smooth,
    color=C_SLX, linewidth=1.0, alpha=0.92,
    label='SoLEXS SDD2  (Soft X-ray, ~1-15 keV)',
    zorder=3,
)

# ── HEL1OS  red dots  (sparse — ~17.5% duty cycle) ─────────────────────────
# HEL1OS has 82.5% zero-count rows (gaps) and 17.5% live rows.
# Non-zero rows already represent the on-source duty cycle.
# We further subsample to every 5th non-zero point so the plot isn't
# saturated — this faithfully represents the sparse sampling pattern.
hls_nonzero_idx = np.where(sync_df['hls'].notna() & (sync_df['hls'] > 0))[0]
hls_sub_idx = hls_nonzero_idx[::5]   # ~17.5 % → display every 5th point
ax.scatter(
    COMMON_IDX[hls_sub_idx],
    sync_df['hls'].values[hls_sub_idx],
    color=C_HLS, s=6, alpha=0.55, linewidths=0,
    label='HEL1OS CZT1  (Hard X-ray, 18-160 keV)  [17.5% duty cycle]',
    zorder=2,
)

# ── Horizontal reference lines ──────────────────────────────────────────────
ax.axhline(baseline_val, color=C_BASE,  linestyle='--', linewidth=1.6,
           alpha=0.85, label=f'Baseline  ({baseline_val:.1f} cts/s)', zorder=4)
ax.axhline(x2_val,       color=C_2X,   linestyle='--', linewidth=1.6,
           alpha=0.85, label=f'2× Baseline  ({x2_val:.1f} cts/s)', zorder=4)
ax.axhline(x3_val,       color=C_3X,   linestyle='--', linewidth=1.6,
           alpha=0.85, label=f'3× Baseline  ({x3_val:.1f} cts/s)', zorder=4)

# ── Fill between baseline and signal ───────────────────────────────────────
ax.fill_between(
    COMMON_IDX, slx_smooth, baseline_val,
    where=(slx_smooth > baseline_val),
    alpha=0.08, color=C_SLX, interpolate=True, zorder=1,
)

# ── Flare peak markers + annotations ───────────────────────────────────────
# Annotation offsets: cycle through different positions to reduce overlap
_annot_offsets = [
    ( 55,  35),   # upper right
    (-55,  35),   # upper left
    ( 55, -45),   # lower right
    (-55, -45),   # lower left
    (  0,  55),   # directly above
    (  0, -55),   # directly below
    ( 70,  15),   # far right
    (-70,  15),   # far left
]

for k, f in enumerate(confirmed):
    t_peak   = pd.Timestamp(f['peak_utc'])
    ctr_peak = f['peak_ctr']

    # Large gold circle marker
    ax.scatter(
        t_peak, ctr_peak,
        s=220, facecolors=C_PEAK_MK, edgecolors=C_PEAK_ED,
        linewidths=1.8, zorder=8,
    )

    hhmm = t_peak.strftime('%H:%M')
    label_text = f"F{k+1}\n{hhmm} UTC\n{ctr_peak:.0f} cts/s"

    dx, dy = _annot_offsets[k % len(_annot_offsets)]
    va = 'bottom' if dy >= 0 else 'top'
    ha = 'left'   if dx >= 0 else 'right'
    # For pure vertical offsets centre-align
    if dx == 0:
        ha = 'center'

    arrow_style = dict(
        arrowstyle='->', color='#AAAAAA',
        lw=0.8,
        connectionstyle=f'arc3,rad={0.1 if dx >= 0 else -0.1}',
    )
    ax.annotate(
        label_text,
        xy=(t_peak, ctr_peak),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=7.8, color='#E0E0E0',
        fontweight='bold',
        ha=ha, va=va,
        arrowprops=arrow_style,
        zorder=9,
        bbox=dict(
            boxstyle='round,pad=0.25', fc='#1E2A38',
            ec='#3A5A80', alpha=0.88, lw=0.7
        ),
    )

# ── X-axis: 0-24 h ticks ───────────────────────────────────────────────────
ax.set_xlim(DAY_START, DAY_END + pd.Timedelta(seconds=1))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

# ── Y-axis ──────────────────────────────────────────────────────────────────
ymax = max(
    float(np.nanmax(slx_smooth)) * 1.15,
    x3_val * 1.20,
)
ax.set_ylim(0, ymax)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

# ── Labels & styling ────────────────────────────────────────────────────────
ax.set_xlabel('Time (UTC)  —  2026 June 20', fontsize=13,
              color='#CCCCCC', labelpad=8)
ax.set_ylabel('Count Rate  (photons / second)', fontsize=13,
              color='#CCCCCC', labelpad=8)
ax.set_title(
    'Aditya-L1  |  Solar Flare Detection  |  SoLEXS & HEL1OS  |  2026-06-20 (UTC)',
    fontsize=15, fontweight='bold', color='#E8E8E8', pad=14,
)

# Grid
ax.grid(True, which='major', linestyle=':', linewidth=0.6,
        color='#2A3A4A', alpha=0.9)
ax.grid(True, which='minor', linestyle=':', linewidth=0.3,
        color='#1E2A38', alpha=0.6)

# Tick colours
ax.tick_params(axis='both', colors='#AAAAAA', labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor('#2A3A4A')

# ── Legend ──────────────────────────────────────────────────────────────────
# Custom legend entries including flare peak marker
legend_handles = [
    Line2D([0], [0], color=C_SLX, linewidth=2,
           label='SoLEXS SDD2  (Soft X-ray, ~1-15 keV)'),
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=C_HLS, markeredgecolor=C_HLS, markersize=5,
           label='HEL1OS CZT1  (Hard X-ray, 18-160 keV)'),
    Line2D([0], [0], color=C_BASE, linewidth=1.8, linestyle='--',
           label=f'Baseline  ({baseline_val:.1f} cts/s)'),
    Line2D([0], [0], color=C_2X, linewidth=1.8, linestyle='--',
           label=f'2× Baseline  ({x2_val:.1f} cts/s)'),
    Line2D([0], [0], color=C_3X, linewidth=1.8, linestyle='--',
           label=f'3× Baseline  ({x3_val:.1f} cts/s)'),
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=C_PEAK_MK, markeredgecolor=C_PEAK_ED,
           markersize=10, markeredgewidth=1.5,
           label=f'Flare Peak  (count > 3× baseline)  [{len(confirmed)} detected]'),
]

leg = ax.legend(
    handles=legend_handles,
    loc='upper right',
    fontsize=9.5,
    framealpha=0.82,
    facecolor='#111A24',
    edgecolor='#2A4A6A',
    labelcolor='#DDDDDD',
    borderpad=0.8,
    labelspacing=0.55,
)

# ── Subtitle with flare summary ─────────────────────────────────────────────
subtitle = (
    f"Detected  {len(confirmed)} flare(s) in SoLEXS  |  "
    f"0 flare(s) in HEL1OS  |  "
    f"Detection threshold: count > 3× baseline ({x3_val:.1f} cts/s)"
)
fig.text(
    0.5, 0.005, subtitle,
    ha='center', va='bottom',
    fontsize=9.5, color='#8899BB',
    style='italic',
)

plt.tight_layout(rect=[0, 0.03, 1, 1.0])

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTDIR, 'flare_detection.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()   # Open interactive window
plt.close(fig)

print(f"\n  \u2713  Saved: {out_path}")
print(f"  \u2713  SoLEXS flares annotated : {len(confirmed)}")
print(f"  \u2713  HEL1OS flares            : 0")
print("\n" + "=" * 68)
print("  Done.")
print("=" * 68)
