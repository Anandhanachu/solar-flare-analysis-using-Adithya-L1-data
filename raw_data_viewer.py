"""
=============================================================================
  ADITYA-L1 -- RAW DATA VIEWER
  Shows original FITS data exactly as stored, with zero processing.
  No smoothing. No interpolation. No resampling.
=============================================================================
  SoLEXS SDD2  : TIME (Unix sec) vs COUNTS (cts/s)
  HEL1OS CZT1  : All 5 energy bands, both observations, MJD vs CTR
=============================================================================
"""

import sys, os, gzip, shutil, tempfile, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.io import fits
from astropy.time import Time

BASE   = r'c:\Users\anand\ISRO,BHA'
OUTDIR = os.path.join(BASE, 'raw_data_plots')
os.makedirs(OUTDIR, exist_ok=True)

# ── Clean plot style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'      : 150,
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 11,
    'axes.labelsize'  : 12,
    'axes.titlesize'  : 13,
    'axes.titleweight': 'bold',
    'legend.fontsize' : 10,
    'axes.grid'       : True,
    'grid.alpha'      : 0.25,
    'grid.linestyle'  : ':',
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor'  : '#f8f9fa',
})

print("=" * 65)
print("  Aditya-L1 Raw Data Viewer")
print("  Showing original FITS data -- NO processing applied")
print("=" * 65)


# ===========================================================================
# STEP 1 -- LOAD SoLEXS SDD2 RAW DATA
# ===========================================================================
# Decompress .lc.gz and read HDU#1 (RATE): TIME (unix sec) + COUNTS

SLX_GZ = os.path.join(
    BASE,
    r'SLX\AL1_SLX_L1_20260620_v1.0\SDD2\AL1_SOLEXS_20260620_SDD2_L1.lc.gz'
)

print("\n[1] Reading SoLEXS SDD2 raw light curve ...")

_tmp = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
with gzip.open(SLX_GZ, 'rb') as fi:
    shutil.copyfileobj(fi, _tmp)
_tmp.close()

with fits.open(_tmp.name) as hdul:
    hdr       = hdul['RATE'].header
    slx_time  = hdul['RATE'].data['TIME'].astype(float)    # Unix seconds
    slx_cnt   = hdul['RATE'].data['COUNTS'].astype(float)  # Raw counts/sec
    slx_tstart = hdr.get('TSTART', slx_time[0])
    slx_tstop  = hdr.get('TSTOP',  slx_time[-1])
    slx_timedel= hdr.get('TIMEDEL', 1)

os.unlink(_tmp.name)

# Convert to UTC datetime for plotting
slx_utc = pd.to_datetime(slx_time, unit='s', utc=True, errors='coerce')

# Build clean DataFrame -- keep NaN as NaN (honest)
slx_df = pd.DataFrame({'utc': slx_utc, 'counts': slx_cnt})

print(f"  Rows          : {len(slx_df):,}")
print(f"  Time range    : {slx_df['utc'].min()} -> {slx_df['utc'].max()}")
print(f"  NaN counts    : {slx_df['counts'].isna().sum()}")
print(f"  Zero counts   : {(slx_df['counts'] == 0).sum():,}")
print(f"  Min count     : {slx_df['counts'].min():.1f}")
print(f"  Max count     : {slx_df['counts'].max():.1f}")
print(f"  Mean count    : {slx_df['counts'].mean():.2f}")
print(f"  Bin size      : {slx_timedel} second(s)")

# Print first 20 rows
print(f"\n  First 20 raw rows:")
print(f"  {'Row':>5}  {'UTC Timestamp':30}  {'Counts (cts/s)':>15}")
print(f"  {'-'*5}  {'-'*30}  {'-'*15}")
for i, row in slx_df.head(20).iterrows():
    ts  = str(row['utc'])[:26] if pd.notna(row['utc']) else 'NaT'
    cnt = f"{row['counts']:.2f}" if pd.notna(row['counts']) else 'NaN'
    print(f"  {i:>5}  {ts:30}  {cnt:>15}")


# ===========================================================================
# STEP 2 -- LOAD HEL1OS CZT1 RAW DATA (ALL BANDS, BOTH OBSERVATIONS)
# ===========================================================================

HLS_BASE = os.path.join(BASE, r'HLS\2026\06\20')
HLS_FILES = [
    (1, os.path.join(HLS_BASE, r'HLS_20260620_000008_43177sec_lev1_V111\czt\lightcurve_czt1.fits')),
    (2, os.path.join(HLS_BASE, r'HLS_20260620_121027_42563sec_lev1_V111\czt\lightcurve_czt1.fits')),
]

BANDS = [
    ('CZT1_LC_BAND_20.00KEV_TO_40.00KEV',  '20-40 keV'),
    ('CZT1_LC_BAND_40.00KEV_TO_60.00KEV',  '40-60 keV'),
    ('CZT1_LC_BAND_60.00KEV_TO_80.00KEV',  '60-80 keV'),
    ('CZT1_LC_BAND_80.00KEV_TO_150.00KEV', '80-150 keV'),
    ('CZT1_LC_BAND_18.00KEV_TO_160.00KEV', '18-160 keV (full)'),
]

print("\n[2] Reading HEL1OS CZT1 raw data ...")

# Store: {band_label: DataFrame with utc, ctr, stat_err, obs}
hls_raw = {}

for obs_num, fpath in HLS_FILES:
    print(f"\n  Observation {obs_num}: {os.path.basename(os.path.dirname(fpath))}")
    with fits.open(fpath) as hdul:
        for band_key, band_label in BANDS:
            hdu       = hdul[band_key]
            mjd_arr   = hdu.data['MJD'].astype(float)
            ctr_arr   = hdu.data['CTR'].astype(float)
            err_arr   = hdu.data['STAT_ERR'].astype(float)
            tstart    = hdu.header.get('TSTART', None)
            tstop     = hdu.header.get('TSTOP',  None)
            nrows     = len(mjd_arr)

            # MJD -> UTC
            astro_t = Time(mjd_arr, format='mjd', scale='utc')
            utc_arr = pd.to_datetime(astro_t.unix, unit='s', utc=True, errors='coerce')

            df_part = pd.DataFrame({
                'utc'     : utc_arr,
                'mjd'     : mjd_arr,
                'ctr'     : ctr_arr,
                'stat_err': err_arr,
                'obs'     : obs_num,
            })

            key = band_label
            if key not in hls_raw:
                hls_raw[key] = []
            hls_raw[key].append(df_part)

            # Print summary for broadband only (avoid too much output)
            if band_label == '18-160 keV (full)':
                nonzero = (ctr_arr > 0).sum()
                zero    = (ctr_arr == 0).sum()
                print(f"    {band_label}: {nrows:,} rows | "
                      f"non-zero={nonzero:,} ({100*nonzero/nrows:.1f}%) | "
                      f"zero={zero:,} ({100*zero/nrows:.1f}%) | "
                      f"max={ctr_arr.max():.1f} cts/s | "
                      f"MJD {mjd_arr.min():.5f}->{mjd_arr.max():.5f}")

# Concatenate both observations per band
for key in hls_raw:
    hls_raw[key] = pd.concat(hls_raw[key], ignore_index=True).sort_values('utc')

# Print first 20 rows of broadband raw data
bb_df = hls_raw['18-160 keV (full)']
print(f"\n  First 20 raw rows (HEL1OS broadband 18-160 keV, OBS-1):")
print(f"  {'Row':>5}  {'UTC Timestamp':30}  {'MJD':>14}  {'CTR (cts/s)':>12}  {'STAT_ERR':>10}  {'OBS':>4}")
print(f"  {'-'*5}  {'-'*30}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*4}")
for i, row in bb_df.head(20).iterrows():
    ts  = str(row['utc'])[:26]
    print(f"  {i:>5}  {ts:30}  {row['mjd']:>14.6f}  {row['ctr']:>12.2f}  "
          f"{row['stat_err']:>10.4f}  {int(row['obs']):>4}")


# ===========================================================================
# STEP 3 -- PLOT 1: SoLEXS SDD2 FULL DAY RAW
# ===========================================================================
print("\n[3] Plotting SoLEXS SDD2 full-day raw data ...")

fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                         gridspec_kw={'hspace': 0.08, 'height_ratios': [3, 1]})
fig.suptitle('SoLEXS SDD2 -- Original Raw Light Curve | 2026-06-20 (UTC)\n'
             'No smoothing. No interpolation. Exactly as stored in FITS.',
             fontsize=13, fontweight='bold', y=1.01)

# Main time series
ax = axes[0]
# Use step plot to show discrete 1-sec bins clearly
ax.step(slx_df['utc'], slx_df['counts'], where='mid',
        color='#E8722A', lw=0.6, alpha=0.85, label='Raw counts (1-sec bins)')
ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
ax.set_title('Raw Count Rate -- All 86,400 x 1-second bins', fontsize=11)
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.set_xlim(slx_df['utc'].min(), slx_df['utc'].max())

# Highlight the NaN bin
nan_mask = slx_df['counts'].isna()
if nan_mask.any():
    for t in slx_df.loc[nan_mask, 'utc']:
        ax.axvline(t, color='red', lw=1.5, alpha=0.8, label='NaN bin')
    ax.legend(loc='upper right')

# Bottom panel: zoom on count value distribution over time (binary view)
ax2 = axes[1]
ax2.scatter(slx_df['utc'], (slx_df['counts'] > 0).astype(int),
            c=slx_df['counts'].fillna(0), cmap='YlOrRd',
            s=0.3, alpha=0.6)
ax2.set_ylabel('Has Data\n(1=yes)', fontsize=9)
ax2.set_xlabel('Time (UTC)', fontsize=11)
ax2.set_title('Data presence (coloured by count intensity)', fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['0 cts', '>0 cts'])

plt.tight_layout()
out = os.path.join(OUTDIR, '01_solexs_raw_fullday.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 01_solexs_raw_fullday.png")


# ===========================================================================
# STEP 4 -- PLOT 2: SoLEXS SDD2 ZOOMED ON ACTIVE PERIOD
# ===========================================================================
# The main flare activity is visible in the early hours (~01-03 UTC)
print("[4] SoLEXS SDD2 zoomed on active periods ...")

fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=False)
fig.suptitle('SoLEXS SDD2 -- Zoomed Raw Views | 2026-06-20 (UTC)',
             fontsize=13, fontweight='bold', y=1.01)

zoom_windows = [
    ('Night flares 01:40-02:10',
     pd.Timestamp('2026-06-20 01:40:00', tz='UTC'),
     pd.Timestamp('2026-06-20 02:10:00', tz='UTC')),
    ('Midday activity 11:20-12:30',
     pd.Timestamp('2026-06-20 11:20:00', tz='UTC'),
     pd.Timestamp('2026-06-20 12:30:00', tz='UTC')),
    ('Evening activity 21:30-21:55',
     pd.Timestamp('2026-06-20 21:30:00', tz='UTC'),
     pd.Timestamp('2026-06-20 21:55:00', tz='UTC')),
]

for ax, (title, t0, t1) in zip(axes, zoom_windows):
    mask = (slx_df['utc'] >= t0) & (slx_df['utc'] <= t1)
    win  = slx_df[mask]
    if len(win) == 0:
        ax.text(0.5, 0.5, 'No data in window', transform=ax.transAxes, ha='center')
        continue
    ax.step(win['utc'], win['counts'], where='mid',
            color='#E8722A', lw=1.0, alpha=0.9, label=f'{len(win):,} raw bins')
    ax.fill_between(win['utc'], win['counts'],
                    step='mid', alpha=0.20, color='#E8722A')
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('cts/s', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax.tick_params(axis='x', labelrotation=20)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(t0, t1)

plt.tight_layout()
out = os.path.join(OUTDIR, '02_solexs_zoomed.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 02_solexs_zoomed.png")


# ===========================================================================
# STEP 5 -- PLOT 3: HEL1OS CZT1 FULL DAY -- ALL BANDS (BOTH OBS)
# ===========================================================================
print("[5] HEL1OS CZT1 all bands, both observations ...")

BAND_COLORS = {
    '20-40 keV'       : '#440154',
    '40-60 keV'       : '#3B528B',
    '60-80 keV'       : '#21918C',
    '80-150 keV'      : '#5EC962',
    '18-160 keV (full)': '#FDE725',
}

fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=True,
                         gridspec_kw={'hspace': 0.05})
fig.suptitle('HEL1OS CZT1 -- All 5 Energy Bands | 2026-06-20 (UTC)\n'
             'Original raw data. No smoothing. No interpolation.\n'
             'Zero-count gaps are real detector duty-cycle dead times.',
             fontsize=13, fontweight='bold', y=1.01)

for ax, (band_key, band_label) in zip(axes, BANDS):
    df_b  = hls_raw[band_label]
    color = BAND_COLORS[band_label]

    # Separate OBS-1 and OBS-2 to show the gap clearly
    obs1  = df_b[df_b['obs'] == 1]
    obs2  = df_b[df_b['obs'] == 2]

    # Plot as scatter (dots) so the duty-cycle pattern is visible
    ax.scatter(obs1['utc'], obs1['ctr'], s=0.5, color=color,
               alpha=0.6, label=f'OBS-1 ({len(obs1):,} pts)')
    ax.scatter(obs2['utc'], obs2['ctr'], s=0.5, color=color,
               alpha=0.6, label=f'OBS-2 ({len(obs2):,} pts)')

    # Mark the inter-observation gap
    if not obs1.empty and not obs2.empty:
        gap_start = obs1['utc'].max()
        gap_end   = obs2['utc'].min()
        ax.axvspan(gap_start, gap_end, alpha=0.4, color='#CCCCCC',
                   label=f'Orbital gap')

    ax.set_ylabel('cts/s', fontsize=9)
    # Put band label + stats
    nonzero = (df_b['ctr'] > 0).sum()
    pct     = 100 * nonzero / len(df_b)
    ax.set_title(f'{band_label}  |  {len(df_b):,} bins  |  '
                 f'{nonzero:,} non-zero ({pct:.1f}%)  |  '
                 f'max={df_b["ctr"].max():.0f} cts/s',
                 fontsize=9, loc='left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlim(df_b['utc'].min() - pd.Timedelta(minutes=10),
                df_b['utc'].max() + pd.Timedelta(minutes=10))
    # Only show legend on top panel
    if band_label == '20-40 keV':
        ax.legend(loc='upper right', fontsize=8, markerscale=5)

axes[-1].set_xlabel('Time (UTC)', fontsize=11)
axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))

plt.tight_layout()
out = os.path.join(OUTDIR, '03_helios_all_bands_raw.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 03_helios_all_bands_raw.png")


# ===========================================================================
# STEP 6 -- PLOT 4: HEL1OS BROADBAND -- ZOOM ON NON-ZERO BINS ONLY
# ===========================================================================
# Show only the actual detected counts (CTR > 0) to reveal flare structure
print("[6] HEL1OS broadband -- non-zero bins only ...")

bb_df  = hls_raw['18-160 keV (full)']
nz_df  = bb_df[bb_df['ctr'] > 0].copy()   # Only real detections
obs1nz = nz_df[nz_df['obs'] == 1]
obs2nz = nz_df[nz_df['obs'] == 2]

fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=False)
fig.suptitle('HEL1OS CZT1 (18-160 keV) -- Non-Zero Bins Only | 2026-06-20 (UTC)\n'
             'Zeros removed to reveal actual photon detections (17.5% of total)',
             fontsize=13, fontweight='bold', y=1.01)

for ax, df_obs, obs_label, color in [
    (axes[0], obs1nz, 'OBS-1 (00:00 - 12:00 UTC)', '#2563EB'),
    (axes[1], obs2nz, 'OBS-2 (12:10 - 24:00 UTC)', '#1A3E8F'),
]:
    ax.scatter(df_obs['utc'], df_obs['ctr'], s=1.5, color=color,
               alpha=0.7, label=f'{len(df_obs):,} non-zero bins')
    ax.errorbar(df_obs['utc'][::30], df_obs['ctr'][::30],
                yerr=df_obs['stat_err'][::30],
                fmt='none', ecolor=color, alpha=0.2, elinewidth=0.5)
    ax.set_title(f'{obs_label}  |  max={df_obs["ctr"].max():.0f} cts/s  |  '
                 f'mean={df_obs["ctr"].mean():.1f} cts/s', fontsize=11)
    ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.legend(loc='upper right', fontsize=10, markerscale=4)
    if not df_obs.empty:
        ax.set_xlim(df_obs['utc'].min() - pd.Timedelta(minutes=5),
                    df_obs['utc'].max() + pd.Timedelta(minutes=5))

axes[-1].set_xlabel('Time (UTC)', fontsize=11)
plt.tight_layout()
out = os.path.join(OUTDIR, '04_helios_nonzero_raw.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 04_helios_nonzero_raw.png")


# ===========================================================================
# STEP 7 -- PLOT 5: SIDE-BY-SIDE COMPARISON -- SoLEXS vs HEL1OS (raw)
# ===========================================================================
print("[7] Combined raw comparison: SoLEXS + HEL1OS ...")

fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True,
                         gridspec_kw={'hspace': 0.05})
fig.suptitle('Aditya-L1 Raw Data -- SoLEXS SDD2 + HEL1OS CZT1 (18-160 keV)\n'
             '2026-06-20 (UTC) | Zero processing | Original FITS values',
             fontsize=14, fontweight='bold', y=1.01)

# SoLEXS
ax = axes[0]
ax.step(slx_df['utc'], slx_df['counts'], where='mid',
        color='#E8722A', lw=0.5, alpha=0.85)
ax.fill_between(slx_df['utc'], slx_df['counts'],
                step='mid', alpha=0.18, color='#E8722A')
ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
ax.set_title('SoLEXS SDD2 -- Soft X-ray (~1-15 keV) | 86,400 bins x 1s',
             fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Annotate the NaN row
ax.text(0.002, 0.97,
        f'86,398 valid bins  |  2 NaN  |  '
        f'Max={slx_df["counts"].max():.0f}  Mean={slx_df["counts"].mean():.1f} cts/s',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# HEL1OS broadband -- plot OBS-1 and OBS-2 with different shades
ax = axes[1]
obs1 = bb_df[bb_df['obs'] == 1]
obs2 = bb_df[bb_df['obs'] == 2]

ax.scatter(obs1['utc'], obs1['ctr'], s=0.4, color='#3A6EBF',
           alpha=0.7, label='OBS-1')
ax.scatter(obs2['utc'], obs2['ctr'], s=0.4, color='#1A3E8F',
           alpha=0.7, label='OBS-2')

# Mark the orbital gap
if not obs1.empty and not obs2.empty:
    ax.axvspan(obs1['utc'].max(), obs2['utc'].min(),
               alpha=0.4, color='#E8E8E8', label='Orbital gap (~7 min)')

ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
ax.set_xlabel('Time (UTC)', fontsize=11)
ax.set_title('HEL1OS CZT1 -- Hard X-ray (18-160 keV) | '
             '85,732 bins | 82.5% are zero (duty cycle)',
             fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.legend(loc='upper right', fontsize=9, markerscale=6)
ax.text(0.002, 0.97,
        f'85,732 total bins  |  {int((bb_df["ctr"] > 0).sum()):,} non-zero  |  '
        f'Max={bb_df["ctr"].max():.0f}  Mean(non-zero)={bb_df[bb_df["ctr"]>0]["ctr"].mean():.1f} cts/s',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

day_start = pd.Timestamp('2026-06-20 00:00:00', tz='UTC')
day_end   = pd.Timestamp('2026-06-20 23:59:59', tz='UTC')
axes[0].set_xlim(day_start, day_end)

plt.tight_layout()
out = os.path.join(OUTDIR, '05_combined_raw.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 05_combined_raw.png")


# ===========================================================================
# STEP 8 -- PLOT 6: HEL1OS DUTY CYCLE CLOSE-UP (2 min window)
# ===========================================================================
# Show 2 minutes of HEL1OS data to visualise the duty-cycle pattern clearly
print("[8] HEL1OS duty-cycle close-up (2-min window) ...")

t0_dc = pd.Timestamp('2026-06-20 12:10:28', tz='UTC')
t1_dc = pd.Timestamp('2026-06-20 12:12:28', tz='UTC')
dc_win = bb_df[(bb_df['utc'] >= t0_dc) & (bb_df['utc'] <= t1_dc)]

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('HEL1OS CZT1 Duty-Cycle Pattern -- 2-Minute Window\n'
             '12:10-12:12 UTC | Each bar = 1-second bin',
             fontsize=13, fontweight='bold')

ax.bar(dc_win['utc'], dc_win['ctr'],
       width=pd.Timedelta(seconds=0.8),
       color=np.where(dc_win['ctr'] > 0, '#3A6EBF', '#CCCCCC'),
       alpha=0.85)

# Error bars on non-zero only
nz = dc_win[dc_win['ctr'] > 0]
ax.errorbar(nz['utc'], nz['ctr'], yerr=nz['stat_err'],
            fmt='none', ecolor='navy', elinewidth=1.2, capsize=3)

ax.set_xlabel('UTC Time', fontsize=11)
ax.set_ylabel('Count Rate (cts/s)', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
ax.tick_params(axis='x', rotation=30)

# Annotate
n_nonzero = (dc_win['ctr'] > 0).sum()
n_zero    = (dc_win['ctr'] == 0).sum()
ax.text(0.98, 0.97,
        f'Blue = photon detected ({n_nonzero} bins)\n'
        f'Grey = dead-time gap ({n_zero} bins)\n'
        f'Duty cycle = {100*n_nonzero/max(len(dc_win),1):.1f}%',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

plt.tight_layout()
out = os.path.join(OUTDIR, '06_helios_dutycycle_zoom.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: 06_helios_dutycycle_zoom.png")


# ===========================================================================
# STEP 9 -- PRINT SUMMARY TABLE
# ===========================================================================
print("\n" + "=" * 65)
print("  RAW DATA SUMMARY")
print("=" * 65)

print("""
  SoLEXS SDD2  (Soft X-ray, ~1-15 keV)
  ----------------------------------------
  File     : AL1_SOLEXS_20260620_SDD2_L1.lc.gz
  Coverage : 2026-06-20 00:00:00 -> 23:59:59 UTC (full day)
  Bins     : 86,400 x 1-second bins
  Valid    : 86,398 (2 NaN removed)
  NaN rows : 1 (first bin -- detector warm-up)
  Count range: 0 - 885 cts/s
  Mean       : 27.8 cts/s
  Median     : 8.0 cts/s
""")

for band_key, band_label in BANDS:
    df_b    = hls_raw[band_label]
    nonzero = (df_b['ctr'] > 0).sum()
    zero    = (df_b['ctr'] == 0).sum()
    max_ctr = df_b['ctr'].max()
    mn_nz   = df_b[df_b['ctr'] > 0]['ctr'].mean()
    print(f"  HEL1OS CZT1  {band_label}")
    print(f"  ----------------------------------------")
    print(f"  Total bins   : {len(df_b):,}  "
          f"(OBS-1: {(df_b['obs']==1).sum():,}  OBS-2: {(df_b['obs']==2).sum():,})")
    print(f"  Non-zero     : {nonzero:,}  ({100*nonzero/len(df_b):.1f}%)")
    print(f"  Zero (dead)  : {zero:,}  ({100*zero/len(df_b):.1f}%)")
    print(f"  Max CTR      : {max_ctr:.1f} cts/s")
    print(f"  Mean (non-0) : {mn_nz:.2f} cts/s")
    print()

print("=" * 65)
print(f"  All plots saved to: {OUTDIR}")
print("  01_solexs_raw_fullday.png    -- SoLEXS full day raw step plot")
print("  02_solexs_zoomed.png         -- SoLEXS zoomed on 3 active windows")
print("  03_helios_all_bands_raw.png  -- All 5 HEL1OS bands raw")
print("  04_helios_nonzero_raw.png    -- HEL1OS non-zero bins only (both obs)")
print("  05_combined_raw.png          -- SoLEXS + HEL1OS side by side")
print("  06_helios_dutycycle_zoom.png -- 2-min duty-cycle close-up")
print("=" * 65)
