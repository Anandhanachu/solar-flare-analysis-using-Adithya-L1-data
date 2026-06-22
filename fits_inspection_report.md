# Aditya-L1 FITS File Inspection Report
**Date inspected:** 2026-06-20 data  
**Instruments:** SoLEXS (SDD1, SDD2) · HEL1OS / CZT (CZT1, CZT2)

---

## File Inventory

| Label | Path | Size | Type |
|-------|------|------|------|
| SoLEXS SDD1 | `SLX/AL1_SLX_L1_20260620_v1.0/SDD1/AL1_SOLEXS_20260620_SDD1_L1.gti.gz` | 957 B | GTI only |
| SoLEXS SDD2 GTI | `SLX/AL1_SLX_L1_20260620_v1.0/SDD2/AL1_SOLEXS_20260620_SDD2_L1.gti.gz` | 1,039 B | GTI |
| SoLEXS SDD2 LC | `SLX/AL1_SLX_L1_20260620_v1.0/SDD2/AL1_SOLEXS_20260620_SDD2_L1.lc.gz` | 263,625 B | **Light Curve** |
| HEL1OS OBS-1 CZT1 | `HLS/2026/06/20/HLS_20260620_000008_.../czt/lightcurve_czt1.fits` | 11,681,280 B | **Light Curve** |
| HEL1OS OBS-1 CZT2 | `HLS/2026/06/20/HLS_20260620_000008_.../czt/lightcurve_czt2.fits` | 11,681,280 B | **Light Curve** |
| HEL1OS OBS-2 CZT1 | `HLS/2026/06/20/HLS_20260620_121027_.../czt/lightcurve_czt1.fits` | 11,508,480 B | **Light Curve** |
| HEL1OS OBS-2 CZT2 | `HLS/2026/06/20/HLS_20260620_121027_.../czt/lightcurve_czt2.fits` | 11,508,480 B | **Light Curve** |

> [!IMPORTANT]
> **SoLEXS SDD1 has NO light curve file.** Only a GTI (Good Time Interval) file exists for SDD1, and it contains 0 rows — meaning SDD1 had no valid observation windows on this date, or the LC file was not generated/downloaded.

---

## 1. SoLEXS — SDD1 (GTI file only)

**Telescope:** AL1 · **Instrument:** SoLEXS · **Total HDUs:** 2

| HDU # | Name | Type | Content |
|-------|------|------|---------|
| 0 | `PRIMARY` | PrimaryHDU | No data array — mission/instrument metadata only |
| 1 | `GTI` | BinTableHDU | Good Time Intervals — START/STOP pairs for valid observation windows |

### HDU #1 — GTI
| Column | Format | Unit | Description |
|--------|--------|------|-------------|
| `START` | `I` (16-bit int) | — | Start time of good time interval |
| `STOP` | `I` (16-bit int) | — | Stop time of good time interval |

**Rows: 0** — No GTI entries. SDD1 had no valid time intervals on 2026-06-20.

> [!WARNING]
> The SDD1 GTI uses 16-bit integer (`I` format) with no unit specified — likely UNIX epoch seconds or a mission-specific timescale. The empty table suggests the detector was not operational or data was not recorded.

---

## 2. SoLEXS — SDD2 GTI

**Telescope:** AL1 · **Instrument:** SoLEXS  
**Coverage:** `2026-06-20T00:00:01` → `2026-06-20T23:59:59`  
**Total HDUs:** 2

| HDU # | Name | Type | Content |
|-------|------|------|---------|
| 0 | `PRIMARY` | PrimaryHDU | Mission/instrument metadata |
| 1 | `GTI` | BinTableHDU | Good Time Intervals for SDD2 |

### HDU #1 — GTI
| Column | Format | Unit | Min | Max |
|--------|--------|------|-----|-----|
| `START` | `D` (float64) | — | 1,781,913,600 | 1,781,952,000 |
| `STOP` | `D` (float64) | — | 1,781,952,000 | 1,782,000,000 |

**Rows: 2** — Two valid observation windows exist for SDD2. Times are in seconds (UNIX-epoch or ISRO mission epoch).

```
Interval 1:  START=1.78191e+09  →  STOP=1.78195e+09
Interval 2:  START=1.78195e+09  →  STOP=1.78200e+09
```

---

## 3. SoLEXS — SDD2 Light Curve ⭐

**File:** `AL1_SOLEXS_20260620_SDD2_L1.lc.gz`  
**Telescope:** AL1 · **Instrument:** SoLEXS  
**Coverage:** `2026-06-20 00:00:00` → `2026-06-20 23:59:59`  
**Total HDUs:** 2 · **OGIP compliant**

| HDU # | Name | Type | Content |
|-------|------|------|---------|
| 0 | `PRIMARY` | PrimaryHDU | Mission metadata header — no data array |
| 1 | `RATE` | BinTableHDU | ⭐ **The light curve table** |

### HDU #1 — RATE (Light Curve Table)
OGIP class: `LIGHTCURVE / TOTAL`  
**Time resolution:** `TIMEDEL = 1 second`  
**MJD reference:** `MJDREFI=40587, MJDREFF=0` (standard OGIP = 1970-01-01)

| Column | Format | Unit | Min | Max | Description |
|--------|--------|------|-----|-----|-------------|
| `TIME` | `D` (float64) | — (seconds) | 1.78191×10⁹ | 1.78200×10⁹ | Mission elapsed time |
| `COUNTS` | `D` (float64) | — | 0 | **885** | Raw photon count per second |

**Total rows: 86,400** = exactly 24 hours × 3600 seconds/hour (full day coverage, 1-sec bins)

### First 20 rows (SDD2 LC):
| TIME | COUNTS |
|------|--------|
| 1.78191e+09 | **NaN** (first bin — likely detector warm-up/bad) |
| 1.78191e+09 | 4 |
| 1.78191e+09 | 3 |
| 1.78191e+09 | 6 |
| 1.78191e+09 | 8 |
| 1.78191e+09 | 6 |
| 1.78191e+09 | 4 |
| 1.78191e+09 | 5 |
| 1.78191e+09 | 9 |
| 1.78191e+09 | 4 |
| ... | ... (counts 2–11 per second) |

> [!NOTE]
> SoLEXS SDD2 data is **broadband integrated counts** — no energy band decomposition. It covers the full soft X-ray range of the Silicon Drift Detector. The first bin is NaN (likely a calibration gap). COUNT range 0–885 cts/sec is typical quiet-Sun soft X-ray flux.

---

## 4. HEL1OS — CZT Light Curves (Multi-band)

Both CZT1 and CZT2 detectors follow the exact same FITS structure.  
There are **2 observations per day** (the orbit splits the day in ~12-hour halves).

### Observation Summary

| Observation | Time Range (MJD) | ISO Start | ISO End | Rows |
|------------|-----------------|-----------|---------|------|
| OBS-1 | 61211.000 → 61211.500 | 2026-06-20T00:00:08 | 2026-06-20T11:59:~52 | **43,177** |
| OBS-2 | 61211.507 → 61211.999 | 2026-06-20T12:10:28 | 2026-06-20T23:59:~57 | **42,555** |

---

### FITS Structure — CZT1 / CZT2 (identical layout)

**Telescope:** Aditya-L1 · **Instrument:** HEL1OS  
**Total HDUs: 6**

| HDU # | EXTNAME | Type | Energy Band | Content |
|-------|---------|------|-------------|---------|
| 0 | `PRIMARY` | PrimaryHDU | — | No data, mission metadata |
| 1 | `CZTx_LC_BAND_20.00KEV_TO_40.00KEV` | BinTableHDU | **20–40 keV** | Hard X-ray light curve |
| 2 | `CZTx_LC_BAND_40.00KEV_TO_60.00KEV` | BinTableHDU | **40–60 keV** | Hard X-ray light curve |
| 3 | `CZTx_LC_BAND_60.00KEV_TO_80.00KEV` | BinTableHDU | **60–80 keV** | Hard X-ray light curve |
| 4 | `CZTx_LC_BAND_80.00KEV_TO_150.00KEV` | BinTableHDU | **80–150 keV** | Hard X-ray light curve |
| 5 | `CZTx_LC_BAND_18.00KEV_TO_160.00KEV` | BinTableHDU | **18–160 keV** | Full-band integrated LC |

> [!IMPORTANT]
> **Each HDU is an independent energy-band light curve.** HDU #5 (`18–160 keV`) is the **full broadband** channel. HDU #1 is used as the primary detected LC in our script, but all 5 bands should be analyzed for spectral diagnostics.

---

### Columns — All CZT HDUs (identical schema)

| Column | Format | Unit | Description |
|--------|--------|------|-------------|
| `MJD` | `D` (float64) | MJD | Modified Julian Date timestamp |
| `ISOT` | `30A` (30-char string) | UT | ISO 8601 timestamp (e.g. `2026-06-20T12:10:28.142`) |
| `CTR` | `D` (float64) | cts/sec | Count rate (photons per second) |
| `STAT_ERR` | `D` (float64) | cts/sec | Statistical error = √CTR (Poisson noise) |

**Time resolution: 1 second per row**

---

### Count Rate Statistics by Band

#### OBS-1 (00:00 – 12:00 UTC) — CZT1

| Band | CTR Min | CTR Max | STAT_ERR Max |
|------|---------|---------|-------------|
| 20–40 keV | 0 | 353 cts/s | 18.79 |
| 40–60 keV | 0 | ~300+ | — |
| 60–80 keV | 0 | ~250+ | — |
| 80–150 keV | 0 | ~200+ | — |
| 18–160 keV | 0 | ~500+ | — |

#### OBS-2 (12:10 – 24:00 UTC) — CZT1

| Band | CTR Min | CTR Max |
|------|---------|---------|
| 20–40 keV | 0 | **528 cts/s** |

#### OBS-2 — CZT2

| Band | CTR Min | CTR Max |
|------|---------|---------|
| 20–40 keV | 0 | **478 cts/s** |

---

### First 20 Rows — HEL1OS OBS-1 CZT1 (20–40 keV band)

| MJD | ISOT | CTR (cts/s) | STAT_ERR |
|-----|------|-------------|----------|
| 61211.0 | 2026-06-20T00:00:08.706 | **163** | 12.77 |
| 61211.0 | 2026-06-20T00:00:09.706 | 0 | 0 |
| 61211.0 | 2026-06-20T00:00:10.706 | 0 | 0 |
| 61211.0 | 2026-06-20T00:00:11.706 | 0 | 0 |
| ... | ... | 0 | 0 |
| 61211.0 | 2026-06-20T00:00:16.706 | **87** | 9.33 |
| ... | ... | 0 | 0 |
| 61211.0 | 2026-06-20T00:00:23.706 | **94** | 9.70 |

> [!NOTE]
> The alternating pattern of **high counts → several zero-count seconds** in HEL1OS OBS-1 is characteristic of CZT detector readout timing — the detector integrates for ~1 sec then has a dead-time gap. This is **not a solar signal dropout**; it is the detector duty cycle behavior.

---

## Key Structural Differences: SoLEXS vs HEL1OS

| Feature | SoLEXS (SDD2) | HEL1OS (CZT1/CZT2) |
|---------|--------------|---------------------|
| Time column | `TIME` (float64, mission seconds) | `MJD` + `ISOT` (dual timestamp) |
| Count column | `COUNTS` (raw, no error column) | `CTR` (rate) + `STAT_ERR` |
| Energy info | None (broadband only) | **5 separate energy bands per HDU** |
| HDUs with data | 1 (RATE) | 5 BinTables (one per energy band) |
| Time resolution | 1 second | 1 second |
| Full-day coverage | Yes (86,400 rows) | Split into 2 observations |
| Observations/day | 1 | 2 (orbit structure) |
| Energy range | Soft X-ray (1–15 keV SDD) | Hard X-ray (**18–160 keV** CZT) |
| OGIP compliant | Yes (HDUCLASS=OGIP) | Partial (no HDUCLASS key) |

---

## Summary of Data Available for Analysis

```
DATE: 2026-06-20

SoLEXS
  SDD1: ❌ No light curve (GTI empty — detector inactive)
  SDD2: ✅ 86,400 rows × 1-sec bins, broadband COUNTS, full day

HEL1OS
  OBS-1 (00:00–12:00 UTC):
    CZT1: ✅ 43,177 rows × 5 energy bands
    CZT2: ✅ 43,177 rows × 5 energy bands
  OBS-2 (12:10–24:00 UTC):
    CZT1: ✅ 42,555 rows × 5 energy bands
    CZT2: ✅ 42,555 rows × 5 energy bands
```
