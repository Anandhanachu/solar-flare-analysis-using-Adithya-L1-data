"""
Aditya-L1 FITS File Inspector
Inspects SoLEXS (SDD1, SDD2) and HEL1OS (CZT1, CZT2) light curve FITS files.
Identifies all HDUs, column metadata, data types, units, and first 20 records.
"""

import os
import gzip
import shutil
import tempfile
from astropy.io import fits
import numpy as np

SEPARATOR = "=" * 80

def print_section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def inspect_fits(filepath, label):
    print_section(f"FILE: {label}")
    print(f"  Path : {filepath}")
    print(f"  Size : {os.path.getsize(filepath):,} bytes")

    with fits.open(filepath, memmap=True) as hdul:
        print(f"\n  {'─'*70}")
        print(f"  TOTAL HDUs: {len(hdul)}")
        print(f"  {'─'*70}")

        lc_hdu_index = None

        # ── Pass 1: describe every HDU ──────────────────────────────────────
        for i, hdu in enumerate(hdul):
            hdu_type = type(hdu).__name__
            name = hdu.name if hdu.name else "(unnamed)"
            print(f"\n  ┌─ HDU #{i}  name='{name}'  type={hdu_type}")

            # Primary header summary
            hdr = hdu.header
            interesting_keys = [
                'TELESCOP','INSTRUME','DETECTOR','OBJECT','DATE-OBS','DATE-END',
                'TSTART','TSTOP','TIMEDEL','TIMEPIXR','NAXIS','NAXIS1','NAXIS2',
                'TFIELDS','EXTNAME','HDUCLASS','HDUCLAS1','HDUCLAS2',
                'TIMEUNIT','TIMEZERO','MJDREFI','MJDREFF'
            ]
            printed = []
            for key in interesting_keys:
                if key in hdr:
                    printed.append(key)
                    print(f"  │   {key:12s} = {hdr[key]!r}")
            if not printed:
                print(f"  │   (no standard solar/timing keys found in header)")

            # BinTable columns
            if isinstance(hdu, fits.BinTableHDU):
                cols = hdu.columns
                print(f"  │   Columns ({len(cols)}):")
                for col in cols:
                    unit = col.unit if col.unit else "—"
                    print(f"  │     [{col.name:20s}]  format={col.format:8s}  unit={unit}")
                nrows = hdu.data.shape[0] if hdu.data is not None else 0
                print(f"  │   Rows: {nrows:,}")

                # Detect the light curve HDU
                col_names_upper = [c.name.upper() for c in cols]
                if 'TIME' in col_names_upper and any(
                    kw in col_names_upper for kw in ['RATE','COUNTS','COUNT_RATE','FLUX','CTS']
                ):
                    lc_hdu_index = i
                    print(f"  │   *** Detected as LIGHT CURVE table ***")

            elif isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):
                shape = hdu.data.shape if hdu.data is not None else "(no data)"
                dtype = hdu.data.dtype if hdu.data is not None else "N/A"
                print(f"  │   Data shape : {shape}")
                print(f"  │   Data dtype : {dtype}")

            print(f"  └{'─'*69}")

        # ── Pass 2: print first 20 records from the light curve HDU ─────────
        if lc_hdu_index is None:
            # Fallback: pick first BinTableHDU that has data
            for i, hdu in enumerate(hdul):
                if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None and len(hdu.data) > 0:
                    lc_hdu_index = i
                    break

        if lc_hdu_index is not None:
            hdu = hdul[lc_hdu_index]
            data = hdu.data
            cols = hdu.columns
            nrows = len(data)

            print(f"\n  {'─'*70}")
            print(f"  LIGHT CURVE DATA  (HDU #{lc_hdu_index} — '{hdu.name}')")
            print(f"  Total rows : {nrows:,}")
            print(f"  {'─'*70}")

            # Column summary table
            print(f"\n  {'Column':<22} {'Format':<10} {'Unit':<15} {'Min':>18} {'Max':>18}")
            print(f"  {'─'*22} {'─'*10} {'─'*15} {'─'*18} {'─'*18}")
            for col in cols:
                unit = col.unit if col.unit else "—"
                try:
                    arr = data[col.name]
                    if np.issubdtype(arr.dtype, np.number):
                        arr_flat = arr.flatten()
                        valid = arr_flat[np.isfinite(arr_flat)]
                        mn = f"{np.min(valid):.6g}" if len(valid) else "N/A"
                        mx = f"{np.max(valid):.6g}" if len(valid) else "N/A"
                    else:
                        mn = mx = "(non-numeric)"
                except Exception:
                    mn = mx = "error"
                print(f"  {col.name:<22} {col.format:<10} {unit:<15} {mn:>18} {mx:>18}")

            # First 20 rows
            n_show = min(20, nrows)
            print(f"\n  ── First {n_show} rows ──")
            # Header row
            header = "  " + "  ".join(f"{c.name:>18}" for c in cols)
            print(header)
            print("  " + "─" * (20 * len(cols) + 2))
            for row_idx in range(n_show):
                row_str = "  "
                for col in cols:
                    val = data[col.name][row_idx]
                    if hasattr(val, '__len__') and not isinstance(val, str):
                        cell = f"[array len={len(val)}]"
                    elif isinstance(val, float) or isinstance(val, np.floating):
                        cell = f"{val:.6g}"
                    else:
                        cell = str(val)
                    row_str += f"{cell:>18}  "
                print(row_str)
        else:
            print("\n  *** No BinTableHDU with data found — cannot show light curve records ***")


# ── FILE REGISTRY ────────────────────────────────────────────────────────────

BASE = r"c:\Users\anand\ISRO,BHA"

files = {}

# ── SoLEXS ──────────────────────────────────────────────────────────────────
# SDD1: only .gti.gz (no .lc.gz present); we inspect what we have
slx_sdd1_gti_gz = os.path.join(BASE, r"SLX\AL1_SLX_L1_20260620_v1.0\SDD1\AL1_SOLEXS_20260620_SDD1_L1.gti.gz")
slx_sdd2_lc_gz  = os.path.join(BASE, r"SLX\AL1_SLX_L1_20260620_v1.0\SDD2\AL1_SOLEXS_20260620_SDD2_L1.lc.gz")
slx_sdd2_gti_gz = os.path.join(BASE, r"SLX\AL1_SLX_L1_20260620_v1.0\SDD2\AL1_SOLEXS_20260620_SDD2_L1.gti.gz")

# ── HEL1OS ──────────────────────────────────────────────────────────────────
hls_obs1_czt1   = os.path.join(BASE, r"HLS\2026\06\20\HLS_20260620_000008_43177sec_lev1_V111\czt\lightcurve_czt1.fits")
hls_obs1_czt2   = os.path.join(BASE, r"HLS\2026\06\20\HLS_20260620_000008_43177sec_lev1_V111\czt\lightcurve_czt2.fits")
hls_obs2_czt1   = os.path.join(BASE, r"HLS\2026\06\20\HLS_20260620_121027_42563sec_lev1_V111\czt\lightcurve_czt1.fits")
hls_obs2_czt2   = os.path.join(BASE, r"HLS\2026\06\20\HLS_20260620_121027_42563sec_lev1_V111\czt\lightcurve_czt2.fits")

def open_gz_fits(gz_path, label):
    """Decompress a .gz FITS file to a temp file and inspect it."""
    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            shutil.copyfileobj(f_in, tmp)
        tmp.close()
        inspect_fits(tmp.name, label + f"  [decompressed from {os.path.basename(gz_path)}]")
    finally:
        os.unlink(tmp.name)


print("\n" + SEPARATOR)
print("  ADITYA-L1  ·  FITS File Inspector")
print("  SoLEXS (SDD1, SDD2)  &  HEL1OS (CZT1, CZT2)")
print(SEPARATOR)

# ── SoLEXS SDD1 (only GTI available) ─────────────────────────────────────
print_section("SoLEXS SDD1  — GTI file (no light-curve .lc.gz found for SDD1)")
if os.path.exists(slx_sdd1_gti_gz):
    open_gz_fits(slx_sdd1_gti_gz, "SoLEXS SDD1 GTI")
else:
    print(f"  FILE NOT FOUND: {slx_sdd1_gti_gz}")

# ── SoLEXS SDD2 GTI ───────────────────────────────────────────────────────
print_section("SoLEXS SDD2  — GTI file")
if os.path.exists(slx_sdd2_gti_gz):
    open_gz_fits(slx_sdd2_gti_gz, "SoLEXS SDD2 GTI")
else:
    print(f"  FILE NOT FOUND: {slx_sdd2_gti_gz}")

# ── SoLEXS SDD2 Light Curve ───────────────────────────────────────────────
print_section("SoLEXS SDD2  — Light Curve (.lc.gz)")
if os.path.exists(slx_sdd2_lc_gz):
    open_gz_fits(slx_sdd2_lc_gz, "SoLEXS SDD2 LC")
else:
    print(f"  FILE NOT FOUND: {slx_sdd2_lc_gz}")

# ── HEL1OS Observation 1 ─────────────────────────────────────────────────
print_section("HEL1OS  CZT1  — Observation 1  (000008)")
if os.path.exists(hls_obs1_czt1):
    inspect_fits(hls_obs1_czt1, "HEL1OS OBS-1 CZT1")
else:
    print(f"  FILE NOT FOUND: {hls_obs1_czt1}")

print_section("HEL1OS  CZT2  — Observation 1  (000008)")
if os.path.exists(hls_obs1_czt2):
    inspect_fits(hls_obs1_czt2, "HEL1OS OBS-1 CZT2")
else:
    print(f"  FILE NOT FOUND: {hls_obs1_czt2}")

# ── HEL1OS Observation 2 ─────────────────────────────────────────────────
print_section("HEL1OS  CZT1  — Observation 2  (121027)")
if os.path.exists(hls_obs2_czt1):
    inspect_fits(hls_obs2_czt1, "HEL1OS OBS-2 CZT1")
else:
    print(f"  FILE NOT FOUND: {hls_obs2_czt1}")

print_section("HEL1OS  CZT2  — Observation 2  (121027)")
if os.path.exists(hls_obs2_czt2):
    inspect_fits(hls_obs2_czt2, "HEL1OS OBS-2 CZT2")
else:
    print(f"  FILE NOT FOUND: {hls_obs2_czt2}")

print(f"\n{SEPARATOR}")
print("  Inspection complete.")
print(SEPARATOR)
