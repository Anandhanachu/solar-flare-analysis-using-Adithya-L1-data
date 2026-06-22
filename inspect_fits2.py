import sys, os, gzip, shutil, tempfile
import numpy as np
from astropy.io import fits

sys.stdout.reconfigure(encoding='utf-8')

SEP = '='*80
DASH = '-'*80

def sec(title):
    print(f'\n{SEP}')
    print(f'  {title}')
    print(SEP)

def inspect(filepath, label):
    sec(f'FILE: {label}')
    print(f'  Path : {filepath}')
    print(f'  Size : {os.path.getsize(filepath):,} bytes')

    with fits.open(filepath, memmap=True) as hdul:
        print(f'\n  Total HDUs: {len(hdul)}')
        lc_idx = None

        for i, hdu in enumerate(hdul):
            htype = type(hdu).__name__
            name  = hdu.name if hdu.name else '(unnamed)'
            print(f'\n  HDU #{i}  name={name!r}  type={htype}')
            hdr = hdu.header
            keys_to_show = [
                'TELESCOP','INSTRUME','DETECTOR','OBJECT',
                'DATE-OBS','DATE-END','TSTART','TSTOP','TIMEDEL',
                'NAXIS','NAXIS1','NAXIS2','TFIELDS','EXTNAME',
                'TIMEUNIT','TIMEZERO','MJDREFI','MJDREFF',
                'HDUCLASS','HDUCLAS1','HDUCLAS2','CHANTYPE','DETCHANS'
            ]
            for key in keys_to_show:
                if key in hdr:
                    print(f'    {key:<12} = {hdr[key]!r}')

            if isinstance(hdu, fits.BinTableHDU):
                cols = hdu.columns
                print(f'    Columns ({len(cols)}):')
                for c in cols:
                    unit_str = c.unit if c.unit else '-'
                    print(f'      {c.name:<25}  fmt={c.format:<8}  unit={unit_str}')
                nrows = len(hdu.data) if hdu.data is not None else 0
                print(f'    Rows: {nrows:,}')
                upper = [c.name.upper() for c in cols]
                if 'TIME' in upper and any(
                    k in upper for k in ['RATE','COUNTS','COUNT_RATE','FLUX','CTS']
                ):
                    lc_idx = i
                    print('    *** DETECTED: LIGHT CURVE TABLE ***')
            else:
                shape = hdu.data.shape if hdu.data is not None else 'no data'
                dtype = hdu.data.dtype  if hdu.data is not None else 'N/A'
                print(f'    Image/Primary  shape={shape}  dtype={dtype}')

        # Fallback: first non-empty BinTable
        if lc_idx is None:
            for i, hdu in enumerate(hdul):
                if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None and len(hdu.data):
                    lc_idx = i
                    break

        if lc_idx is not None:
            hdu   = hdul[lc_idx]
            data  = hdu.data
            cols  = hdu.columns
            nrows = len(data)
            print(f'\n{DASH}')
            print(f'  LIGHT CURVE  HDU#{lc_idx}  name={hdu.name!r}  total_rows={nrows:,}')
            print(DASH)

            # Column stats
            print(f'\n  {"Column":<25} {"Format":<10} {"Unit":<15} {"Min":>18} {"Max":>18}')
            print(f'  {"-"*25} {"-"*10} {"-"*15} {"-"*18} {"-"*18}')
            for c in cols:
                unit_str = c.unit if c.unit else '-'
                try:
                    arr = data[c.name]
                    flat = np.array(arr).flatten().astype(float)
                    valid = flat[np.isfinite(flat)]
                    if len(valid):
                        mn = f'{np.min(valid):.6g}'
                        mx = f'{np.max(valid):.6g}'
                    else:
                        mn = mx = 'all NaN'
                except Exception:
                    mn = mx = '(non-numeric)'
                print(f'  {c.name:<25} {c.format:<10} {unit_str:<15} {mn:>18} {mx:>18}')

            # First 20 rows
            n = min(20, nrows)
            print(f'\n  First {n} rows:')
            # Build header
            hdr_line = '  ' + ''.join(f'{c.name:>20}' for c in cols)
            print(hdr_line)
            print('  ' + '-'*20*len(cols))
            for r in range(n):
                row = '  '
                for c in cols:
                    v = data[c.name][r]
                    if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                        cell = f'[array:{len(v)}]'
                    elif isinstance(v, (float, np.floating)):
                        cell = f'{v:.6g}'
                    else:
                        cell = str(v)
                    row += f'{cell:>20}'
                print(row)
        else:
            print('\n  No BinTableHDU with data found — cannot display light curve records.')


def gz_inspect(gz_path, label):
    tmp = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
    try:
        with gzip.open(gz_path, 'rb') as fi:
            shutil.copyfileobj(fi, tmp)
        tmp.close()
        inspect(tmp.name, label)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


BASE = r'c:\Users\anand\ISRO,BHA'

slx_base = BASE + r'\SLX\AL1_SLX_L1_20260620_v1.0'
hls_base = BASE + r'\HLS\2026\06\20'

file_list = [
    ('gz', slx_base + r'\SDD1\AL1_SOLEXS_20260620_SDD1_L1.gti.gz',  'SoLEXS SDD1 — GTI (only file available for SDD1)'),
    ('gz', slx_base + r'\SDD2\AL1_SOLEXS_20260620_SDD2_L1.gti.gz',  'SoLEXS SDD2 — GTI'),
    ('gz', slx_base + r'\SDD2\AL1_SOLEXS_20260620_SDD2_L1.lc.gz',   'SoLEXS SDD2 — Light Curve (.lc)'),
    ('ft', hls_base + r'\HLS_20260620_000008_43177sec_lev1_V111\czt\lightcurve_czt1.fits', 'HEL1OS OBS-1 CZT1'),
    ('ft', hls_base + r'\HLS_20260620_000008_43177sec_lev1_V111\czt\lightcurve_czt2.fits', 'HEL1OS OBS-1 CZT2'),
    ('ft', hls_base + r'\HLS_20260620_121027_42563sec_lev1_V111\czt\lightcurve_czt1.fits',  'HEL1OS OBS-2 CZT1'),
    ('ft', hls_base + r'\HLS_20260620_121027_42563sec_lev1_V111\czt\lightcurve_czt2.fits',  'HEL1OS OBS-2 CZT2'),
]

print(f'\n{SEP}')
print('  ADITYA-L1  FITS File Inspector')
print('  SoLEXS (SDD1, SDD2)  &  HEL1OS (CZT1, CZT2)')
print(SEP)

for ftype, path, label in file_list:
    if not os.path.exists(path):
        sec(label)
        print(f'  FILE NOT FOUND: {path}')
        continue
    if ftype == 'gz':
        gz_inspect(path, label)
    else:
        inspect(path, label)

print(f'\n{SEP}')
print('  Inspection complete.')
print(SEP)
