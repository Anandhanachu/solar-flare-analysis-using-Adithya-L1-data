import sys
sys.stdout.reconfigure(encoding='utf-8')

print('Python:', sys.version)
print()

# Proper submodule import test
checks = [
    ('numpy',                   'numpy',        lambda: __import__('numpy')),
    ('pandas',                  'pandas',       lambda: __import__('pandas')),
    ('matplotlib',              'matplotlib',   lambda: __import__('matplotlib')),
    ('matplotlib.pyplot',       'matplotlib',   lambda: __import__('matplotlib.pyplot')),
    ('astropy',                 'astropy',      lambda: __import__('astropy')),
    ('astropy.io.fits',         'astropy',      lambda: __import__('astropy.io.fits')),
    ('astropy.time (Time)',      'astropy',      lambda: __import__('astropy.time', fromlist=['Time'])),
    ('scipy',                   'scipy',        lambda: __import__('scipy')),
    ('scipy.signal (find_peaks)','scipy',       lambda: __import__('scipy.signal', fromlist=['find_peaks'])),
    ('scipy.signal (correlate)', 'scipy',       lambda: __import__('scipy.signal', fromlist=['correlate'])),
    ('scipy.stats (pearsonr)',   'scipy',       lambda: __import__('scipy.stats', fromlist=['pearsonr'])),
    ('sklearn',                 'scikit-learn', lambda: __import__('sklearn')),
    ('sklearn.ensemble',        'scikit-learn', lambda: __import__('sklearn.ensemble')),
    ('streamlit',               'streamlit',    lambda: __import__('streamlit')),
    ('gzip',                    'built-in',     lambda: __import__('gzip')),
    ('shutil',                  'built-in',     lambda: __import__('shutil')),
    ('tempfile',                'built-in',     lambda: __import__('tempfile')),
    ('warnings',                'built-in',     lambda: __import__('warnings')),
]

print(f"  {'Module':<35} {'Package':<20} {'Status'}")
print(f"  {'-'*35} {'-'*20} {'-'*10}")

all_ok = True
for label, pkg, fn in checks:
    try:
        fn()
        print(f"  {label:<35} {pkg:<20} OK")
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        print(f"  {label:<35} {pkg:<20} MISSING  ({e})")
        all_ok = False

print()
if all_ok:
    print("  All required packages are installed and importable.")
else:
    print("  Some packages are MISSING - install them below.")
