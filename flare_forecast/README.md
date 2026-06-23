# Aditya-L1 Solar Flare LSTM Forecasting System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: Research](https://img.shields.io/badge/license-research--only-green)]()

A research-grade **multiclass probability forecasting** system for operational solar flare prediction using time-series data from **Aditya-L1** satellite instruments (SoLEXS and HEL1OS).

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dataset Format](#dataset-format)
4. [Features](#features)
5. [Model Architecture](#model-architecture)
6. [Installation](#installation)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Operational Forecast](#operational-forecast)
10. [Outputs](#outputs)
11. [Configuration](#configuration)
12. [Future Improvements](#future-improvements)

---

## Overview

This project predicts the **probability of solar flare occurrence** within a configurable future horizon using a **bidirectional LSTM + Bahdanau Attention** neural network.

### Flare Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Quiet Sun | No significant flare activity |
| 1 | B/C Class | Minor to moderate soft X-ray enhancement |
| 2 | M Class | Moderate flare — geomagnetic storm risk |
| 3 | X Class | Extreme flare — satellite/communication disruption |

### Example Output

```
====================================================
  ADITYA-L1  SOLAR FLARE OPERATIONAL FORECAST
  Timestamp : 2026-06-20 23:59:00+00:00
====================================================

  Quiet Sun    : ████████░░░░░░░░░░░░░░░░░░░░░░   5.0%
  B/C Class    : ████████████████░░░░░░░░░░░░░░  25.0%
  M Class      : ████████████████████░░░░░░░░░░  60.0%
  X Class      : ████░░░░░░░░░░░░░░░░░░░░░░░░░░  10.0%

  Predicted class : M Class
  Risk Level      : 🔴  HIGH
====================================================
```

---

## Project Structure

```
ISRO,BHA/
└── flare_forecast/
    ├── __init__.py            Package init
    ├── config.py              All hyperparameters & paths (edit here first)
    ├── dataset.py             FITS loading → synchronized grid → labels
    ├── feature_engineering.py 22-feature computation per time bin
    ├── preprocessing.py       Clean → scale → window → SMOTE
    ├── model.py               LSTM + BahdanauAttention Keras model
    ├── trainer.py             Training loop + callbacks
    ├── evaluate.py            Metrics: Acc, F1, TSS, ROC-AUC, CM
    ├── visualize.py           8 publication-quality dark-mode plots
    ├── predict.py             OperationalForecaster class
    ├── utils.py               Logging, seeds, class weights, SHAP
    ├── main.py                CLI entry point
    └── outputs/               Generated artifacts (auto-created)
        ├── best_model.keras
        ├── scaler.pkl
        ├── training_curves.png
        ├── confusion_matrix.png
        ├── roc_curves.png
        ├── pr_curves.png
        ├── probability_distribution.png
        ├── prediction_timeline.png
        ├── shap_summary.png
        ├── operational_forecast.png
        ├── metrics.json
        └── training_history.json
```

---

## Dataset Format

### SoLEXS SDD2 (Soft X-ray, ~1-15 keV)
- **File**: `SLX/AL1_SLX_L1_20260620_v1.0/SDD2/AL1_SOLEXS_20260620_SDD2_L1.lc.gz`
- **Format**: gzip-compressed FITS, HDU `RATE`
- **Columns**: `TIME` (Unix seconds), `COUNTS` (cts/s)
- **Coverage**: 86,400 rows (1-second cadence, full day)

### HEL1OS CZT1 (Hard X-ray, 18-160 keV)
- **Files**: Two observation blocks under `HLS/2026/06/20/`
- **Format**: FITS, multiple HDUs (one per energy band)
- **Key HDU**: `CZT1_LC_BAND_18.00KEV_TO_160.00KEV`
- **Columns**: `MJD`, `CTR` (cts/s), `STAT_ERR`
- **Duty cycle**: ~17.5% (detector integration gaps)

### GOES-Proxy Label Assignment
Since the Aditya-L1 data does not include a direct GOES-class tag, labels are
assigned based on the SoLEXS count rate relative to a rolling 25th-percentile baseline:

| Class | Threshold |
|-------|-----------|
| 0 – Quiet | ratio < 2× baseline |
| 1 – B/C   | 2× ≤ ratio < 10× |
| 2 – M     | 10× ≤ ratio < 50× |
| 3 – X     | ratio ≥ 50× |

---

## Features

22 features are computed per 1-minute bin:

### SoLEXS (10 features)
`slx_count`, `slx_peak_5m`, `slx_mean_5m`, `slx_std_5m`,
`slx_mean_15m`, `slx_std_15m`, `slx_d1`, `slx_d2`,
`slx_rise_flag`, `slx_bg_ratio`

### HEL1OS (7 features)
`hls_bb`, `hls_mean_5m`, `hls_std_5m`, `hls_hardness`,
`hls_d1`, `hls_peak_5m`, `hls_bg_ratio`

### Temporal (5 features)
`hour_sin`, `hour_cos`, `prev_flare_count`,
`minutes_since_flare`, `day_norm`

---

## Model Architecture

```
Input  (batch, window=120, features=22)
  │
  ├─ LSTM(128, return_sequences=True)  + L2
  │
  ├─ Dropout(0.3)
  │
  ├─ LSTM(64, return_sequences=True)   + L2
  │
  ├─ BahdanauAttention(units=32)  ← single-head additive attention
  │
  ├─ Dense(64, relu) + L2
  │
  ├─ Dropout(0.3)
  │
  └─ Dense(4, softmax)   ← [Quiet, B/C, M, X] probabilities
```

- **Loss**: Sparse categorical cross-entropy + class weights
- **Optimizer**: Adam with learning rate scheduling
- **Attention**: Custom Bahdanau (additive) attention for timestep weighting

---

## Installation

```bash
# Clone / navigate to project
cd c:\Users\anand\ISRO,BHA

# Create virtual environment (already created as .venv)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional dependencies for full features
```bash
pip install tensorflow>=2.16  imbalanced-learn  shap
```

---

## Training

```bash
# Full pipeline with defaults (120-min window, 60-min horizon)
python -m flare_forecast.main train

# Custom window and horizon
python -m flare_forecast.main train --window 60 --horizon 30

# GRU instead of LSTM
python -m flare_forecast.main train --model gru

# Disable SMOTE oversampling
python -m flare_forecast.main train --no-smote

# More epochs, smaller batch
python -m flare_forecast.main train --epochs 200 --batch 16

# See all options
python -m flare_forecast.main train --help
```

---

## Evaluation

```bash
# Evaluate saved model on test set
python -m flare_forecast.main evaluate
```

Metrics computed:
- Accuracy, Precision, Recall, F1 (macro + per class)
- Confusion Matrix (normalized)
- ROC-AUC (one-vs-rest, macro)
- Precision-Recall AUC per class
- **True Skill Statistic (TSS)** — primary operational metric

---

## Operational Forecast

```bash
# Generate forecast from latest available FITS data
python -m flare_forecast.main forecast
```

Outputs:
- Console: probability per class + risk level
- `outputs/operational_forecast.png` — visual dashboard

### Risk Levels

| Level | P(M) + P(X) | Action |
|-------|-------------|--------|
| 🟢 LOW | < 15% | Routine monitoring |
| 🟡 MODERATE | 15–40% | Heightened awareness |
| 🔴 HIGH | 40–70% | Alert operations |
| 🚨 EXTREME | ≥ 70% | Emergency procedures |

---

## Outputs

After `train`, the `flare_forecast/outputs/` directory contains:

| File | Description |
|------|-------------|
| `best_model.keras` | Best checkpoint (lowest val_loss) |
| `scaler.pkl` | Fitted StandardScaler for inference |
| `training_curves.png` | Loss & accuracy vs epoch |
| `confusion_matrix.png` | Normalized 4×4 heatmap |
| `roc_curves.png` | Per-class ROC curves with AUC |
| `pr_curves.png` | Precision-Recall curves |
| `probability_distribution.png` | Probability histograms per class |
| `prediction_timeline.png` | Predicted vs true class over test period |
| `shap_summary.png` | Feature importance (SHAP or permutation) |
| `operational_forecast.png` | Operational probability dashboard |
| `metrics.json` | All evaluation metrics |
| `training_history.json` | Loss/accuracy per epoch |

---

## Configuration

All hyperparameters are in `config.py`. Key settings:

```python
cfg = Config()
cfg.window_size        = 120   # lookback window (minutes)
cfg.prediction_horizon = 60    # forecast horizon (minutes)
cfg.model_type         = "lstm" # "lstm" or "gru"
cfg.epochs             = 100
cfg.batch_size         = 32
cfg.learning_rate      = 1e-3
cfg.use_smote          = True
cfg.use_attention      = True
```

---

## Future Improvements

The codebase is designed for easy extension:

1. **Transformer backbone**: Add `_build_transformer_model()` in `model.py` and register in `_BUILDERS`.
2. **XGBoost ensemble**: Add an `XGBoostForecaster` class in a new `xgb_model.py`.
3. **Multi-day training**: `dataset.load_dataset()` accepts any date in `cfg.obs_date`; wrap in a loop to concatenate multiple days.
4. **Real-time streaming**: Replace `load_dataset()` in `predict.py` with a streaming FITS reader.
5. **Uncertainty quantification**: Replace Dropout with MC-Dropout (`training=True` at inference) for epistemic uncertainty estimates.
6. **GOES ground truth**: When GOES-class labels are available, replace the proxy thresholds in `config.py` with true labels.

---

## References

- Jain, R., et al. (2021). SoLEXS calibration and soft X-ray flux mapping.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015.
- Barnes, G. & Leka, K.D. (2008). Evaluating the performance of solar flare forecasting methods. *ApJ Letters*, 688(2), L107.
- Bloomfield, D.S., et al. (2012). Toward reliable benchmarking of solar flare forecasting methods. *ApJ Letters*, 747(2), L41.
