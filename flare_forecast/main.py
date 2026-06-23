"""
main.py
=======
Command-line entry point for the Aditya-L1 Solar Flare LSTM Forecasting System.

Sub-commands
------------
train    — Run the full training pipeline (data → features → preprocess → train → evaluate → plot)
evaluate — Load saved model and re-evaluate on test set
forecast — Operational forecast using the latest available FITS data

Examples
--------
    python -m flare_forecast.main train
    python -m flare_forecast.main evaluate
    python -m flare_forecast.main forecast
    python -m flare_forecast.main train --window 60 --epochs 50
    python -m flare_forecast.main train --model gru
"""

from __future__ import annotations

import argparse
import sys
import os
import time

# ── Force UTF-8 on Windows ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .config import Config
from .utils import set_seed, get_logger, load_json
from ._tf_check import require_tensorflow
require_tensorflow()   # prints actionable error and stops if TF is absent

_LOG = get_logger("main", log_file=None)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Parser
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "flare_forecast",
        description = "Aditya-L1 Solar Flare LSTM Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m flare_forecast.main train
  python -m flare_forecast.main train --window 60 --horizon 30 --epochs 80
  python -m flare_forecast.main train --model gru --no-smote
  python -m flare_forecast.main evaluate
  python -m flare_forecast.main forecast
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Run full training pipeline")
    p_train.add_argument("--window",   type=int,   default=None,
                         help="Lookback window size in minutes (default 120)")
    p_train.add_argument("--horizon",  type=int,   default=None,
                         help="Prediction horizon in minutes (default 60)")
    p_train.add_argument("--epochs",   type=int,   default=None,
                         help="Maximum training epochs (default 100)")
    p_train.add_argument("--batch",    type=int,   default=None,
                         help="Batch size (default 32)")
    p_train.add_argument("--lr",       type=float, default=None,
                         help="Initial learning rate (default 0.001)")
    p_train.add_argument("--model",    type=str,   default=None,
                         choices=["lstm", "gru"],
                         help="Model architecture (default lstm)")
    p_train.add_argument("--no-attention", action="store_true",
                         help="Disable Bahdanau attention layer")
    p_train.add_argument("--seed",     type=int,   default=None,
                         help="Random seed (default 42)")
    p_train.add_argument("--tune-batch", action="store_true",
                         help="Enable batch size auto-tuning (8, 16, 32)")

    # ── evaluate ────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate",
                             help="Evaluate saved model on test set")
    p_eval.add_argument("--window",  type=int, default=None)
    p_eval.add_argument("--horizon", type=int, default=None)

    # ── forecast ────────────────────────────────────────────────────────────
    p_fore = sub.add_parser("forecast",
                             help="Operational forecast from latest FITS data")

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Stages
# ══════════════════════════════════════════════════════════════════════════════

def _load_data_and_features(cfg: Config):
    """Shared: load FITS → features → cleaning."""
    from .dataset          import load_dataset
    from .feature_engineering import build_features
    from .preprocessing    import clean_features

    _LOG.info("=" * 60)
    _LOG.info("  STAGE 1: Loading FITS data")
    _LOG.info("=" * 60)
    raw_df = load_dataset(cfg)

    _LOG.info("=" * 60)
    _LOG.info("  STAGE 2: Feature engineering")
    _LOG.info("=" * 60)
    feat_df = build_features(raw_df, cfg)
    feat_df = clean_features(feat_df)
    return feat_df


def run_train(cfg: Config) -> None:
    """Full training pipeline."""
    from .preprocessing import preprocess
    from .model         import build_model
    from .trainer       import train_model
    from .evaluate      import evaluate_model
    from .visualize     import (plot_training_curves, plot_confusion_matrix,
                                plot_roc_curves, plot_pr_curves,
                                plot_probability_distribution,
                                plot_prediction_timeline, plot_feature_importance)
    from .utils         import load_json

    t0 = time.time()

    # 1-2. Data + Features
    feat_df = _load_data_and_features(cfg)

    # 3. Preprocessing
    _LOG.info("=" * 60)
    _LOG.info("  STAGE 3: Preprocessing & windowing")
    _LOG.info("=" * 60)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     scaler) = preprocess(feat_df, cfg)

    # 4. Batch Size Tuning
    import numpy as np
    if cfg.auto_tune_batch_size:
        _LOG.info("=" * 60)
        _LOG.info("  STAGE 4A: Batch Size Auto-Tuning")
        _LOG.info("=" * 60)
        best_batch = cfg.batch_size
        best_f1 = -1.0
        
        for cand_batch in cfg.candidate_batch_sizes:
            _LOG.info(f"Testing batch_size = {cand_batch} for 3 epochs...")
            tmp_cfg = Config()
            for k, v in cfg.__dict__.items():
                setattr(tmp_cfg, k, v)
            tmp_cfg.batch_size = cand_batch
            tmp_cfg.epochs = 3
            
            tmp_model = build_model(tmp_cfg)
            tmp_history = train_model(tmp_model, X_train, y_train, X_val, y_val, tmp_cfg)
            
            # evaluate validation F1
            val_preds = tmp_model.predict(X_val, batch_size=cand_batch, verbose=0)
            from sklearn.metrics import f1_score
            val_f1 = f1_score(y_val, np.argmax(val_preds, axis=1), average='macro')
            _LOG.info(f"  --> Batch {cand_batch} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_batch = cand_batch
                
        cfg.batch_size = best_batch
        _LOG.info(f"Auto-tuning selected batch_size = {cfg.batch_size}")

    # 4B. Model
    _LOG.info("=" * 60)
    _LOG.info("  STAGE 4B: Building model")
    _LOG.info("=" * 60)
    model = build_model(cfg)

    # 5. Train
    _LOG.info("=" * 60)
    _LOG.info("  STAGE 5: Training")
    _LOG.info("=" * 60)
    history = train_model(model, X_train, y_train, X_val, y_val, cfg)

    # 5B. Calibration
    _LOG.info("=" * 60)
    _LOG.info("  STAGE 5B: Probability Calibration")
    _LOG.info("=" * 60)
    try:
        from .calibration import run_calibration
        run_calibration(model, X_val, y_val, cfg)
    except Exception as exc:
        _LOG.warning(f"Calibration failed: {exc}")

    # 6. Evaluate
    _LOG.info("=" * 60)
    _LOG.info("  STAGE 6: Evaluation")
    _LOG.info("=" * 60)
    # Attach true labels for plotting
    results = evaluate_model(model, X_test, y_test, cfg)
    results["y_pred_true"] = y_test.tolist()

    import numpy as np
    y_true  = np.array(results["y_pred_true"])
    y_pred  = np.array(results["y_pred"])
    proba   = np.array(results["proba"])

    # 7. Visualize
    _LOG.info("=" * 60)
    _LOG.info("  STAGE 7: Visualization")
    _LOG.info("=" * 60)
    plot_training_curves(history.history, cfg)
    plot_confusion_matrix(results["confusion_matrix_norm"], cfg)

    if y_true.shape[0] > 0:
        plot_roc_curves(y_true, proba, cfg)
        plot_pr_curves(y_true, proba, cfg)
        plot_prediction_timeline(y_true, y_pred, proba, cfg)

    plot_probability_distribution(proba, cfg)

    # 8. Integrated Gradients / Feature importance
    _LOG.info("=" * 60)
    _LOG.info("  STAGE 8: Feature importance (Integrated Gradients)")
    _LOG.info("=" * 60)
    try:
        from .explain import generate_ig_explanations
        generate_ig_explanations(model, X_train, X_test, cfg)
    except Exception as exc:
        _LOG.warning(f"Feature importance failed: {exc}")

    elapsed = time.time() - t0
    _LOG.info("=" * 60)
    _LOG.info(f"  Training pipeline complete in {elapsed:.1f}s")
    _LOG.info(f"  Accuracy : {results['accuracy']:.4f}")
    _LOG.info(f"  TSS      : {results['tss_macro']:.4f}")
    _LOG.info(f"  ROC-AUC  : {results['roc_auc_macro']:.4f}")
    _LOG.info(f"  Outputs  : {cfg.output_dir}")
    _LOG.info("=" * 60)

    # Print classification report
    print("\n" + results["classification_report"])


def run_evaluate(cfg: Config) -> None:
    """Load saved model and evaluate on test set."""
    from .preprocessing import preprocess
    from .evaluate      import evaluate_model
    from tensorflow import keras
    from .model import BahdanauAttention
    import numpy as np

    feat_df = _load_data_and_features(cfg)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     scaler) = preprocess(feat_df, cfg)

    model_path = os.path.join(cfg.output_dir, "best_model.keras")
    if not os.path.exists(model_path):
        _LOG.error(f"No saved model at {model_path}. Run 'train' first.")
        sys.exit(1)

    model = keras.models.load_model(
        model_path,
        custom_objects={"BahdanauAttention": BahdanauAttention}
    )
    results = evaluate_model(model, X_test, y_test, cfg)
    print("\n" + results["classification_report"])
    _LOG.info(f"Accuracy={results['accuracy']:.4f}  "
              f"TSS={results['tss_macro']:.4f}  "
              f"ROC-AUC={results['roc_auc_macro']:.4f}")


def run_forecast(cfg: Config) -> None:
    """Operational forecast from latest FITS data."""
    from .predict import OperationalForecaster
    forecaster = OperationalForecaster(cfg)
    result     = forecaster.forecast_latest()
    forecaster.print_forecast(result)


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Build config
    cfg = Config()

    # Apply CLI overrides
    if hasattr(args, "window")  and args.window  is not None:
        cfg.window_size        = args.window
    if hasattr(args, "horizon") and args.horizon is not None:
        cfg.prediction_horizon = args.horizon
    if hasattr(args, "epochs")  and args.epochs  is not None:
        cfg.epochs             = args.epochs
    if hasattr(args, "batch")   and args.batch   is not None:
        cfg.batch_size         = args.batch
    if hasattr(args, "lr")      and args.lr      is not None:
        cfg.learning_rate      = args.lr
    if hasattr(args, "model")   and args.model   is not None:
        cfg.model_type         = args.model
    if hasattr(args, "no_attention") and args.no_attention:
        cfg.use_attention      = False
    if hasattr(args, "seed")    and args.seed    is not None:
        cfg.random_seed        = args.seed
    if hasattr(args, "tune_batch") and args.tune_batch:
        cfg.auto_tune_batch_size = True

    # Seed all RNGs
    set_seed(cfg.random_seed)

    _LOG.info(f"Command: {args.command}")
    _LOG.info(f"Config  model={cfg.model_type}  window={cfg.window_size}  "
              f"horizon={cfg.prediction_horizon}  epochs={cfg.epochs}")

    if args.command == "train":
        run_train(cfg)
    elif args.command == "evaluate":
        run_evaluate(cfg)
    elif args.command == "forecast":
        run_forecast(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
