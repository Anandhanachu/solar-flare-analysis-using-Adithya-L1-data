"""
_tf_check.py
============
Single authoritative TensorFlow availability check.

Import this module before any TF usage. It raises ImportError with
clear, actionable installation instructions if TensorFlow is absent.

All other modules (model, trainer, predict, main) should check via:

    from .tf_check import require_tensorflow
    require_tensorflow()

This ensures the error message fires ONCE, cleanly, instead of
repeating through the exception chain.
"""

from __future__ import annotations

_TF_INSTALL_MSG = (
    "\n"
    + "=" * 66 + "\n"
    + "  TensorFlow is REQUIRED but is NOT installed.\n"
    + "=" * 66 + "\n"
    + "\n"
    + "  This project uses a Keras LSTM neural network.\n"
    + "  It will NOT substitute a different algorithm.\n"
    + "\n"
    + "  ROOT CAUSE\n"
    + "  ----------\n"
    + "  Your active virtual environment uses Python 3.14, which\n"
    + "  is NOT yet supported by TensorFlow (supported: 3.9-3.12).\n"
    + "  Python 3.12 is already installed on this machine.\n"
    + "\n"
    + "  FIX (one-time setup)\n"
    + "  --------------------\n"
    + "  Step 1 -- Create a Python 3.12 virtual environment:\n"
    + "      py -3.12 -m venv .venv312\n"
    + "      .venv312\\Scripts\\activate\n"
    + "\n"
    + "  Step 2 -- Install all dependencies:\n"
    + "      pip install tensorflow>=2.16\n"
    + "      pip install imbalanced-learn shap\n"
    + "      pip install -r requirements.txt\n"
    + "\n"
    + "  Step 3 -- Run the pipeline:\n"
    + "      python -m flare_forecast.main train\n"
    + "      python -m flare_forecast.main evaluate\n"
    + "      python -m flare_forecast.main forecast\n"
    + "\n"
    + "  (CPU-only alternative: pip install tensorflow-cpu>=2.16)\n"
    + "\n"
    + "=" * 66 + "\n"
)


def require_tensorflow() -> None:
    """
    Check that TensorFlow is importable.

    Raises
    ------
    ImportError
        With full installation instructions if TensorFlow is absent.
    """
    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:
        raise ImportError(_TF_INSTALL_MSG) from None
