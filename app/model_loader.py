"""
GreenLeaf Grocery - Singleton Model Loader
Provides thread-safe in-memory caching of the serialized ML model artifact.
Ensures final_model.joblib is loaded exactly once into RAM at application startup,
strictly avoiding disk I/O re-loading on incoming HTTP requests.
"""

import os
import sys
import threading
import logging
from typing import Optional, Any
import pandas as pd
import joblib

# Ensure project root is available for unpickling custom classes
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import model definition classes to enable joblib deserialization
import features  # noqa: F401, E402
import train_model  # noqa: F401, E402

logger = logging.getLogger("GreenLeafModelLoader")


class ModelLoader:
    """
    Thread-safe Singleton Model Loader.
    Caches the TunedDemandForecaster in RAM for the entire lifecycle of the FastAPI process.
    """

    _instance: Optional["ModelLoader"] = None
    _lock: threading.Lock = threading.Lock()
    _model: Optional[Any] = None
    _model_path: str = os.path.join(PROJECT_ROOT, "final_model.joblib")

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_model(cls, model_path: Optional[str] = None) -> Any:
        """
        Loads model artifact from disk once and caches in class attribute _model.
        """
        if model_path:
            cls._model_path = model_path

        with cls._lock:
            if cls._model is None:
                if not os.path.exists(cls._model_path):
                    # Check relative fallback
                    fallback = "final_model.joblib"
                    if os.path.exists(fallback):
                        cls._model_path = fallback
                    else:
                        raise FileNotFoundError(
                            f"Model artifact not found at {cls._model_path}"
                        )

                logger.info("Loading model artifact into memory from: %s", cls._model_path)
                cls._model = joblib.load(cls._model_path)
                logger.info("Model artifact successfully loaded and cached as Singleton.")
        return cls._model

    @classmethod
    def get_model(cls) -> Any:
        """
        Returns the cached singleton model instance.
        If not yet loaded, loads it lazily.
        """
        if cls._model is None:
            return cls.load_model()
        return cls._model

    @classmethod
    def is_loaded(cls) -> bool:
        """Checks if model artifact is currently loaded in RAM."""
        return cls._model is not None

    @classmethod
    def predict(cls, input_data: pd.DataFrame) -> float:
        """
        Runs inference on the provided input DataFrame using cached singleton model.
        Returns prediction rounded to 2 decimal places.
        """
        model = cls.get_model()
        preds = model.predict(input_data)
        # Prediction is array of floats; return scalar
        pred_val = float(preds[0]) if len(preds) > 0 else 0.0
        return round(max(0.0, pred_val), 2)
