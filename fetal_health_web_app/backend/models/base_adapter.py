from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModelAdapter(ABC):
    name: str  # human-readable display name, set in each subclass

    @abstractmethod
    def load_model(self) -> None:
        """Load weights from disk. Called once at server startup."""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, **context) -> np.ndarray:
        """
        Clean and normalize the CTG signal.
        **context may carry baby= and mother= objects for models that use metadata.
        Returns array ready for inference.
        """

    @abstractmethod
    def predict(self, processed_data: np.ndarray) -> dict:
        """
        Run inference on preprocessed data.
        Returns {"label": str, "confidence": float | None}
        """

    @abstractmethod
    def explain(self, processed_data: np.ndarray, prediction: dict, signal_features: dict) -> dict:
        """Returns {"important_parameters": [...], "summary": str}"""
