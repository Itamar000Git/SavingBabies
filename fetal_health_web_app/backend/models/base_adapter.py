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
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Clean and normalize the CTG signal.
        Stores raw signal stats on self for use by explain().
        Returns normalized array ready for inference.
        """

    @abstractmethod
    def predict(self, processed_data: np.ndarray) -> dict:
        """
        Run inference on preprocessed data.
        Returns {"label": str, "confidence": float | None}
        """

    @abstractmethod
    def explain(self, processed_data: np.ndarray, prediction: dict) -> dict:
        """
        Produce a human-readable explanation.
        Returns {"important_parameters": list[dict], "summary": str}
        Each parameter dict: {"name": str, "value": float|str, "impact": "normal"|"elevated"|"critical"}
        """
