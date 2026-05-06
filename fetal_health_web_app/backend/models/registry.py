from __future__ import annotations
from models.base_adapter import BaseModelAdapter
from models.binarycnn_adapter import BinaryCNNAdapter
from models.minirocket_adapter import MiniRocketAdapter
from models.xgboost_adapter import XGBoostAdapter
from models.ensemble_adapter import EnsembleAdapter

# To add a new model: import its adapter and add one line below.
MODEL_REGISTRY: dict[str, BaseModelAdapter] = {
    "binarycnn": BinaryCNNAdapter(),
    "minirocket": MiniRocketAdapter(),
    "xgboost": XGBoostAdapter(),
    "ensemble": EnsembleAdapter(),
}
