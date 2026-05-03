from __future__ import annotations
from models.base_adapter import BaseModelAdapter

# To add a new model: import its adapter and add one line below.
MODEL_REGISTRY: dict[str, BaseModelAdapter] = {}
