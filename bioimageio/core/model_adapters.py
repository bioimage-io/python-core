"""DEPRECATED"""

from typing import List

from .backends._model_adapter import (
    DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER,
    ModelAdapter,
    create_model_adapter,
)

__all__ = [
    "ModelAdapter",
    "create_model_adapter",
    "get_weight_formats",
]


def get_weight_formats() -> List[str]:
    """
    Return list of supported weight types
    """
    return list(DEFAULT_WEIGHT_FORMAT_PRIORITY_ORDER)
