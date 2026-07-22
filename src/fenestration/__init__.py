"""Pooling Windows is a python library for generating foveated pooling windows."""

__all__ = [
    "PoolingWindows",
    "create_pooling_windows",
    "pooling",
    "sampling",
    "_tensors",
]

from . import _tensors, pooling, sampling
from .pooling import create_pooling_windows
from .pooling_windows import PoolingWindows


def __dir__() -> list[str]:
    return __all__
