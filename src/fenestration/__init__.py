"""Pooling Windows is a python library for generating foveated pooling windows."""

__all__ = ["PoolingWindows", "pooling", "sampling", "tensors"]

from . import pooling, sampling, tensors
from .pooling_windows import PoolingWindows


def __dir__() -> list[str]:
    return __all__
