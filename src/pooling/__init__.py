"""Pooling Windows is a python library for generating foveated pooling windows."""

__all__ = ["PoolingWindows", "pooling", "sampling", "utils"]

from . import pooling, sampling, utils
from .pooling_windows import PoolingWindows


def __dir__() -> list[str]:
    return __all__
