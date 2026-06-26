#!/usr/bin/env python3

__all__ = ["__version__", "pooling_windows", "pooling", "sampling", "utils"]

from . import pooling, sampling, utils
from .pooling_windows import PoolingWindows as PoolingWindows
from .version import __version__


def __dir__() -> list[str]:
    return __all__
