"""Compatibility helpers for loading checkpoints across NumPy 1.x / 2.x."""

from __future__ import annotations

import sys


def ensure_numpy_core_alias() -> None:
    """
    Torch checkpoints saved under NumPy 2 reference `numpy._core`.
    NumPy 1.25 (required by flappy-bird-env) only has `numpy.core`.
    """
    try:
        import numpy._core  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import numpy.core as core

    sys.modules.setdefault("numpy._core", core)
    for name in ("multiarray", "umath", "numerictypes", "_multiarray_umath"):
        mod = getattr(core, name, None)
        if mod is not None:
            sys.modules.setdefault(f"numpy._core.{name}", mod)
