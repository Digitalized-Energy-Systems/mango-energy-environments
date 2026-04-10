"""Data structure utilities.

Equivalent to MangoEnergyEnvironments.jl/src/util/data_structure.jl.
"""

from __future__ import annotations

from typing import Any


class DotDict(dict):
    """A :class:`dict` subclass that additionally supports attribute-style access.

    Keys are accessible as attributes in addition to the standard ``[]`` syntax.
    Nested plain dicts are *not* auto-converted; wrap them explicitly if needed.

    Example::

        d = DotDict({"x": 1, "y": 2})
        d.x        # 1
        d["x"]     # 1
        d.z = 3    # same as d["z"] = 3
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'DotDict' object has no attribute '{key}'"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'DotDict' object has no attribute '{key}'"
            ) from None

    def __repr__(self) -> str:
        return f"DotDict({dict.__repr__(self)})"
