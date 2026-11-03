"""Networks, independent of any environment: source → ``from_<source>`` → ``prepare`` → Network.

Every environment runs from the prepared :class:`Network`, whatever source it came from.
"""

from .model import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    Component,
    Network,
    connection_graph,
)
from .prepare import add_quadratic_cost, drop_unusable_units, set_storage_defaults
from .pypsa import from_pypsa, keep_buses, scale_to_targets

__all__ = [
    "THERMAL",
    "RENEWABLE",
    "LOAD",
    "STORAGE",
    "Component",
    "Network",
    "connection_graph",
    "drop_unusable_units",
    "add_quadratic_cost",
    "set_storage_defaults",
    "from_pypsa",
    "scale_to_targets",
    "keep_buses",
]
