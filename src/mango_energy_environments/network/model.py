"""The prepared network every environment runs from, whatever source it was converted from.

Sources (PyPSA, later pandapower, ...) are converted into a :class:`Network`
(``from_<source>``), then prepared with the source-independent steps in ``prepare``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "THERMAL",
    "RENEWABLE",
    "LOAD",
    "STORAGE",
    "Component",
    "Network",
    "connection_graph",
]

# Component kinds; the values double as pandapower table names (PowerSystemsBehavior uses them).
#: Synchronous / thermal generator.
THERMAL = "gen"
#: Static / renewable generator (wind, solar, run-of-river).
RENEWABLE = "sgen"
#: Demand / load.
LOAD = "load"
#: Energy storage / battery.
STORAGE = "storage"


@dataclass
class Component:
    """One component. Power in MW per time step; positive = injection (load: consumption)."""

    kind: str  # THERMAL, RENEWABLE, LOAD or STORAGE
    bus: str
    p_min: np.ndarray
    p_max: np.ndarray  # for a load: its demand
    cost: tuple[float, float] = (0.0, 0.0)  # c1 €/MWh, c2 €/MW²h
    # extra data, e.g. carrier, p_nom, storage energy and efficiency
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Network:
    time_index: pd.DatetimeIndex
    components: dict[str, Component]
    lines: list[tuple[str, str]]  # bus pairs joined by an AC line or link


def bridge_lines(pairs, used) -> list[tuple[str, str]]:
    """Lines among the *used* buses (those with components). A path through unused buses (dropped
    by the converters) becomes direct lines between the used buses at its ends, so dropping a
    transit substation does not split the grid."""
    graph = nx.Graph(pairs)
    lines = [(a, b) for a, b in pairs if a in used and b in used]
    bridged = set()
    unused = graph.subgraph(n for n in graph if n not in used)
    for part in nx.connected_components(unused):
        ends = sorted({m for n in part for m in graph[n] if m in used})
        bridged.update(combinations(ends, 2))
    return lines + sorted(bridged)


def connection_graph(network: Network) -> nx.Graph:
    """Component graph: same bus → connected; buses joined by a line → all-to-all."""
    by_bus: dict[str, list[str]] = defaultdict(list)
    for cid, c in network.components.items():
        by_bus[c.bus].append(cid)
    graph = nx.Graph()
    graph.add_nodes_from(network.components)
    for cids in by_bus.values():
        graph.add_edges_from(combinations(cids, 2))
    for a, b in network.lines:
        graph.add_edges_from(product(by_bus[a], by_bus[b]))
    return graph
