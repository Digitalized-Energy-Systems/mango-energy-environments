"""Copper-plate backend: energy moves freely between all components, no physics.

Lines only decide which components (agents) are connected; the balance is one pool per time
step. Agents see their whole horizon at once and set a whole-horizon schedule.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from mango.simulation.environment import Behavior

from .scheduling.power_systems_scheduling import LOAD, RENEWABLE, STORAGE, THERMAL

logger = logging.getLogger(__name__)

__all__ = [
    "Component",
    "CopperPlateNetwork",
    "CopperPlateBehavior",
    "connection_graph",
    "from_pypsa",
]


@dataclass
class Component:
    """One component. Power in MW per time step; positive = injection (load: consumption)."""

    kind: str  # THERMAL, RENEWABLE, LOAD or STORAGE
    bus: str
    p_min: np.ndarray
    p_max: np.ndarray  # for a load: its demand
    cost: tuple[float, float] = (0.0, 0.0)  # c1 €/MWh, c2 €/MW²h
    # extra data, e.g. storage energy and efficiency
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CopperPlateNetwork:
    time_index: pd.DatetimeIndex
    components: dict[str, Component]
    lines: list[tuple[str, str]]  # bus pairs joined by an AC line or link


def connection_graph(network: CopperPlateNetwork) -> nx.Graph:
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


# PyPSA component types the converter maps (or that only hold catalogue data).
_PYPSA_MAPPED = {
    *("Bus", "Carrier", "Generator", "Load", "StorageUnit", "Line", "Link"),
    *("LineType", "TransformerType", "SubNetwork", "Shape"),
}


def from_pypsa(n) -> CopperPlateNetwork:
    """Convert a ``pypsa.Network`` into a copper-plate network (full horizon; cut by the caller).

    Renewable = generator with an hourly ``p_max_pu`` profile, else thermal. Lines and links both
    count as connections. Storage data goes into ``params`` as-is. Buses without components are
    dropped with their lines (a warning names them), so they do not connect their neighbours.
    """
    # shortcut: static marginal cost only and single-period snapshots; extend when a network needs it.
    for c in n.components:
        if len(c.static) and c.name not in _PYPSA_MAPPED:
            logger.warning("from_pypsa: %d %s ignored", len(c.static), c.name)

    components: dict[str, Component] = {}

    def add(name: str, component: Component) -> None:
        if name in components:
            raise ValueError(f"component name {name!r} used twice")
        components[name] = component

    profiled = set(n.generators_t.p_max_pu.columns)
    g_max = n.get_switchable_as_dense("Generator", "p_max_pu")
    g_min = n.get_switchable_as_dense("Generator", "p_min_pu")
    for name, g in n.generators.iterrows():
        add(
            name,
            Component(
                RENEWABLE if name in profiled else THERMAL,
                g.bus,
                (g_min[name] * g.p_nom).to_numpy(),
                (g_max[name] * g.p_nom).to_numpy(),
                cost=(float(g.marginal_cost), float(g.marginal_cost_quadratic)),
                params={"carrier": g.carrier},
            ),
        )

    p_set = n.get_switchable_as_dense("Load", "p_set")
    for name, load in n.loads.iterrows():
        demand = p_set[name].to_numpy()
        add(name, Component(LOAD, load.bus, demand, demand))

    s_max = n.get_switchable_as_dense("StorageUnit", "p_max_pu")
    s_min = n.get_switchable_as_dense("StorageUnit", "p_min_pu")
    for name, su in n.storage_units.iterrows():
        add(
            name,
            Component(
                STORAGE,
                su.bus,
                (s_min[name] * su.p_nom).to_numpy(),
                (s_max[name] * su.p_nom).to_numpy(),
                cost=(float(su.marginal_cost), 0.0),
                params={
                    "carrier": su.carrier,
                    "max_hours": float(su.max_hours),
                    "efficiency_store": float(su.efficiency_store),
                    "efficiency_dispatch": float(su.efficiency_dispatch),
                    "state_of_charge_initial": float(su.state_of_charge_initial),
                    "p_nom": float(su.p_nom),
                },
            ),
        )

    used = {c.bus for c in components.values()}
    empty = sorted(set(n.buses.index) - used)
    if empty:
        logger.warning(
            "from_pypsa: %d buses without components dropped: %s", len(empty), empty
        )
    lines = [
        (a, b)
        for a, b in [*zip(n.lines.bus0, n.lines.bus1), *zip(n.links.bus0, n.links.bus1)]
        if a in used and b in used
    ]
    return CopperPlateNetwork(n.snapshots, components, lines)


class CopperPlateBehavior(Behavior):
    """mango behaviour: agents observe horizon arrays and set schedules; pool balance."""

    def __init__(self, network: CopperPlateNetwork) -> None:
        self.network = network
        self.schedules: dict[str, np.ndarray] = {}  # component id -> MW per step
        self._component_of: dict[str, str] = {}  # agent id -> component id

    def install(self, agent, **kwargs) -> None:
        # mango's world.register calls install for every agent; only bind those given an id.
        if "id" not in kwargs:
            return
        cid = kwargs["id"]
        if cid not in self.network.components:
            raise KeyError(f"unknown component {cid!r}")
        self._component_of[agent.aid] = cid

    def observe(self, agent_id: str, name: str) -> Any:
        c = self.network.components[self._component_of[agent_id]]
        if name == "statics":
            return {"kind": c.kind, "bus": c.bus, "cost": c.cost, **c.params}
        if name == "p_min":
            return c.p_min
        if name in ("p_max", "demand"):
            return c.p_max
        raise KeyError(f"no observation {name!r}")

    def act(self, agent_id: str, action: str, schedule) -> None:
        if action != "set_schedule":
            raise KeyError(f"no action {action!r}")
        schedule = np.asarray(schedule, dtype=float)
        if schedule.shape != (len(self.network.time_index),):
            raise ValueError(
                f"schedule shape {schedule.shape}, expected one value per step"
            )
        self.schedules[self._component_of[agent_id]] = schedule

    def balance(self) -> np.ndarray:
        """Residual per step: injections − demand (MW). Loads always draw their demand."""
        residual = np.zeros(len(self.network.time_index))
        for cid, c in self.network.components.items():
            if c.kind == LOAD:
                residual -= c.p_max
            elif cid in self.schedules:
                residual += self.schedules[cid]
        return residual
