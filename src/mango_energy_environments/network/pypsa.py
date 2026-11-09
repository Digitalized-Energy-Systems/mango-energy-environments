"""PyPSA source: convert a ``pypsa.Network`` into a :class:`Network`.

Also two PyPSA-only steps for building dataset files (they need PyPSA data such as
``p_nom_max`` and write back a PyPSA network): ``scale_to_targets``, ``keep_buses``.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .model import LOAD, RENEWABLE, STORAGE, THERMAL, Component, Network, bridge_lines

logger = logging.getLogger(__name__)

__all__ = ["from_pypsa", "scale_to_targets", "keep_buses"]

# PyPSA component types the converter maps (or that only hold catalogue data).
_PYPSA_MAPPED = {
    *("Bus", "Carrier", "Generator", "Load", "StorageUnit", "Line", "Link"),
    *("LineType", "TransformerType", "SubNetwork", "Shape"),
}


def from_pypsa(n) -> Network:
    """Convert a ``pypsa.Network`` into a :class:`Network` (full horizon; cut by the caller).

    Renewable = generator with an hourly ``p_max_pu`` profile, else thermal. Lines and links both
    count as connections. Storage data goes into ``params`` as-is. Buses without components are
    dropped (a warning names them); lines through them are bridged (``bridge_lines``).
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
                params={"carrier": g.carrier, "p_nom": float(g.p_nom)},
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
    lines = bridge_lines(
        [*zip(n.lines.bus0, n.lines.bus1), *zip(n.links.bus0, n.links.bus1)], used
    )
    return Network(n.snapshots, components, lines)


def scale_to_targets(n, targets: dict[str, float]) -> None:
    """Raise the total ``p_nom`` per carrier (generators and storage units) to *targets* [MW].

    The increase is spread in proportion to each unit's ``p_nom_max − p_nom`` headroom (evenly
    when the total headroom is zero or infinite). Carriers already at or above their target are
    left alone. Generators and storage units are scaled separately, so a carrier must belong to
    only one of them (else both are raised to the target).
    """
    for carrier, target in targets.items():
        for component, df in (
            ("Generator", n.generators),
            ("StorageUnit", n.storage_units),
        ):
            sel = df.index[df.carrier == carrier]
            current = df.loc[sel, "p_nom"].sum()
            if len(sel) == 0 or current >= target:
                continue
            headroom = (df.loc[sel, "p_nom_max"] - df.loc[sel, "p_nom"]).clip(lower=0)
            total = headroom.sum()
            weights = headroom / total if 0 < total < np.inf else 1.0 / len(sel)
            n.components[component].static.loc[sel, "p_nom"] += (
                target - current
            ) * weights


def keep_buses(n, buses) -> None:
    """Keep only *buses*: drop every other bus, the elements attached to it and every branch
    with an end outside."""
    drop = n.buses.index.difference(buses)
    for c in n.components:
        if c.name == "Bus" or c.static.empty:
            continue
        cols = [col for col in c.static.columns if re.fullmatch(r"bus\d*", col)]
        if cols:
            n.remove(c.name, c.static.index[c.static[cols].isin(drop).any(axis=1)])
    n.remove("Bus", drop)
