"""pandapower source and target.

``from_pandapower``: a pandapower net + absolute profiles → :class:`Network`.
``to_pandapower``: a prepared :class:`Network` as pandapower net + time series, the input of
:class:`~mango_energy_environments.environments.scheduling.PowerSystemsBehavior`.

Lines are not converted: a Network carries no physical line data (impedance, length) and
PowerSystemsBehavior runs no power flow; they stay in ``Network.lines`` for ``connection_graph``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandapower as pp
import pandas as pd

from .model import LOAD, RENEWABLE, STORAGE, THERMAL, Component, Network

logger = logging.getLogger(__name__)

__all__ = ["from_pandapower", "to_pandapower"]

# element tables from_pandapower does not map (warned when non-empty)
_PP_IGNORED = {
    *("ext_grid", "storage", "trafo3w", "impedance", "dcline", "ward", "xward"),
    *("shunt", "asymmetric_load", "asymmetric_sgen", "motor"),
}


def from_pandapower(pp_net, profiles: dict[str, pd.DataFrame]) -> Network:
    """Convert a pandapower net into a :class:`Network` (full horizon; cut by the caller).

    *profiles*: absolute active power per element table (``"load"``, ``"sgen"``), MW, time ×
    element index; elements without a column keep their static ``p_mw`` × ``scaling``. Time index = the
    profiles' index. Only in-service elements; component id = element name (must be unique),
    bus id = pandapower bus index as str.

    - gen → thermal, 0 … ``max_p_mw`` (``min_p_mw`` and gen profiles ignored: no unit
      commitment); sgen → renewable, 0 … profile; load → demand profile.
    - cost (c₁, c₂) from ``poly_cost`` (``cp1_eur_per_mw``, ``cp2_eur_per_mw2``), else 0.
    - connections: lines, trafos and closed bus–bus switches. Buses without components are
      dropped with their connections (as in ``from_pypsa``).
    """
    for table in sorted(_PP_IGNORED):
        if table in pp_net and len(pp_net[table]):
            logger.warning("from_pandapower: %d %s ignored", len(pp_net[table]), table)
    if any(df.isna().any().any() for df in profiles.values()):
        # a NaN size would make prepare drop the unit silently
        raise ValueError("from_pandapower: profiles contain NaN")
    time_index = next(iter(profiles.values())).index
    cost = {
        (r.et, r.element): (float(r.cp1_eur_per_mw), float(r.cp2_eur_per_mw2))
        for r in pp_net.poly_cost.itertuples()
    }
    components: dict[str, Component] = {}

    def add(table: str, idx, kind: str, p_min, p_max, params=None) -> None:
        name = pp_net[table].at[idx, "name"]
        if name in components:
            raise ValueError(f"component name {name!r} used twice")
        bus = str(pp_net[table].at[idx, "bus"])
        c = cost.get((table, idx), (0.0, 0.0))
        components[name] = Component(kind, bus, p_min, p_max, c, params or {})

    def series(table: str, idx) -> np.ndarray:
        prof = profiles.get(table)
        if prof is not None and idx in prof.columns:
            return prof[idx].to_numpy(dtype=float)
        row = pp_net[table].loc[idx]
        return np.full(len(time_index), float(row.p_mw * row.scaling))

    zero = np.zeros(len(time_index))
    for idx, g in pp_net.gen[pp_net.gen.in_service].iterrows():
        p_max = np.full(len(time_index), float(g.max_p_mw))
        add(
            "gen",
            idx,
            THERMAL,
            zero,
            p_max,
            {"carrier": g.get("type"), "p_nom": float(g.max_p_mw)},
        )
    for idx, g in pp_net.sgen[pp_net.sgen.in_service].iterrows():
        p_max = series("sgen", idx)
        add(
            "sgen",
            idx,
            RENEWABLE,
            zero,
            p_max,
            {"carrier": g.get("type"), "p_nom": float(p_max.max())},
        )
    for idx in pp_net.load.index[pp_net.load.in_service]:
        demand = series("load", idx)
        add("load", idx, LOAD, demand, demand)

    used = {c.bus for c in components.values()}
    empty = sorted(set(map(str, pp_net.bus.index)) - used, key=int)
    if empty:
        logger.warning(
            "from_pandapower: %d buses without components dropped: %s",
            len(empty),
            empty,
        )
    sw = pp_net.switch[(pp_net.switch.et == "b") & pp_net.switch.closed]
    pairs = [
        *zip(
            pp_net.line.from_bus[pp_net.line.in_service],
            pp_net.line.to_bus[pp_net.line.in_service],
        ),
        *zip(
            pp_net.trafo.hv_bus[pp_net.trafo.in_service],
            pp_net.trafo.lv_bus[pp_net.trafo.in_service],
        ),
        *zip(sw.bus, sw.element),
    ]
    lines = [
        (str(a), str(b))
        for a, b in pairs
        if a != b and str(a) in used and str(b) in used
    ]
    return Network(time_index, components, lines)


def to_pandapower(net: Network):
    """Return ``(pp_net, timeseries, refs)`` for ``PowerSystemsBehavior(pp_net, timeseries)``.

    ``refs`` maps component id → ``(kind, index)``, the id to register its agent with.
    ``timeseries`` maps a ref → hourly series as PowerSystemsBehavior replays it into
    ``max_p_mw``: renewables always, per unit of ``p_nom``; thermal units and loads in MW, only
    when ``p_max`` varies. Static table values are those of the first time step. Storage follows
    pandapower's sign convention (positive = charging), so its power bounds are flipped.
    Loads: the current demand is the ``max_active_power`` observation (the replay updates
    ``max_p_mw``); ``p_mw`` / ``active_power`` keeps the first hour's demand.
    """
    pp_net = pp.create_empty_network()
    buses = dict.fromkeys(c.bus for c in net.components.values())
    # shortcut: Network has no voltage levels; 1 kV placeholder until a power flow is needed.
    bus = {b: pp.create_bus(pp_net, vn_kv=1.0, name=b) for b in buses}
    timeseries: dict[tuple[str, int], pd.Series] = {}
    refs: dict[str, tuple[str, int]] = {}
    for cid, c in net.components.items():
        b, p_min, p_max = bus[c.bus], float(c.p_min[0]), float(c.p_max[0])
        if c.kind == THERMAL:
            idx = pp.create_gen(
                pp_net, b, 0.0, min_p_mw=p_min, max_p_mw=p_max, name=cid
            )
        elif c.kind == RENEWABLE:
            p_nom = c.params["p_nom"]  # the replay scales per-unit values by max_p_mw
            idx = pp.create_sgen(
                pp_net, b, 0.0, min_p_mw=p_min, max_p_mw=p_nom, name=cid
            )
        elif c.kind == LOAD:
            idx = pp.create_load(
                pp_net, b, p_max, min_p_mw=p_min, max_p_mw=p_max, name=cid
            )
        elif c.kind == STORAGE:
            p = c.params
            e_max = p["p_nom"] * p["max_hours"]
            soc = 100 * p["state_of_charge_initial"] / e_max if e_max else 0.0
            idx = pp.create_storage(
                pp_net,
                b,
                0.0,
                e_max,
                soc_percent=soc,
                min_p_mw=-p_max,
                max_p_mw=-p_min,
                name=cid,
                efficiency_store=p["efficiency_store"],
                efficiency_dispatch=p["efficiency_dispatch"],
            )
        else:
            raise ValueError(f"unknown component kind {c.kind!r} ({cid})")
        if c.kind != LOAD:
            pp.create_poly_cost(
                pp_net, idx, c.kind, cp1_eur_per_mw=c.cost[0], cp2_eur_per_mw2=c.cost[1]
            )
        ref = refs[cid] = (c.kind, int(idx))
        if c.kind == RENEWABLE:
            timeseries[ref] = pd.Series(
                c.p_max / c.params["p_nom"], index=net.time_index
            )
        # shortcut: time-varying storage bounds are not replayed; add when a network has them.
        elif c.kind != STORAGE and np.ptp(c.p_max) > 0:
            timeseries[ref] = pd.Series(c.p_max, index=net.time_index)
    return pp_net, timeseries, refs
