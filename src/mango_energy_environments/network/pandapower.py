"""pandapower target: a prepared :class:`Network` as pandapower net + time series, the input of
:class:`~mango_energy_environments.environments.scheduling.PowerSystemsBehavior`.

Lines are not converted: a Network carries no physical line data (impedance, length) and
PowerSystemsBehavior runs no power flow; they stay in ``Network.lines`` for ``connection_graph``.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pandas as pd

from .model import LOAD, RENEWABLE, STORAGE, THERMAL, Network

__all__ = ["to_pandapower"]


def to_pandapower(net: Network):
    """Return ``(pp_net, timeseries, refs)`` for ``PowerSystemsBehavior(pp_net, timeseries)``.

    ``refs`` maps component id → ``(kind, index)``, the id to register its agent with.
    ``timeseries`` maps a ref → hourly series as PowerSystemsBehavior replays it into
    ``max_p_mw``: renewables always, per unit of ``p_nom``; thermal units and loads in MW, only
    when ``p_max`` varies. Static table values are those of the first time step. Storage follows
    pandapower's sign convention (positive = charging), so its power bounds are flipped.
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
