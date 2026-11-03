"""Source-independent preparation steps on a :class:`Network`, applied in memory after converting.

Generic: every benchmark value (cost factor, storage hours, carriers to drop) is a parameter.
Converters put ``carrier`` and the rated power ``p_nom`` (MW) into ``params`` of every
generator and storage unit; storage also carries ``max_hours``.
"""

from __future__ import annotations

import numpy as np

from .model import LOAD, RENEWABLE, STORAGE, THERMAL, Network

__all__ = ["drop_unusable_units", "add_quadratic_cost", "set_storage_defaults"]


def drop_unusable_units(net: Network, storage_carriers: tuple[str, ...] = ()) -> None:
    """Remove generators/storage units with zero or non-finite ``p_nom``, and storage units of
    the given carriers (e.g. ``("hydro",)`` when natural inflow is not modelled)."""
    for cid, c in list(net.components.items()):
        if c.kind == LOAD:
            continue
        p_nom = c.params["p_nom"]
        bad_carrier = c.kind == STORAGE and c.params.get("carrier") in storage_carriers
        if bad_carrier or not (np.isfinite(p_nom) and p_nom > 0):
            del net.components[cid]


def add_quadratic_cost(net: Network, k: float) -> None:
    """Set c₂ = k·c₁/p_nom on thermal units, 0 on renewables (storage keeps its cost).

    So the marginal cost c₁ + 2·c₂·p rises from c₁ at zero output to (1 + 2k)·c₁ at p_nom.
    ``k = 0`` gives the linear cost model. Overwrites existing values (idempotent).
    """
    for c in net.components.values():
        c1 = c.cost[0]
        if c.kind == THERMAL:
            p_nom = c.params["p_nom"]
            # zero-capacity units get c₂ = 0, not NaN
            c.cost = (c1, k * max(c1, 0.0) / p_nom if p_nom > 0 else 0.0)
        elif c.kind == RENEWABLE:
            c.cost = (c1, 0.0)


def set_storage_defaults(
    net: Network, max_hours: dict[str, float], soc_initial: float | None = None
) -> None:
    """Set ``max_hours`` per storage carrier (others keep theirs) and, if given, the initial
    state of charge as a fraction of the energy capacity (p_nom · max_hours)."""
    for c in net.components.values():
        if c.kind != STORAGE:
            continue
        p = c.params
        p["max_hours"] = max_hours.get(p.get("carrier"), p["max_hours"])
        if soc_initial is not None:
            p["state_of_charge_initial"] = soc_initial * p["p_nom"] * p["max_hours"]
