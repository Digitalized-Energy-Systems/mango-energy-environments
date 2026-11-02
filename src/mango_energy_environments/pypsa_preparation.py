"""Preparation steps for a ``pypsa.Network``, applied in memory after loading (raw files untouched).

Generic: every benchmark value (cost factor, storage hours, carriers to drop) is a parameter.
"Thermal" = generator without an hourly ``p_max_pu`` profile, as in ``from_pypsa``.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "drop_unusable_units",
    "add_quadratic_cost",
    "set_storage_defaults",
    "scale_to_targets",
    "keep_buses",
]


def drop_unusable_units(n, storage_carriers: tuple[str, ...] = ()) -> None:
    """Remove generators/storage units with zero or non-finite ``p_nom``, and storage units of
    the given carriers (e.g. ``("hydro",)`` when natural inflow is not modelled)."""
    g = n.generators
    n.remove("Generator", g.index[~np.isfinite(g.p_nom) | ~(g.p_nom > 0)])
    su = n.storage_units
    bad = su.carrier.isin(storage_carriers) | ~np.isfinite(su.p_nom) | ~(su.p_nom > 0)
    n.remove("StorageUnit", su.index[bad])


def add_quadratic_cost(n, k: float) -> None:
    """Set ``marginal_cost_quadratic`` c₂ = k·c₁/p_nom on thermal generators, 0 on renewables.

    So the marginal cost c₁ + 2·c₂·p rises from c₁ at zero output to (1 + 2k)·c₁ at p_nom.
    ``k = 0`` gives the linear cost model. Overwrites existing values (idempotent).
    """
    g = n.generators
    thermal = ~g.index.isin(n.generators_t.p_max_pu.columns)
    # zero-capacity units get c₂ = 0, not NaN
    p_nom = g.p_nom.where(g.p_nom > 0, np.inf)
    c2 = np.where(thermal, k * g.marginal_cost.clip(lower=0) / p_nom, 0.0)
    n.generators["marginal_cost_quadratic"] = c2


def set_storage_defaults(
    n, max_hours: dict[str, float], soc_initial: float | None = None
) -> None:
    """Set ``max_hours`` per storage carrier (others keep theirs) and, if given, the initial
    state of charge as a fraction of the energy capacity (p_nom · max_hours)."""
    su = n.storage_units
    n.storage_units["max_hours"] = su.carrier.map(max_hours).fillna(su.max_hours)
    if soc_initial is not None:
        su = n.storage_units
        n.storage_units["state_of_charge_initial"] = (
            soc_initial * su.p_nom * su.max_hours
        )


def scale_to_targets(n, targets: dict[str, float]) -> None:
    """Raise the total ``p_nom`` per carrier (generators and storage units) to *targets* [MW].

    The increase is spread in proportion to each unit's ``p_nom_max − p_nom`` headroom (evenly
    when there is none). Carriers already at or above their target are left alone.
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
