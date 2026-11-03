"""Save / load a prepared :class:`Network` as a NetCDF (``.nc``) file.

Schema (``format`` attribute ``mango_energy_environments.Network/1``): dims ``component``,
``time``, ``line``; ``p_min``/``p_max`` (component × time, MW, zlib-compressed); per component
``kind``, ``bus``, ``c1``, ``c2`` and ``params`` (JSON); ``line_bus0``/``line_bus1``; the caller's
``meta`` (e.g. the preparation recipe) as a JSON attribute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .model import Component, Network

__all__ = ["save_network", "load_network"]

FORMAT = "mango_energy_environments.Network/1"


def save_network(net: Network, path: str | Path, meta: dict | None = None) -> None:
    """Write *net* to *path* (NetCDF); *meta* is stored as a JSON attribute."""
    cids = list(net.components)
    comps = list(net.components.values())

    def per_component(values, dtype=None):
        return ("component", np.array(values, dtype=dtype))

    ds = xr.Dataset(
        {
            "p_min": (("component", "time"), np.array([c.p_min for c in comps])),
            "p_max": (("component", "time"), np.array([c.p_max for c in comps])),
            "kind": per_component([c.kind for c in comps], object),
            "bus": per_component([c.bus for c in comps], object),
            "c1": per_component([c.cost[0] for c in comps], float),
            "c2": per_component([c.cost[1] for c in comps], float),
            "params": per_component([json.dumps(c.params) for c in comps], object),
            "line_bus0": ("line", np.array([a for a, _ in net.lines], dtype=object)),
            "line_bus1": ("line", np.array([b for _, b in net.lines], dtype=object)),
        },
        coords={"component": np.array(cids, dtype=object), "time": net.time_index},
        attrs={"format": FORMAT, "meta": json.dumps(meta or {})},
    )
    packed = {"zlib": True, "complevel": 4}
    ds.to_netcdf(path, engine="netcdf4", encoding={"p_min": packed, "p_max": packed})


def load_network(path: str | Path) -> Network:
    """Read a :class:`Network` written by :func:`save_network`."""
    with xr.open_dataset(path, engine="netcdf4") as ds:
        if ds.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a saved Network ({FORMAT})")
        ds = ds.load()
    p_min, p_max = ds.p_min.values, ds.p_max.values
    components = {
        str(cid): Component(
            str(ds.kind.values[i]),
            str(ds.bus.values[i]),
            p_min[i].copy(),
            p_max[i].copy(),
            (float(ds.c1.values[i]), float(ds.c2.values[i])),
            json.loads(str(ds.params.values[i])),
        )
        for i, cid in enumerate(ds.component.values)
    }
    lines = [(str(a), str(b)) for a, b in zip(ds.line_bus0.values, ds.line_bus1.values)]
    return Network(pd.DatetimeIndex(ds.time.values), components, lines)
