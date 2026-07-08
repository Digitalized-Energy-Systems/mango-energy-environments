"""PyPSA-backed mango :class:`~mango.simulation.environment.Behavior`.

Plays the same role as
:class:`~mango_energy_environments.environments.scheduling.power_systems_scheduling.PowerSystemsBehavior`
but uses a PyPSA ``Network`` as the underlying power-system data model.

PyPSA stores components in pandas DataFrames on the ``Network`` object:

* ``n.generators`` — dispatchable and non-dispatchable generators
* ``n.loads`` — power demands
* ``n.storage_units`` — batteries / storage
* ``n.buses``, ``n.lines``, ``n.snapshots`` — topology & time

This behavior classifies generators into :data:`THERMAL` and :data:`RENEWABLE`
via their ``carrier`` attribute.  Carriers can be customised by passing
``renewable_carriers`` to the constructor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

import pandas as pd
from mango.simulation.environment import Environment

from ._scheduling_behavior_base import SchedulingBehaviorBase
from .base import (
    DEFAULT_RENEWABLE_CARRIERS,
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    ComponentRef,
    PowerUpdateInfo,
)

__all__ = [
    "PyPSABehavior",
    "extract_timeseries",
]

# PyPSA column names
_COL_P_NOM = "p_nom"
_COL_P_SET = "p_set"
_COL_P_MAX_PU = "p_max_pu"
_COL_P_MIN_PU = "p_min_pu"
_COL_MARGINAL_COST = "marginal_cost"
_COL_CARRIER = "carrier"


class PyPSABehavior(SchedulingBehaviorBase):
    """Mango :class:`~mango.simulation.environment.Behavior` backed by a PyPSA network.

    Parameters
    ----------
    net:
        A ``pypsa.Network`` with buses, generators, loads and (optionally)
        storage units populated.
    timeseries:
        Mapping of :class:`ComponentRef` (or ``(element_type, component_id)``
        tuples) to :class:`pandas.Series` with a ``DatetimeIndex``.  Values
        are MW set-points (or per-unit availability for renewables; see
        :meth:`_apply_timeseries_update`).
    relevant_types:
        Element types to manage.  Defaults to all four supported types.
    renewable_carriers:
        PyPSA ``carrier`` values considered renewable.  Generators whose
        carrier is in this set are classified as :data:`RENEWABLE`; all
        other generators are :data:`THERMAL`.
    start_datetime:
        Reference datetime for converting timeseries timestamps to
        simulation seconds.  Defaults to the earliest timestamp in
        *timeseries* (or ``datetime.now`` if empty).
    """

    def __init__(
        self,
        net,
        *,
        timeseries: dict[ComponentRef | tuple, pd.Series] | None = None,
        relevant_types: list[str] | None = None,
        renewable_carriers: Iterable[str] | None = None,
        start_datetime: datetime | None = None,
    ) -> None:
        super().__init__(
            net=net,
            timeseries=timeseries,
            relevant_types=relevant_types,
            start_datetime=start_datetime,
        )
        self._renewable_carriers: frozenset[str] = frozenset(
            renewable_carriers
            if renewable_carriers is not None
            else DEFAULT_RENEWABLE_CARRIERS
        )
        self._original_nom_capacities: dict[ComponentRef, float] = {}
        self._store_original_nominal_capacities()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_network(
        cls,
        net,
        *,
        auto_timeseries: bool = True,
        timeseries: dict | None = None,
        relevant_types: list[str] | None = None,
        renewable_carriers: Iterable[str] | None = None,
        start_datetime: datetime | None = None,
    ) -> PyPSABehavior:
        """Build a behavior directly from a :class:`pypsa.Network`.

        When *auto_timeseries* is ``True`` (default) and no explicit
        ``timeseries`` mapping is given, timeseries are extracted from the
        network's ``generators_t`` / ``loads_t`` / ``storage_units_t``
        attributes via :func:`extract_timeseries`.

        Use this when you loaded a network with
        :func:`~energy_scheduling_benchmark.networks.load_scenario` or
        :func:`~energy_scheduling_benchmark.networks.load_example` —
        the network already carries its own snapshots and time-dependent
        data.
        """
        if timeseries is None and auto_timeseries:
            timeseries = extract_timeseries(net, renewable_carriers=renewable_carriers)

        return cls(
            net=net,
            timeseries=timeseries,
            relevant_types=relevant_types,
            renewable_carriers=renewable_carriers,
            start_datetime=start_datetime,
        )

    @classmethod
    def from_scenario(cls, scenario, **kwargs) -> PyPSABehavior:
        """Build a behavior from a scenario bundle exposing ``net``, ``timeseries`` and ``start``.

        See :class:`~energy_scheduling_benchmark.networks.ScenarioData`.
        """
        return cls(
            net=scenario.net,
            timeseries=scenario.timeseries,
            start_datetime=kwargs.pop("start_datetime", scenario.start),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Component discovery
    # ------------------------------------------------------------------

    def get_components_by_type(self, types: list[str]) -> list[ComponentRef]:
        refs: list[ComponentRef] = []
        for et in types:
            refs.extend(self._refs_for_type(et))
        return refs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refs_for_type(self, element_type: str) -> list[ComponentRef]:
        df, predicate = self._dataframe_and_predicate(element_type)
        if df is None or df.empty:
            return []
        mask = (
            df.index.to_series().apply(predicate)
            if predicate
            else pd.Series(True, index=df.index)
        )
        return [ComponentRef(element_type, str(idx)) for idx in df.index[mask]]

    def _dataframe_and_predicate(self, element_type: str):
        """Return ``(df, predicate)`` for a given element type.

        The predicate, if not ``None``, filters rows of the underlying PyPSA
        DataFrame (used for THERMAL vs RENEWABLE distinction on ``n.generators``).
        """
        if element_type == THERMAL:
            df = getattr(self._net, "generators", None)
            if df is None:
                return None, None
            renewables = self._renewable_carriers

            def predicate(idx):
                carrier = (
                    str(df.at[idx, _COL_CARRIER]) if _COL_CARRIER in df.columns else ""
                )
                return carrier.lower() not in renewables

            return df, predicate

        if element_type == RENEWABLE:
            df = getattr(self._net, "generators", None)
            if df is None:
                return None, None
            renewables = self._renewable_carriers

            def predicate(idx):
                carrier = (
                    str(df.at[idx, _COL_CARRIER]) if _COL_CARRIER in df.columns else ""
                )
                return carrier.lower() in renewables

            return df, predicate

        if element_type == LOAD:
            return getattr(self._net, "loads", None), None

        if element_type == STORAGE:
            return getattr(self._net, "storage_units", None), None

        return None, None

    def _store_original_nominal_capacities(self) -> None:
        ren_df, predicate = self._dataframe_and_predicate(RENEWABLE)
        if ren_df is not None:
            for idx in ren_df.index:
                if predicate is None or predicate(idx):
                    ref = ComponentRef(RENEWABLE, str(idx))
                    if _COL_P_NOM in ren_df.columns:
                        self._original_nom_capacities[ref] = float(
                            ren_df.at[idx, _COL_P_NOM]
                        )

    def _dataframe_for(self, element_type: str):
        df, _ = self._dataframe_and_predicate(element_type)
        return df

    def _build_observers(self, ref: ComponentRef) -> dict[str, Callable[[], Any]]:
        """Build the ``observe()`` registry for *ref*.

        Names: ``"statics"`` (full row dict), ``"max_active_power"`` (MW),
        ``"active_power"`` (current set-point, MW), ``"cost"`` (marginal
        cost, generators only).
        """
        et, cid = ref
        df = self._dataframe_for(et)
        if df is None:
            return {}

        def statics() -> dict:
            return df.loc[cid].to_dict()

        def max_active_power() -> float:
            row = df.loc[cid]
            if et == LOAD:
                return float(row.get(_COL_P_SET, 0.0))
            return float(row.get(_COL_P_NOM, row.get(_COL_P_SET, float("nan"))))

        def active_power() -> float:
            return float(df.at[cid, _COL_P_SET]) if _COL_P_SET in df.columns else 0.0

        def cost() -> float:
            return (
                float(df.at[cid, _COL_MARGINAL_COST])
                if _COL_MARGINAL_COST in df.columns
                else 0.0
            )

        return {
            "statics": statics,
            "max_active_power": max_active_power,
            "active_power": active_power,
            "cost": cost,
        }

    def _build_actions(self, ref: ComponentRef) -> dict[str, Callable]:
        et, cid = ref
        if et == LOAD:
            return {}

        df = self._dataframe_for(et)
        if df is None:
            return {}

        def regulate(active_power_mw: float) -> None:
            df.at[cid, _COL_P_SET] = float(active_power_mw)

        return {"regulate": regulate}

    def _apply_timeseries_update(
        self, ref: ComponentRef, value: float, environment: Environment
    ) -> None:
        et, cid = ref
        df = self._dataframe_for(et)
        if df is None:
            return

        if et == RENEWABLE:
            nominal = self._original_nom_capacities.get(ref)
            if nominal is None:
                nominal = (
                    float(df.at[cid, _COL_P_NOM]) if _COL_P_NOM in df.columns else 1.0
                )
                self._original_nom_capacities[ref] = nominal
            df.at[cid, _COL_P_NOM] = value * nominal
        elif et == LOAD:
            df.at[cid, _COL_P_SET] = float(value)
        else:  # THERMAL or STORAGE
            df.at[cid, _COL_P_NOM] = float(value)

        aid = self._ref_to_aid.get(ref)
        if aid is not None:
            environment.emit_agent_event(PowerUpdateInfo(), aid)


# ---------------------------------------------------------------------------
# Timeseries extraction from a PyPSA network
# ---------------------------------------------------------------------------


def extract_timeseries(
    net,
    *,
    renewable_carriers: frozenset[str] | set[str] | None = None,
) -> dict[ComponentRef, pd.Series]:
    """Extract per-component timeseries from ``net.<component>_t`` DataFrames.

    Returns a mapping that can be passed as the ``timeseries`` argument of
    :class:`PyPSABehavior`.  The rules applied per component type are:

    * **Generators** — ``generators_t.p_max_pu`` is preferred (renewable
      availability per unit of ``p_nom``); when absent, ``generators_t.p_set``
      is used.  Classification as :data:`THERMAL`/:data:`RENEWABLE` follows
      the generator's ``carrier`` attribute.
    * **Loads** — ``loads_t.p_set`` (MW).
    * **Storage units** — ``storage_units_t.p_set`` (MW).
    """
    renewables = {c.lower() for c in (renewable_carriers or DEFAULT_RENEWABLE_CARRIERS)}
    ts: dict[ComponentRef, pd.Series] = {}

    # --- Generators ---
    gens = getattr(net, "generators", None)
    gens_t = getattr(net, "generators_t", None)
    if gens is not None and gens_t is not None:
        p_max_pu = _get_t_frame(gens_t, "p_max_pu")
        p_set_t = _get_t_frame(gens_t, "p_set")
        consumed: set[str] = set()
        if p_max_pu is not None:
            for col in p_max_pu.columns:
                et = RENEWABLE if _is_renewable(gens, col, renewables) else THERMAL
                ts[ComponentRef(et, str(col))] = p_max_pu[col].copy()
                consumed.add(col)
        if p_set_t is not None:
            for col in p_set_t.columns:
                if col in consumed:
                    continue
                et = RENEWABLE if _is_renewable(gens, col, renewables) else THERMAL
                ts[ComponentRef(et, str(col))] = p_set_t[col].copy()

    # --- Loads ---
    loads_t = getattr(net, "loads_t", None)
    p_set = _get_t_frame(loads_t, "p_set") if loads_t is not None else None
    if p_set is not None:
        for col in p_set.columns:
            ts[ComponentRef(LOAD, str(col))] = p_set[col].copy()

    # --- Storage units ---
    su_t = getattr(net, "storage_units_t", None)
    p_set = _get_t_frame(su_t, "p_set") if su_t is not None else None
    if p_set is not None:
        for col in p_set.columns:
            ts[ComponentRef(STORAGE, str(col))] = p_set[col].copy()

    return ts


def _is_renewable(gens_df: pd.DataFrame, name: str, renewables: set[str]) -> bool:
    if "carrier" not in gens_df.columns or name not in gens_df.index:
        return False
    return str(gens_df.at[name, "carrier"]).lower() in renewables


def _get_t_frame(t_attr, key: str) -> pd.DataFrame | None:
    """Safely fetch ``t_attr[key]`` (or ``t_attr.key``) and return non-empty frame."""
    frame = None
    if hasattr(t_attr, "get"):
        try:
            frame = t_attr.get(key)
        except TypeError:
            frame = None
    if frame is None and hasattr(t_attr, key):
        frame = getattr(t_attr, key)
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame
