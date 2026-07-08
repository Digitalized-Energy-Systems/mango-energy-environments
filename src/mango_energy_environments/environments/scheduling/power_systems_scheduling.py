"""Power systems scheduling environment.

Uses **pandapower** as the power system data model and **HiGHS** (via scipy)
for copper-plate economic dispatch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from mango.simulation.environment import Environment

from ._scheduling_behavior_base import SchedulingBehaviorBase
from .base import LOAD, RENEWABLE, STORAGE, THERMAL, ComponentRef, PowerUpdateInfo

logger = logging.getLogger(__name__)

__all__ = [
    "PowerSystemsBehavior",
]

#: Maps the shared taxonomy to pandapower's table attribute names.
_TABLE_NAME = {THERMAL: "gen", RENEWABLE: "sgen", LOAD: "load", STORAGE: "storage"}

_COL_P_MW = "p_mw"
_COL_MAX_P_MW = "max_p_mw"
_COL_MIN_P_MW = "min_p_mw"
_COL_IN_SERVICE = "in_service"


class PowerSystemsBehavior(SchedulingBehaviorBase):
    """Mango environment behavior for power-systems scheduling and dispatch.

    Manages a pandapower network, wires up per-agent observers and actions,
    and replays timeseries data as the simulation progresses.

    Parameters
    ----------
    net:
        A ``pandapower.Network`` instance containing buses, generators,
        loads, and (optionally) storage units.
    timeseries:
        Mapping of :class:`ComponentRef` (or ``(element_type, index)`` tuples)
        to :class:`pandas.Series` with a :class:`pandas.DatetimeIndex`.
        Each series entry represents the ``p_mw`` value at that timestamp.
    relevant_types:
        Element types to manage.  Defaults to all four standard types.
    start_datetime:
        Reference point for converting timeseries timestamps to simulation
        seconds.  Defaults to the earliest timestamp found in *timeseries*.
        If *timeseries* is empty, defaults to :func:`datetime.now`.
    """

    def get_components_by_type(self, types: list[str]) -> list[ComponentRef]:
        """Return :class:`ComponentRef` objects for all components of the given types."""
        refs: list[ComponentRef] = []
        for et in types:
            df = getattr(self._net, _TABLE_NAME[et], None)
            if df is None or df.empty:
                continue
            for idx in df.index:
                refs.append(ComponentRef(et, idx))
        return refs

    def solve_central(self) -> dict:
        """Solve a copper-plate economic dispatch (lossless, no network constraints).

        Minimises generation cost subject to:

        - Power balance: total generation = total fixed load − fixed renewables.
        - Generator limits: ``min_p_mw ≤ p_mw ≤ max_p_mw``.
        - Storage limits: ``min_p_mw ≤ p_mw ≤ max_p_mw`` (only when STORAGE
          is in :attr:`relevant_types`).

        Returns
        -------
        dict with keys ``"success"`` (bool), ``"net"`` (updated network),
        ``"objective"`` (float).
        """
        from scipy.optimize import linprog

        controllable: list[tuple[str, int]] = []
        costs: list[float] = []
        p_min: list[float] = []
        p_max: list[float] = []

        for et in (THERMAL, STORAGE):
            if et not in self._relevant_types:
                continue
            df = getattr(self._net, _TABLE_NAME[et], None)
            if df is None or df.empty:
                continue
            active = df[df.get(_COL_IN_SERVICE, pd.Series(True, index=df.index))]
            for idx, row in active.iterrows():
                controllable.append((et, idx))
                costs.append(float(row.get("cost_per_mw", 1.0)))
                p_min.append(float(row.get(_COL_MIN_P_MW, 0.0)))
                p_max.append(float(row.get(_COL_MAX_P_MW, row.get(_COL_P_MW, 0.0))))

        if not controllable:
            logger.warning("solve_central: no controllable generators found")
            return {"success": False, "net": self._net, "objective": float("nan")}

        sgen_df = getattr(self._net, _TABLE_NAME[RENEWABLE], None)
        fixed_gen_mw = 0.0
        if (
            sgen_df is not None
            and not sgen_df.empty
            and RENEWABLE in self._relevant_types
        ):
            active_sgen = sgen_df[
                sgen_df.get(_COL_IN_SERVICE, pd.Series(True, index=sgen_df.index))
            ]
            fixed_gen_mw = float(active_sgen[_COL_P_MW].sum())

        load_df = getattr(self._net, _TABLE_NAME[LOAD], None)
        total_demand_mw = 0.0
        if load_df is not None and not load_df.empty:
            active_load = load_df[
                load_df.get(_COL_IN_SERVICE, pd.Series(True, index=load_df.index))
            ]
            total_demand_mw = float(active_load[_COL_P_MW].sum())

        net_demand_mw = total_demand_mw - fixed_gen_mw

        result = linprog(
            costs,
            A_eq=[[1.0] * len(controllable)],
            b_eq=[net_demand_mw],
            bounds=list(zip(p_min, p_max)),
            method="highs",
        )

        if result.success:
            for i, (et, idx) in enumerate(controllable):
                getattr(self._net, _TABLE_NAME[et]).at[idx, _COL_P_MW] = result.x[i]
            logger.info(
                "solve_central: dispatch successful, objective=%.4f MW·cost",
                result.fun,
            )
        else:
            logger.warning("solve_central: LP failed — %s", result.message)

        return {
            "success": result.success,
            "net": self._net,
            "objective": result.fun if result.success else float("nan"),
        }

    def _build_observers(self, ref: ComponentRef) -> dict[str, Callable[[], Any]]:
        """Build the ``observe()`` registry for *ref*.

        Names: ``"statics"`` (full row dict), ``"max_active_power"`` (MW),
        ``"active_power"`` (current set-point, MW).
        """
        et, idx = ref
        table = _TABLE_NAME[et]

        def statics() -> dict:
            return getattr(self._net, table).loc[idx].to_dict()

        def max_active_power() -> float:
            row = getattr(self._net, table).loc[idx]
            return float(row.get(_COL_MAX_P_MW, row.get(_COL_P_MW, float("nan"))))

        def active_power() -> float:
            return float(getattr(self._net, table).at[idx, _COL_P_MW])

        return {
            "statics": statics,
            "max_active_power": max_active_power,
            "active_power": active_power,
        }

    def _build_actions(self, ref: ComponentRef) -> dict[str, Callable]:
        et, idx = ref
        actions: dict[str, Callable] = {}

        if et in (THERMAL, RENEWABLE, STORAGE):
            table = _TABLE_NAME[et]

            def regulate(active_power_mw: float) -> None:
                getattr(self._net, table).at[idx, _COL_P_MW] = active_power_mw

            actions["regulate"] = regulate

        return actions

    def _apply_timeseries_update(
        self, ref: ComponentRef, value: float, environment: Environment
    ) -> None:
        et, idx = ref
        table = _TABLE_NAME[et]
        if et == RENEWABLE:
            nominal = getattr(self._net, table).at[idx, _COL_MAX_P_MW]
            getattr(self._net, table).at[idx, _COL_MAX_P_MW] = value * nominal
        else:
            getattr(self._net, table).at[idx, _COL_MAX_P_MW] = value

        aid = self._ref_to_aid.get(ref)
        if aid is not None:
            environment.emit_agent_event(PowerUpdateInfo(), aid)
