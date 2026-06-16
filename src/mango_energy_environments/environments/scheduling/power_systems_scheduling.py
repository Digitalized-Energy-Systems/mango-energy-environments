"""Power systems scheduling environment.

Uses **pandapower** as the power system data model and **HiGHS** (via scipy)
for copper-plate economic dispatch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mango.simulation.environment import Behavior, Environment
from mango.util.clock import Clock

logger = logging.getLogger(__name__)

__all__ = [
    "PowerUpdateInfo",
    "PowerSystemsBehavior",
    "calculate_initial_time",
    "get_possible_components",
    "get_components_by_type",
]

#: Synchronous / thermal generator.
THERMAL = "gen"
#: Static / renewable generator (wind, solar, run-of-river).
RENEWABLE = "sgen"
#: Demand / load.
LOAD = "load"
#: Energy storage / battery.
STORAGE = "storage"

_COL_P_MW = "p_mw"
_COL_MAX_P_MW = "max_p_mw"
_COL_MIN_P_MW = "min_p_mw"
_COL_IN_SERVICE = "in_service"


@dataclass(frozen=True)
class PowerUpdateInfo:
    """Emitted to an agent's event stream whenever its power value changes.

    Carries no payload; the agent re-reads its observer to get the new value.
    """


@dataclass(frozen=True)
class ComponentRef:
    """Identifies a single component in a pandapower network.

    Parameters
    ----------
    element_type:
        One of ``"gen"``, ``"sgen"``, ``"load"``, ``"storage"``.
    index:
        The integer row index in the corresponding DataFrame.
    """

    element_type: str
    index: int

    def __iter__(self):
        yield self.element_type
        yield self.index


class PowerSystemsBehavior(Behavior):
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

    def __init__(
        self,
        net,
        timeseries: dict[ComponentRef | tuple, pd.Series] | None = None,
        relevant_types: list[str] | None = None,
        start_datetime: datetime | None = None,
    ) -> None:
        self._net = net
        self._timeseries: dict[ComponentRef, pd.Series] = {
            (k if isinstance(k, ComponentRef) else ComponentRef(*k)): v
            for k, v in (timeseries or {}).items()
        }
        self._relevant_types: list[str] = relevant_types or [
            THERMAL,
            RENEWABLE,
            LOAD,
            STORAGE,
        ]

        if start_datetime is not None:
            self._start_dt: datetime = start_datetime
        elif self._timeseries:
            self._start_dt = self._earliest_timestamp()
        else:
            self._start_dt = datetime.now(timezone.utc).replace(tzinfo=None)

        # aid -> {obs_name: Callable[[], Any]}
        self._observers: dict[str, dict[str, Callable[[], Any]]] = {}
        # aid -> {action_name: Callable}
        self._actions: dict[str, dict[str, Callable]] = {}
        self._ref_to_aid: dict[ComponentRef, str] = {}
        self._ref_to_agent: dict[ComponentRef, Any] = {}

    @property
    def net(self):
        return self._net

    @property
    def start_datetime(self) -> datetime:
        return self._start_dt

    def initialize(self, environment: Environment, clock: Clock) -> None:
        """Schedule each timeseries point as an agent-managed task."""
        logger.debug("PowerSystemsBehavior: scheduling timeseries tasks via agent schedulers")
        count = 0
        for ref, series in self._timeseries.items():
            if ref.element_type not in self._relevant_types:
                continue
            agent = self._ref_to_agent.get(ref)
            if agent is None:
                logger.debug("No agent installed for %s; skipping timeseries", ref)
                continue
            for ts, value in series.items():
                t_s = self._ts_to_seconds(ts)
                agent.schedule_timestamp_task(
                    self._update_coro(ref, float(value), environment),
                    timestamp=t_s,
                )
                count += 1
        logger.debug("PowerSystemsBehavior: %d timeseries tasks scheduled", count)

    def install(self, agent, **kwargs) -> None:
        """Register standard observers and actions for *agent*.

        Expected keyword arguments
        --------------------------
        id:
            A :class:`ComponentRef` or ``(element_type, index)`` tuple.
        """
        raw_id = kwargs["id"]
        ref = raw_id if isinstance(raw_id, ComponentRef) else ComponentRef(*raw_id)

        self._ref_to_aid[ref] = agent.aid
        self._ref_to_agent[ref] = agent
        self._observers[agent.aid] = self._build_observers(ref)
        self._actions[agent.aid] = self._build_actions(ref)

    def observe(self, agent_id: str, name: str = "active_power") -> Any:
        """Return the named observation for *agent_id*.

        Built-in observer names:

        - ``"statics"``          – full row dict of the component DataFrame.
        - ``"max_active_power"`` – current maximum active power (MW).
        - ``"active_power"``     – current active power setpoint (MW).
        """
        fn = self._observers.get(agent_id, {}).get(name)
        if fn is None:
            logger.warning("No observer %r for agent %r", name, agent_id)
            return None
        return fn()

    def act(self, agent_id: str, action: str, *args: Any, **kwargs: Any) -> None:
        fn = self._actions.get(agent_id, {}).get(action)
        if fn is not None:
            fn(*args, **kwargs)
        else:
            logger.warning("No action %r for agent %r", action, agent_id)

    def has_action(self, agent_id: str, action: str) -> bool:
        return action in self._actions.get(agent_id, {})

    def get_components_by_type(self, types: list[str]) -> list[ComponentRef]:
        """Return :class:`ComponentRef` objects for all components of the given types."""
        refs: list[ComponentRef] = []
        for et in types:
            df = getattr(self._net, et, None)
            if df is None or df.empty:
                continue
            for idx in df.index:
                refs.append(ComponentRef(et, idx))
        return refs

    def get_possible_components(self) -> list[ComponentRef]:
        """Return all components matching :attr:`relevant_types`."""
        return self.get_components_by_type(self._relevant_types)

    def calculate_initial_time(self) -> datetime:
        """Return the earliest timeseries timestamp across all managed components."""
        return self._earliest_timestamp()

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
            df = getattr(self._net, et, None)
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

        sgen_df = getattr(self._net, RENEWABLE, None)
        fixed_gen_mw = 0.0
        if sgen_df is not None and not sgen_df.empty and RENEWABLE in self._relevant_types:
            active_sgen = sgen_df[
                sgen_df.get(_COL_IN_SERVICE, pd.Series(True, index=sgen_df.index))
            ]
            fixed_gen_mw = float(active_sgen[_COL_P_MW].sum())

        load_df = getattr(self._net, LOAD, None)
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
                getattr(self._net, et).at[idx, _COL_P_MW] = result.x[i]
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

    async def _update_coro(
        self,
        ref: ComponentRef,
        value: float,
        environment: Environment,
    ) -> None:
        self._apply_timeseries_update(ref, value, environment)

    def _build_observers(self, ref: ComponentRef) -> dict[str, Callable[[], Any]]:
        et, idx = ref

        def statics() -> dict:
            return getattr(self._net, et).loc[idx].to_dict()

        def max_active_power() -> float:
            row = getattr(self._net, et).loc[idx]
            return float(row.get(_COL_MAX_P_MW, row.get(_COL_P_MW, float("nan"))))

        def active_power() -> float:
            return float(getattr(self._net, et).at[idx, _COL_P_MW])

        return {
            "statics": statics,
            "max_active_power": max_active_power,
            "active_power": active_power,
        }

    def _build_actions(self, ref: ComponentRef) -> dict[str, Callable]:
        et, idx = ref
        actions: dict[str, Callable] = {}

        if et in (THERMAL, RENEWABLE, STORAGE):
            def regulate(active_power_mw: float) -> None:
                getattr(self._net, et).at[idx, _COL_P_MW] = active_power_mw

            actions["regulate"] = regulate

        return actions

    def _apply_timeseries_update(
        self, ref: ComponentRef, value: float, environment: Environment
    ) -> None:
        et, idx = ref
        if et == RENEWABLE:
            nominal = getattr(self._net, et).at[idx, _COL_MAX_P_MW]
            getattr(self._net, et).at[idx, _COL_MAX_P_MW] = value * nominal
        else:
            getattr(self._net, et).at[idx, _COL_MAX_P_MW] = value

        aid = self._ref_to_aid.get(ref)
        if aid is not None:
            environment.emit_agent_event(PowerUpdateInfo(), aid)

    def _ts_to_seconds(self, ts) -> float:
        if isinstance(ts, datetime):
            return (ts - self._start_dt).total_seconds()
        if hasattr(ts, "to_pydatetime"):
            return (ts.to_pydatetime() - self._start_dt).total_seconds()
        return float(ts)

    def _earliest_timestamp(self) -> datetime:
        earliest: datetime | None = None
        for series in self._timeseries.values():
            if series.empty:
                continue
            first = series.index[0]
            dt = first.to_pydatetime() if hasattr(first, "to_pydatetime") else first
            if earliest is None or dt < earliest:
                earliest = dt
        return earliest or datetime.now(timezone.utc).replace(tzinfo=None)


def calculate_initial_time(behavior: PowerSystemsBehavior) -> datetime:
    return behavior.calculate_initial_time()


def get_possible_components(behavior: PowerSystemsBehavior) -> list[ComponentRef]:
    return behavior.get_possible_components()


def get_components_by_type(
    behavior: PowerSystemsBehavior, types: list[str]
) -> list[ComponentRef]:
    return behavior.get_components_by_type(types)
