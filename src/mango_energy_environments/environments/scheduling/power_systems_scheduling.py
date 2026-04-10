"""Power systems scheduling environment.

Translates MangoEnergyEnvironments.jl/src/environments/scheduling/power_systems_scheduling.jl
to idiomatic Python, replacing PowerSystems.jl + PowerSimulations.jl with
**pandapower** as the power system data model and **HiGHS** (via scipy) for
copper-plate economic dispatch.

Mapping of Julia → Python concepts
-----------------------------------
``PowerSystems.System``             → ``pandapower.Network``
``ThermalStandard``                 → element type ``"gen"`` (synchronous generator)
``RenewableDispatch``               → element type ``"sgen"`` (static/renewable generator)
``PowerLoad``                       → element type ``"load"``
``EnergyReservoirStorage``          → element type ``"storage"``
component UUID                      → ``(element_type, index)`` tuple
timeseries data                     → ``dict[(element_type, index), pd.Series]``
``PowerSimulations`` copper plate   → HiGHS LP via ``scipy.optimize.linprog``

Design decisions
----------------
- The Julia version schedules timeseries updates via Mango's task system.
  The Python port uses the same approach: each timeseries point becomes an
  asyncio task that ``await``s ``clock.sleep(delay)`` then applies the update.
  This registers the wakeup time in ``ExternalClock._futures``, which drives
  ``discrete_step_until``'s step-size determination — equivalent semantics to
  Julia's ``schedule(env, TimeseriesTaskData(dates))``.
- Observers are keyed by ``(obs_name, agent_id)`` so multiple named
  observers can coexist per agent (matching the Julia ``install_observer``
  overload that accepts a name symbol).
- Actions are keyed by ``action_name`` per agent, matching Julia's
  ``install_action(agent, :regulate) do ...``.
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

# ---------------------------------------------------------------------------
# Component-type constants (string aliases for pandapower element tables)
# ---------------------------------------------------------------------------

#: Synchronous / thermal generator.
THERMAL = "gen"
#: Static / renewable generator (wind, solar, run-of-river).
RENEWABLE = "sgen"
#: Demand / load.
LOAD = "load"
#: Energy storage / battery.
STORAGE = "storage"

# Column names used across element types
_COL_P_MW = "p_mw"
_COL_MAX_P_MW = "max_p_mw"
_COL_MIN_P_MW = "min_p_mw"
_COL_IN_SERVICE = "in_service"


# ---------------------------------------------------------------------------
# Event type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerUpdateInfo:
    """Emitted to an agent's event stream whenever its power value changes.

    Equivalent to the Julia ``struct PowerUpdateInfo end`` — carries no
    payload; the agent simply re-reads its observer to get the new value.
    """


# ---------------------------------------------------------------------------
# Component reference helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentRef:
    """Identifies a single component in a pandapower network.

    Parameters
    ----------
    element_type:
        One of ``"gen"``, ``"sgen"``, ``"load"``, ``"storage"``.
    index:
        The integer row index in the corresponding DataFrame
        (``net.gen.index``, etc.).
    """

    element_type: str
    index: int

    def __iter__(self):
        # Allow unpacking: element_type, index = ref
        yield self.element_type
        yield self.index


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


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

        # Determine reference datetime (used to convert timestamps → seconds)
        if start_datetime is not None:
            self._start_dt: datetime = start_datetime
        elif self._timeseries:
            self._start_dt = self._earliest_timestamp()
        else:
            self._start_dt = datetime.now(timezone.utc).replace(tzinfo=None)

        # Per-agent observer dicts:  aid -> {obs_name: Callable[[], Any]}
        self._observers: dict[str, dict[str, Callable[[], Any]]] = {}
        # Per-agent action dicts:    aid -> {action_name: Callable}
        self._actions: dict[str, dict[str, Callable]] = {}
        # Map ComponentRef -> agent ID for event delivery
        self._ref_to_aid: dict[ComponentRef, str] = {}
        # Map ComponentRef -> agent object (for scheduler-based task scheduling)
        self._ref_to_agent: dict[ComponentRef, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def net(self):
        """The pandapower Network."""
        return self._net

    @property
    def start_datetime(self) -> datetime:
        """Reference datetime for simulation clock offset."""
        return self._start_dt

    # ------------------------------------------------------------------
    # Behavior interface
    # ------------------------------------------------------------------

    def initialize(self, environment: Environment, clock: Clock) -> None:
        """Schedule each timeseries point as an agent-managed task.

        For each timeseries entry, a ``TimestampScheduledTask`` is registered
        on the corresponding agent's scheduler via
        ``agent.schedule_timestamp_task(coro, timestamp)``.  This integrates
        with mango's ``tasks_complete_or_sleeping`` mechanism and registers the
        wakeup time in ``ExternalClock._futures``, driving
        ``discrete_step_until``'s step-size determination — equivalent to
        Julia's ``schedule(env, TimeseriesTaskData(dates))``.
        """
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
            A :class:`ComponentRef` or ``(element_type, index)`` tuple
            identifying the pandapower component.
        """
        raw_id = kwargs["id"]
        ref = raw_id if isinstance(raw_id, ComponentRef) else ComponentRef(*raw_id)

        self._ref_to_aid[ref] = agent.aid
        self._ref_to_agent[ref] = agent
        self._observers[agent.aid] = self._build_observers(ref)
        self._actions[agent.aid] = self._build_actions(ref)

    # ------------------------------------------------------------------
    # Observer / action API
    # ------------------------------------------------------------------

    def observe(self, agent_id: str, name: str = "active_power") -> Any:
        """Return the named observation for *agent_id*.

        Built-in observer names:

        - ``"statics"``        – full row dict of the component DataFrame.
        - ``"max_active_power"`` – current maximum active power (MW).
        - ``"active_power"``   – current active power setpoint (MW).
        """
        fn = self._observers.get(agent_id, {}).get(name)
        if fn is None:
            logger.warning("No observer %r for agent %r", name, agent_id)
            return None
        return fn()

    def act(self, agent_id: str, action: str, *args: Any, **kwargs: Any) -> None:
        """Execute *action* for *agent_id*."""
        fn = self._actions.get(agent_id, {}).get(action)
        if fn is not None:
            fn(*args, **kwargs)
        else:
            logger.warning("No action %r for agent %r", action, agent_id)

    def has_action(self, agent_id: str, action: str) -> bool:
        return action in self._actions.get(agent_id, {})

    # ------------------------------------------------------------------
    # Component queries
    # ------------------------------------------------------------------

    def get_components_by_type(self, types: list[str]) -> list[ComponentRef]:
        """Return :class:`ComponentRef` objects for all components of the given types.

        Parameters
        ----------
        types:
            List of element type strings, e.g. ``["gen", "load"]``.
        """
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

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def calculate_initial_time(self) -> datetime:
        """Return the earliest timeseries timestamp across all managed components.

        Equivalent to the Julia ``calculate_initial_time`` function.
        """
        return self._earliest_timestamp()

    # ------------------------------------------------------------------
    # Economic dispatch (copper-plate)
    # ------------------------------------------------------------------

    def solve_central(self) -> dict:
        """Solve a copper-plate economic dispatch (lossless, no network constraints).

        Maps to the Julia ``solve_central(behavior, time_horizon)`` which uses
        ``CopperPlatePowerModel`` – an energy-balance model with no transmission
        limits.

        The dispatch minimises generation cost (``cost_per_mw`` column if present,
        otherwise uniform cost of 1) subject to:

        - Power balance: total generation = total fixed load − fixed renewables.
        - Generator limits: ``min_p_mw ≤ p_mw ≤ max_p_mw``.
        - Storage limits: ``min_p_mw ≤ p_mw ≤ max_p_mw`` (only when STORAGE
          is in :attr:`relevant_types`).

        Returns
        -------
        dict with keys:
            ``"success"`` (bool), ``"net"`` (updated network), ``"objective"`` (float).
        """
        from scipy.optimize import linprog

        # --- Collect controllable generators (thermal + storage if relevant) --
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

        # --- Fixed renewable generation (sgen) ---------------------------
        sgen_df = getattr(self._net, RENEWABLE, None)
        fixed_gen_mw = 0.0
        if sgen_df is not None and not sgen_df.empty and RENEWABLE in self._relevant_types:
            active_sgen = sgen_df[
                sgen_df.get(_COL_IN_SERVICE, pd.Series(True, index=sgen_df.index))
            ]
            fixed_gen_mw = float(active_sgen[_COL_P_MW].sum())

        # --- Fixed demand (load) -----------------------------------------
        load_df = getattr(self._net, LOAD, None)
        total_demand_mw = 0.0
        if load_df is not None and not load_df.empty:
            active_load = load_df[
                load_df.get(_COL_IN_SERVICE, pd.Series(True, index=load_df.index))
            ]
            total_demand_mw = float(active_load[_COL_P_MW].sum())

        # Net demand that controllable units must cover
        net_demand_mw = total_demand_mw - fixed_gen_mw

        # --- LP: min c^T x  s.t. sum(x) = net_demand, p_min ≤ x ≤ p_max --
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _update_coro(
        self,
        ref: ComponentRef,
        value: float,
        environment: Environment,
    ) -> None:
        """Coroutine body for a single timeseries update (runs after the scheduler delay)."""
        self._apply_timeseries_update(ref, value, environment)

    def _build_observers(self, ref: ComponentRef) -> dict[str, Callable[[], Any]]:
        """Build the standard observer dict for a component."""
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
        """Build the action dict for a component.

        Only thermal generators, renewable generators, and storage units
        support the *regulate* action (matching the Julia condition on
        ``ThermalStandard | RenewableDispatch | EnergyReservoirStorage``).
        """
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
        """Apply a single timeseries value and notify the agent."""
        et, idx = ref
        # Match Julia's pattern:
        #   RenewableDispatch  → set_rating! (scales maximum output)
        #   others             → set_max_active_power! (absolute MW cap)
        if et == RENEWABLE:
            # For renewables the timeseries represents a per-unit availability
            # factor; multiply by nominal capacity to get MW ceiling.
            nominal = getattr(self._net, et).at[idx, _COL_MAX_P_MW]
            getattr(self._net, et).at[idx, _COL_MAX_P_MW] = value * nominal
        else:
            getattr(self._net, et).at[idx, _COL_MAX_P_MW] = value

        # Notify the corresponding agent
        aid = self._ref_to_aid.get(ref)
        if aid is not None:
            environment.emit_agent_event(PowerUpdateInfo(), aid)

    def _ts_to_seconds(self, ts) -> float:
        """Convert a timeseries index entry to seconds offset from *start_dt*."""
        if isinstance(ts, datetime):
            return (ts - self._start_dt).total_seconds()
        if hasattr(ts, "to_pydatetime"):
            return (ts.to_pydatetime() - self._start_dt).total_seconds()
        # Fallback: assume already numeric seconds
        return float(ts)

    def _earliest_timestamp(self) -> datetime:
        """Return the earliest timestamp across all registered timeseries."""
        earliest: datetime | None = None
        for series in self._timeseries.values():
            if series.empty:
                continue
            first = series.index[0]
            dt = first.to_pydatetime() if hasattr(first, "to_pydatetime") else first
            if earliest is None or dt < earliest:
                earliest = dt
        return earliest or datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (mirror Julia dispatch-style functions)
# ---------------------------------------------------------------------------


def calculate_initial_time(behavior: PowerSystemsBehavior) -> datetime:
    """Return the earliest timeseries timestamp for *behavior*.

    Thin wrapper around :meth:`~PowerSystemsBehavior.calculate_initial_time`.
    """
    return behavior.calculate_initial_time()


def get_possible_components(behavior: PowerSystemsBehavior) -> list[ComponentRef]:
    """Return all relevant components in *behavior*'s network.

    Thin wrapper around :meth:`~PowerSystemsBehavior.get_possible_components`.
    """
    return behavior.get_possible_components()


def get_components_by_type(
    behavior: PowerSystemsBehavior, types: list[str]
) -> list[ComponentRef]:
    """Return components in *behavior*'s network filtered by *types*.

    Thin wrapper around :meth:`~PowerSystemsBehavior.get_components_by_type`.
    """
    return behavior.get_components_by_type(types)
