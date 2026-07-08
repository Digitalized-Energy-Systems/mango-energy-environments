"""Shared implementation for the pandapower and PyPSA scheduling behaviors.

:class:`PowerSystemsBehavior` and :class:`PyPSABehavior` manage different
underlying network models but follow the identical shape: normalise a
``ComponentRef``-or-tuple-keyed timeseries mapping, replay it onto installed
agents as the simulation clock advances, and expose a named
observer/action registry per agent. This module factors that shape out once;
each subclass only supplies the network-specific hooks (``get_components_by_type``,
``_build_observers``, ``_build_actions``, ``_apply_timeseries_update``).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from mango.simulation.environment import Behavior, Environment
from mango.util.clock import Clock

from .base import LOAD, RENEWABLE, STORAGE, THERMAL, ComponentRef

logger = logging.getLogger(__name__)

__all__ = ["SchedulingBehaviorBase"]


class SchedulingBehaviorBase(Behavior):
    """Shared observer/action registry and timeseries-replay boilerplate.

    Parameters
    ----------
    net:
        The backend-specific network object (pandapower or PyPSA).
    timeseries:
        Mapping of :class:`ComponentRef` (or ``(element_type, id)`` tuples)
        to :class:`pandas.Series` with a ``DatetimeIndex``.
    relevant_types:
        Element types to manage. Defaults to all four standard types.
    start_datetime:
        Reference point for converting timeseries timestamps to simulation
        seconds. Defaults to the earliest timestamp in *timeseries*, or
        :func:`datetime.now` if *timeseries* is empty.
    """

    def __init__(
        self,
        net: Any,
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
            self._start_dt = datetime.now(UTC).replace(tzinfo=None)

        # aid -> {obs_name: Callable[[], Any]}
        self._observers: dict[str, dict[str, Callable[[], Any]]] = {}
        # aid -> {action_name: Callable}
        self._actions: dict[str, dict[str, Callable]] = {}
        self._ref_to_aid: dict[ComponentRef, str] = {}
        self._ref_to_agent: dict[ComponentRef, Any] = {}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def net(self):
        return self._net

    @property
    def start_datetime(self) -> datetime:
        return self._start_dt

    # ------------------------------------------------------------------
    # Behavior lifecycle
    # ------------------------------------------------------------------

    def initialize(self, environment: Environment, clock: Clock) -> None:
        """Schedule each timeseries point as an agent-managed task."""
        logger.debug(
            "%s: scheduling timeseries tasks via agent schedulers", type(self).__name__
        )
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
        logger.debug("%s: %d timeseries tasks scheduled", type(self).__name__, count)

    def install(self, agent, **kwargs) -> None:
        """Register standard observers and actions for *agent*.

        Expected keyword arguments
        --------------------------
        id:
            A :class:`ComponentRef` or ``(element_type, id)`` tuple.
        """
        raw_id = kwargs["id"]
        ref = raw_id if isinstance(raw_id, ComponentRef) else ComponentRef(*raw_id)

        self._ref_to_aid[ref] = agent.aid
        self._ref_to_agent[ref] = agent
        self._observers[agent.aid] = self._build_observers(ref)
        self._actions[agent.aid] = self._build_actions(ref)

    # ------------------------------------------------------------------
    # Observer / action interface
    # ------------------------------------------------------------------

    def observe(self, agent_id: str, name: str = "active_power") -> Any:
        """Return the named observation for *agent_id* (``None`` if unknown)."""
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

    def has_timeseries(self, ref: ComponentRef) -> bool:
        """Return True if *ref* has an entry in the replayed timeseries."""
        return ref in self._timeseries

    def get_statics(self, ref: ComponentRef) -> dict:
        """Return the static component row for *ref* as a dict (empty if unknown).

        Same data as the ``"statics"`` observer, but addressable by
        :class:`ComponentRef` so scenarios can query components before (or
        without) installing an agent for them.
        """
        fn = self._build_observers(ref).get("statics")
        if fn is None:
            logger.warning("No statics available for component %r", ref)
            return {}
        return fn()

    # ------------------------------------------------------------------
    # Component discovery
    # ------------------------------------------------------------------

    def get_possible_components(self) -> list[ComponentRef]:
        """Return all components matching :attr:`relevant_types`."""
        return self.get_components_by_type(self._relevant_types)

    def calculate_initial_time(self) -> datetime:
        """Return the earliest timeseries timestamp across all managed components."""
        return self._earliest_timestamp()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _update_coro(
        self,
        ref: ComponentRef,
        value: float,
        environment: Environment,
    ) -> None:
        self._apply_timeseries_update(ref, value, environment)

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
        return earliest or datetime.now(UTC).replace(tzinfo=None)

    # ------------------------------------------------------------------
    # Backend-specific hooks — implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def get_components_by_type(self, types: list[str]) -> list[ComponentRef]:
        """Return :class:`ComponentRef` objects for all components of the given types."""

    @abstractmethod
    def _build_observers(self, ref: ComponentRef) -> dict[str, Callable[[], Any]]:
        """Return the ``{name: getter}`` observer registry for *ref*."""

    @abstractmethod
    def _build_actions(self, ref: ComponentRef) -> dict[str, Callable]:
        """Return the ``{name: setter}`` action registry for *ref*."""

    @abstractmethod
    def _apply_timeseries_update(
        self, ref: ComponentRef, value: float, environment: Environment
    ) -> None:
        """Apply *value* for *ref* at the current timestep and emit a change event."""
