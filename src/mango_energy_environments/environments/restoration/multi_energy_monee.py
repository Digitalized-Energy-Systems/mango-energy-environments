"""Multi-energy system restoration environment using monee.

Translates MangoEnergyEnvironments.jl/src/environments/restoration/multi_energy_monee.jl
to idiomatic Python.  The design keeps the same logical structure:

- :class:`Failure` – data container for a scheduled failure event.
- :class:`BranchFailureEvent`, :class:`NodeFailureEvent`, :class:`CustomFailureEvent`
  – event objects broadcast via the mango environment.
- :class:`RestorationEnvironmentBehavior` – the core :class:`~mango.simulation.environment.Behavior`
  that manages energy flow, failures, and per-agent observers/actions.
- :func:`topology_based_on_grid`, :func:`topology_based_on_grid_groups`
  – helpers that populate a mango :class:`~mango.express.topology.Topology` from
  a monee network.
- :func:`create_branch_aid` – canonical agent-ID string for branch agents.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mango.agent.core import State
from mango.express.topology import Topology
from mango.simulation.environment import Behavior, Environment
from mango.util.clock import Clock

from mango_energy_environments.base.monee import connected_components, energyflow

logger = logging.getLogger(__name__)

__all__ = [
    "Failure",
    "BranchFailureEvent",
    "NodeFailureEvent",
    "CustomFailureEvent",
    "RestorationEnvironmentBehavior",
    "schedule_failure",
    "apply_failures",
    "on_branch_failure",
    "on_node_failure",
    "on_custom_failure",
    "create_branch_aid",
    "topology_based_on_grid",
    "topology_based_on_grid_groups",
]


# ---------------------------------------------------------------------------
# Failure data class
# ---------------------------------------------------------------------------


@dataclass
class Failure:
    """Describes a failure event to be applied to the monee network.

    Parameters
    ----------
    delay_s:
        Seconds from the scheduling moment until the failure triggers.
    branch_ids:
        Sequence of branch IDs (tuples) to deactivate.
    node_ids:
        Sequence of node IDs (integers) to deactivate.
    custom:
        Optional callable ``(net) -> None`` applied to the network on failure.
    custom_id:
        Identifier emitted with the :class:`CustomFailureEvent`; may be any value.
    """

    delay_s: float
    branch_ids: list[tuple] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)
    custom: Callable | None = None
    custom_id: Any = None


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchFailureEvent:
    """Broadcast globally when a branch is deactivated by a failure."""

    branch_id: tuple


@dataclass(frozen=True)
class NodeFailureEvent:
    """Broadcast globally when a node is deactivated by a failure."""

    node_id: int


@dataclass(frozen=True)
class CustomFailureEvent:
    """Broadcast globally when a custom failure function fires."""

    custom_id: Any


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


class RestorationEnvironmentBehavior(Behavior):
    """Mango environment behavior for multi-energy restoration scenarios.

    Integrates a monee :class:`~monee.model.network.Network` into a mango
    :class:`~mango.simulation.world.SimulationWorld`.  On each simulation step
    the behavior:

    1. Checks for scheduled failures that have become due and applies them.
    2. Recomputes the energy flow (via monee) when the network state is dirty.

    Per-agent *observers* (read-only state queries) and *actions* (state
    mutations) are registered in :meth:`install` and exposed via :meth:`observe`
    and :meth:`act`.

    Parameters
    ----------
    net:
        The monee network to simulate.
    on_branch_failure:
        Called with ``(branch_id,)`` after each branch failure.
    on_node_failure:
        Called with ``(node_id,)`` after each node failure.
    on_custom_failure:
        Called with ``(custom_id,)`` after each custom failure.
    """

    def __init__(
        self,
        net,
        on_branch_failure: Callable[[Any], None] = lambda _: None,
        on_node_failure: Callable[[Any], None] = lambda _: None,
        on_custom_failure: Callable[[Any], None] = lambda _: None,
    ) -> None:
        self._net = net
        self._net_results = None
        self._failures: list[Failure] = []
        self._on_branch_failure = on_branch_failure
        self._on_node_failure = on_node_failure
        self._on_custom_failure = on_custom_failure
        self._dirty: bool = False

        # Sorted list of (trigger_time_s, seq, Failure) for pending failures.
        # bisect.insort keeps it ordered; the seq tie-breaker avoids comparing
        # Failure objects (which don't define __lt__).
        self._scheduled_failures: list[tuple[float, int, Failure]] = []
        self._failure_seq: int = 0

        # Per-agent hooks -------------------------------------------------
        # aid -> Callable[[], dict]
        self._observers: dict[str, Callable[[], dict]] = {}
        # aid -> {action_name: Callable}
        self._actions: dict[str, dict[str, Callable]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def net(self):
        """The monee Network (live model)."""
        return self._net

    @property
    def net_results(self):
        """Result object from the last energy flow run (``result.network`` for post-flow state)."""
        return self._net_results

    @property
    def failures(self) -> list[Failure]:
        """All failures registered via :func:`schedule_failure`."""
        return self._failures

    # ------------------------------------------------------------------
    # Behavior interface
    # ------------------------------------------------------------------

    def initialize(self, environment: Environment, clock: Clock) -> None:
        """Compute the initial energy flow before the first simulation step."""
        logger.debug("RestorationEnvironmentBehavior: running initial energy flow")
        self._net_results = energyflow(self._net)

    def on_step(
        self, environment: Environment, clock: Clock, step_size_s: float
    ) -> None:
        """Fire due failures and recompute energy flow when the network is dirty.

        Note: ``on_step`` is called *before* ``clock.set_time`` advances the
        clock.  Use ``clock.time + step_size_s`` as the effective end-of-step
        time to determine which failures have become due.
        """
        end_time = clock.time + step_size_s

        triggered: list[Failure] = []
        remaining: list[tuple[float, int, Failure]] = []
        for entry in self._scheduled_failures:
            t, _seq, failure = entry
            if t <= end_time:
                triggered.append(failure)
            else:
                remaining.append(entry)
        self._scheduled_failures = remaining

        if triggered:
            self._handle_failures(environment, triggered)

        if self._dirty:
            logger.debug(
                "RestorationEnvironmentBehavior: recomputing energy flow "
                "(t=%.3f, dt=%.3f)",
                clock.time,
                step_size_s,
            )
            self._net_results = energyflow(self._net)
            self._dirty = False

    def install(self, agent, **kwargs) -> None:
        """Register observers and actions for *agent* based on its component type.

        Expected keyword arguments
        --------------------------
        id:
            Component ID in the monee network (int for nodes, tuple for branches).
        type:
            ``"child"``, ``"node"``, or ``"branch"``.
        """
        component_id = kwargs["id"]
        component_type = kwargs["type"]

        if component_type == "child":
            self._install_child(agent.aid, component_id)
        elif component_type == "node":
            self._install_node(agent.aid, component_id)
        elif component_type == "branch":
            self._install_branch(agent.aid, component_id)
        else:
            raise ValueError(
                f"Unknown component type {component_type!r}. "
                "Expected 'child', 'node', or 'branch'."
            )

    # ------------------------------------------------------------------
    # Observer / action API (used by agent roles)
    # ------------------------------------------------------------------

    def observe(self, agent_id: str) -> dict:
        """Return the current state dict for the agent identified by *agent_id*.

        Returns an empty dict when no observer has been registered.
        """
        fn = self._observers.get(agent_id)
        return fn() if fn is not None else {}

    def act(self, agent_id: str, action: str, *args: Any, **kwargs: Any) -> None:
        """Invoke *action* for the agent identified by *agent_id*.

        Unknown agents or actions are silently ignored (a warning is logged).
        """
        fn = self._actions.get(agent_id, {}).get(action)
        if fn is not None:
            fn(*args, **kwargs)
        else:
            logger.warning(
                "No action %r registered for agent %r", action, agent_id
            )

    def has_action(self, agent_id: str, action: str) -> bool:
        """Return ``True`` if *agent_id* has the named *action* registered."""
        return action in self._actions.get(agent_id, {})

    # ------------------------------------------------------------------
    # Callback setters (mirroring Julia's module-level functions)
    # ------------------------------------------------------------------

    def set_on_branch_failure(self, callback: Callable[[Any], None]) -> None:
        """Set the callback invoked with ``(branch_id,)`` on each branch failure."""
        self._on_branch_failure = callback

    def set_on_node_failure(self, callback: Callable[[Any], None]) -> None:
        """Set the callback invoked with ``(node_id,)`` on each node failure."""
        self._on_node_failure = callback

    def set_on_custom_failure(self, callback: Callable[[Any], None]) -> None:
        """Set the callback invoked with ``(custom_id,)`` on each custom failure."""
        self._on_custom_failure = callback

    # ------------------------------------------------------------------
    # Failure scheduling and application
    # ------------------------------------------------------------------

    def schedule_failure(self, world, failure: Failure) -> None:
        """Schedule *failure* to trigger after ``failure.delay_s`` seconds.

        Stores the failure in a sorted list checked on each simulation step
        and registers a clock future via ``clock.sleep(delay_s)`` so that
        ``discrete_step_until``'s step-size determination finds an activity at
        the trigger time — equivalent to Julia's ``schedule(env, ...)`` task
        mechanism.
        """
        self._failures.append(failure)
        trigger_time = world.clock.time + failure.delay_s
        seq = self._failure_seq
        self._failure_seq += 1
        bisect.insort(self._scheduled_failures, (trigger_time, seq, failure))
        # Register a future in the clock to drive discrete_step_until stepping.
        # The future is not awaited by any task; it is resolved (and discarded)
        # when set_time(trigger_time) is called, which causes on_step to fire.
        world.clock.sleep(failure.delay_s)
        logger.info(
            "Scheduled failure: delay=%.3f s, trigger at t=%.3f",
            failure.delay_s,
            trigger_time,
        )

    def apply_failures(self, failures: list[Failure]) -> None:
        """Directly apply *failures* to the network without emitting events.

        Useful for setting up a degraded initial state before the simulation
        starts.
        """
        for failure in failures:
            for branch_id in failure.branch_ids:
                self._net.branch_by_id(branch_id).active = False
            for node_id in failure.node_ids:
                self._net.node_by_id(node_id).active = False
            if failure.custom is not None:
                failure.custom(self._net)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_failures(
        self, environment: Environment, failures: list[Failure]
    ) -> None:
        """Apply failures, invoke user callbacks, and broadcast global events."""
        for failure in failures:
            for branch_id in failure.branch_ids:
                self._net.branch_by_id(branch_id).active = False
                self._dirty = True
                self._on_branch_failure(branch_id)
                logger.info("BranchFailureEvent: branch_id=%s", branch_id)
                environment.emit_global_event(BranchFailureEvent(branch_id))

            for node_id in failure.node_ids:
                self._net.node_by_id(node_id).active = False
                self._dirty = True
                self._on_node_failure(node_id)
                logger.info("NodeFailureEvent: node_id=%s", node_id)
                environment.emit_global_event(NodeFailureEvent(node_id))

            if failure.custom is not None:
                failure.custom(self._net)
                self._dirty = True
                self._on_custom_failure(failure.custom_id)
                logger.info("CustomFailureEvent: custom_id=%s", failure.custom_id)
                environment.emit_global_event(CustomFailureEvent(failure.custom_id))

    def _install_child(self, aid: str, child_id: Any) -> None:
        """Wire child component observer and *regulate* action for *aid*."""

        def observer() -> dict:
            results_net = self._net_results.network
            child = results_net.child_by_id(child_id)
            node = results_net.node_by_id(child.node_id)
            return {**dict(node.model.values), **dict(child.model.values)}

        self._observers[aid] = observer

        def regulate(regulation_factor: float) -> None:
            self._net.child_by_id(child_id).model.regulation = regulation_factor
            self._dirty = True

        self._actions[aid] = {"regulate": regulate}

    def _install_node(self, aid: str, node_id: Any) -> None:
        """Wire node component observer and optional *regulate* action for *aid*."""

        def observer() -> dict:
            results_net = self._net_results.network
            return dict(results_net.node_by_id(node_id).model.values)

        self._observers[aid] = observer

        actions: dict[str, Callable] = {}
        if "regulation" in self._net.node_by_id(node_id).model.values:

            def regulate(regulation_factor: float) -> None:
                self._net.node_by_id(node_id).model.regulation = regulation_factor
                self._dirty = True

            actions["regulate"] = regulate

        self._actions[aid] = actions

    def _install_branch(self, aid: str, branch_id: Any) -> None:
        """Wire branch component observer, *switch* action, and optional *regulate* for *aid*."""

        def observer() -> dict:
            results_net = self._net_results.network
            return dict(results_net.branch_by_id(branch_id).model.values)

        self._observers[aid] = observer

        def switch() -> None:
            branch = self._net.branch_by_id(branch_id)
            branch.model.on_off = 0 if branch.model.on_off == 1 else 1
            self._dirty = True

        actions: dict[str, Callable] = {"switch": switch}

        if "regulation" in self._net.branch_by_id(branch_id).model.values:

            def regulate(regulation_factor: float) -> None:
                self._net.branch_by_id(branch_id).model.regulation = regulation_factor
                self._dirty = True

            actions["regulate"] = regulate

        self._actions[aid] = actions


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (mirror Julia's dispatch-based functions)
# ---------------------------------------------------------------------------


def schedule_failure(behavior: RestorationEnvironmentBehavior, world, failure: Failure) -> None:
    """Schedule *failure* on *behavior*.  Thin wrapper around :meth:`~RestorationEnvironmentBehavior.schedule_failure`."""
    behavior.schedule_failure(world, failure)


def apply_failures(net, failures: list[Failure]) -> None:
    """Apply *failures* directly to *net* without going through the behavior.

    Useful for deterministic scenario setup before the simulation starts.
    """
    for failure in failures:
        for branch_id in failure.branch_ids:
            net.branch_by_id(branch_id).active = False
        for node_id in failure.node_ids:
            net.node_by_id(node_id).active = False
        if failure.custom is not None:
            failure.custom(net)


def on_branch_failure(
    callback: Callable, behavior: RestorationEnvironmentBehavior
) -> None:
    """Register *callback* as the branch-failure hook on *behavior*."""
    behavior.set_on_branch_failure(callback)


def on_node_failure(
    callback: Callable, behavior: RestorationEnvironmentBehavior
) -> None:
    """Register *callback* as the node-failure hook on *behavior*."""
    behavior.set_on_node_failure(callback)


def on_custom_failure(
    callback: Callable, behavior: RestorationEnvironmentBehavior
) -> None:
    """Register *callback* as the custom-failure hook on *behavior*."""
    behavior.set_on_custom_failure(callback)


# ---------------------------------------------------------------------------
# Branch agent-ID convention
# ---------------------------------------------------------------------------


def create_branch_aid(branch_id: tuple) -> str:
    """Return the canonical agent-ID string for a branch.

    Convention: higher node ID first → ``"branch-{hi}-{lo}"``.  This ensures
    the branch agent ID is uniquely determined from either endpoint.

    Example::

        create_branch_aid((3, 7))  # "branch-7-3"
        create_branch_aid((7, 3))  # "branch-7-3"
    """
    a, b = branch_id[0], branch_id[1]
    hi, lo = (a, b) if a > b else (b, a)
    return f"branch-{hi}-{lo}"


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def topology_based_on_grid(
    monee_net,
    topology: Topology,
    world,
    *,
    include_childs: bool = False,
    include_cps: bool = False,
) -> None:
    """Populate *topology* nodes and edges from the monee network structure.

    Each monee node becomes a topology node; branches become edges.
    Agents must already be registered in *world* with their ``tid`` as the
    agent ID before calling this function.

    Parameters
    ----------
    monee_net:
        The monee Network whose topology to mirror.
    topology:
        The mango :class:`~mango.express.topology.Topology` to populate.
    world:
        The :class:`~mango.simulation.world.SimulationWorld` containing the
        registered agents.
    include_childs:
        When ``True``, child-device agents are placed in the same topology
        slot as their parent node agent.
    include_cps:
        When ``True``, coupling-point branches are included as topology edges.
    """
    # Map monee node ID → topology node ID (sequential ints assigned by mango)
    monee_to_topo: dict[Any, int] = {}

    for node in monee_net.nodes:
        agents = []
        if node.tid in world._agents:
            agents.append(world._agents[node.tid])
        if include_childs:
            for child in monee_net.childs_by_ids(node.child_ids):
                if child.tid in world._agents:
                    agents.append(world._agents[child.tid])
        topo_id = topology.add_node(*agents)
        monee_to_topo[node.id] = topo_id

    for branch in monee_net.branches:
        if not include_cps and branch.model.is_cp():
            continue
        from_topo = monee_to_topo.get(branch.from_node_id)
        to_topo = monee_to_topo.get(branch.to_node_id)
        if from_topo is None or to_topo is None:
            continue
        # physical-active AND switch on → NORMAL; otherwise INACTIVE
        state = (
            State.NORMAL
            if (branch.active and getattr(branch.model, "on_off", 1) == 1)
            else State.INACTIVE
        )
        topology.add_edge(from_topo, to_topo, state)


def topology_based_on_grid_groups(
    monee_net,
    topology: Topology,
    world,
    *,
    separate_sectors: list[str] | None = None,
    include_cps: bool = False,
    include_nodes: bool = False,
    include_childs: bool = True,
    include_branches: list[str] | None = None,
) -> None:
    """Populate *topology* using connected-component groups from the monee network.

    Each connected component of the network graph becomes a fully-connected
    topology cluster.  Optionally separate clusters are created per energy
    sector string (e.g. ``["electricity", "gas"]``).

    Parameters
    ----------
    monee_net:
        The monee Network.
    topology:
        The mango :class:`~mango.express.topology.Topology` to populate.
    world:
        The :class:`~mango.simulation.world.SimulationWorld`.
    separate_sectors:
        If given, independent clusters are built for each sector substring.
    include_cps:
        Add coupling-point components to topology clusters.
    include_nodes:
        Add network-node agents to topology clusters.
    include_childs:
        Add child-device agents to topology clusters.
    include_branches:
        List of branch model type-name substrings whose agents to include.
    """
    if include_branches is None:
        include_branches = []

    components = connected_components(monee_net)

    if separate_sectors is None:
        _topology_grid_groups_by_sector(
            components,
            monee_net,
            topology,
            world,
            sector=None,
            include_cps=include_cps,
            include_nodes=include_nodes,
            include_childs=include_childs,
            include_branches=include_branches,
        )
    else:
        for sector in separate_sectors:
            _topology_grid_groups_by_sector(
                components,
                monee_net,
                topology,
                world,
                sector=sector,
                include_cps=include_cps,
                include_nodes=include_nodes,
                include_childs=include_childs,
                include_branches=include_branches,
            )


def _topology_grid_groups_by_sector(
    components: list[set],
    monee_net,
    topology: Topology,
    world,
    sector: str | None,
    *,
    include_nodes: bool,
    include_childs: bool,
    include_cps: bool,
    include_branches: list[str],
) -> None:
    """Build topology clusters for each connected component, optionally filtering by sector."""
    for component in components:
        # Collect (node_id, [agent_ids]) pairs for this component
        id_list: list[tuple[Any, list[str]]] = []
        added: list[str] = []  # global dedup across nodes in this component

        for component_id in component:
            node = monee_net.node_by_id(component_id)

            # Filter by energy sector if requested
            if sector is not None:
                grid = node.grid
                grid_str = str(grid) if not isinstance(grid, str) else grid
                if isinstance(grid, list) or sector not in grid_str:
                    continue

            agent_ids: list[str] = []

            if include_nodes and node.tid in world._agents:
                agent_ids.append(node.tid)

            if include_childs:
                for child in monee_net.childs_by_ids(node.child_ids):
                    if child.tid in world._agents and child.tid not in added:
                        agent_ids.append(child.tid)

            if include_cps:
                for comp in monee_net.components_connected_to(node.id):
                    if comp.model.is_cp() and comp.tid not in added:
                        agent_ids.append(comp.tid)
                if node.model.is_cp() and node.tid not in added:
                    agent_ids.append(node.tid)

            for branch_type_substr in include_branches:
                for branch in monee_net.branches_connected_to(node.id):
                    model_type_name = type(branch.model).__name__
                    if branch_type_substr in model_type_name and branch.tid not in added:
                        agent_ids.append(branch.tid)

            id_list.append((node.id, agent_ids))
            added.extend(agent_ids)

        # Build topology nodes and edges for this component
        added_topo_ids: list[int] = []
        first_in_group = True

        for node_id, agent_ids in id_list:
            if not agent_ids:
                continue
            agents = [world._agents[aid] for aid in agent_ids if aid in world._agents]
            if not agents:
                continue

            nid = topology.add_node(*agents)

            if first_in_group and hasattr(topology, "set_characteristic"):
                topology.set_characteristic(nid, agents[0], "leader")
                first_in_group = False

            # Fully connect this new node to all previous nodes in the group
            for prev_nid in added_topo_ids:
                topology.add_edge(nid, prev_nid, State.NORMAL)

            added_topo_ids.append(nid)
