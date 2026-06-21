"""Multi-energy system restoration environment."""

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

from monee.model.child import ExtHydrGrid

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
    "topology_based_on_sector_grid",
]


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
        energy_flow_cooldown_s: float = 0.1,
        energy_flow_max_acts: int = 0,
    ) -> None:
        self._net = net
        self._net_results = None
        self._failures: list[Failure] = []
        self._on_branch_failure = on_branch_failure
        self._on_node_failure = on_node_failure
        self._on_custom_failure = on_custom_failure
        self._dirty: bool = False
        
        self._energy_flow_cooldown_s: float = float(energy_flow_cooldown_s)
        self._last_energy_flow_t: float = float("-inf")
        self._energy_flow_max_acts: int = int(energy_flow_max_acts)
        self._acts_since_solve: int = 0

        self._scheduled_failures: list[tuple[float, int, Failure]] = []
        self._failure_seq: int = 0

        self._observers: dict[str, Callable[[], dict]] = {}
        self._actions: dict[str, dict[str, Callable]] = {}

    @property
    def net(self):
        return self._net

    @property
    def net_results(self):
        return self._net_results

    @property
    def failures(self) -> list[Failure]:
        return self._failures

    def initialize(self, environment: Environment, clock: Clock) -> None:
        logger.debug("RestorationEnvironmentBehavior: running initial energy flow")
        self._net_results = energyflow(self._net)
        self._last_energy_flow_t = clock.time

    @staticmethod
    def _accept_or_keep(prev, candidate):
        """Return *candidate* when the solve succeeded, otherwise keep
        *prev*.  monee's ``SolverResult`` exposes ``success`` (False on
        infeasible / non-OK termination); pyomo's ``load_solutions=True``
        has already pushed a witness / partial solution onto every Var
        by then, so accepting that result would feed garbage to the
        ``observe()`` calls until the next successful solve.  Falling
        back to the last feasible result is the conservative choice
        and matches the energy_flow_cooldown contract (observers may
        see a slightly stale state, never an inconsistent one).
        """
        if prev is None:
            return candidate
        if getattr(candidate, "success", True):
            return candidate
        logger.warning(
            "energyflow infeasible — keeping previous net_results to avoid "
            "propagating an inadmissible witness solution to observers."
        )
        return prev

    def flush_energy_flow(self) -> None:
        """Force an immediate energy-flow recompute, bypassing the
        cooldown.  Use at end-of-simulation (or any other measurement
        boundary) so observers read post-agent-action state rather than
        a stale ``_net_results`` cached from a pre-cooldown solve.
        """
        logger.debug("RestorationEnvironmentBehavior: forced energy-flow flush")
        self._net_results = self._accept_or_keep(
            self._net_results, energyflow(self._net)
        )
        self._acts_since_solve = 0
        self._dirty = False

    def on_step(
        self, environment: Environment, clock: Clock, step_size_s: float
    ) -> None:
        # on_step is called before clock.set_time advances the clock, so use
        # clock.time + step_size_s as the effective end-of-step deadline.
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
            since_last = clock.time - self._last_energy_flow_t
            acts_over = (
                self._energy_flow_max_acts > 0
                and self._acts_since_solve >= self._energy_flow_max_acts
            )
            if since_last < self._energy_flow_cooldown_s and not acts_over:
                return
            
            logger.debug(
                "RestorationEnvironmentBehavior: recomputing energy flow "
                "(t=%.3f, dt=%.3f, since_last=%.3f)",
                clock.time,
                step_size_s,
                since_last,
            )
            self._net_results = self._accept_or_keep(
                self._net_results, energyflow(self._net)
            )
            self._last_energy_flow_t = clock.time
            self._acts_since_solve = 0
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

    def observe(self, agent_id: str) -> dict:
        """Return the current state dict for the agent identified by *agent_id*.

        Returns an empty dict when no observer has been registered.
        """
        fn = self._observers.get(agent_id)
        return fn() if fn is not None else {}

    def act(self, agent_id: str, action: str, *args: Any, **kwargs: Any) -> None:
        """Invoke *action* for the agent identified by *agent_id*."""
        fn = self._actions.get(agent_id, {}).get(action)
        if fn is not None:
            fn(*args, **kwargs)
            if action in ("regulate", "set_q", "set_pressure"):
                self._acts_since_solve += 1
        else:
            logger.warning(
                "No action %r registered for agent %r", action, agent_id
            )

    def has_action(self, agent_id: str, action: str) -> bool:
        """Return ``True`` if *agent_id* has the named *action* registered."""
        return action in self._actions.get(agent_id, {})

    def set_on_branch_failure(self, callback: Callable[[Any], None]) -> None:
        self._on_branch_failure = callback

    def set_on_node_failure(self, callback: Callable[[Any], None]) -> None:
        self._on_node_failure = callback

    def set_on_custom_failure(self, callback: Callable[[Any], None]) -> None:
        self._on_custom_failure = callback

    def schedule_failure(self, world, failure: Failure) -> None:
        """Schedule *failure* to trigger after ``failure.delay_s`` seconds."""
        self._failures.append(failure)
        trigger_time = world.clock.time + failure.delay_s
        seq = self._failure_seq
        self._failure_seq += 1
        bisect.insort(self._scheduled_failures, (trigger_time, seq, failure))
        # Register an orphaned clock future so discrete_step_until advances to
        # the trigger time.  The future is resolved by set_time and discarded.
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

    def _handle_failures(
        self, environment: Environment, failures: list[Failure]
    ) -> None:
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
        def observer() -> dict:
            results_net = self._net_results.network
            child = results_net.child_by_id(child_id)
            node = results_net.node_by_id(child.node_id)
            return {**dict(node.model.values), **dict(child.model.values)}

        self._observers[aid] = observer

        def regulate(regulation_factor: float) -> None:
            self._net.child_by_id(child_id).model.regulation = regulation_factor
            self._dirty = True

        actions: dict[str, Callable] = {"regulate": regulate}

        # Reactive-power dispatch: only present on children whose model
        # carries a ``q_mvar`` attribute (PowerGenerator / PowerLoad /
        # ExtPowerGrid).  Local Q-V droop roles in scare drive this knob
        # at the inverter timescale, independent of the LP-level
        # ``regulation`` setpoint that scales active power.
        if hasattr(self._net.child_by_id(child_id).model, "q_mvar"):

            def set_q(q_mvar_value: float) -> None:
                model = self._net.child_by_id(child_id).model
                # ``q_mvar`` may be a Var (ExtPowerGrid) or a plain
                # scalar (PowerGenerator).  Setting ``.value`` works for
                # the former; direct attribute assignment for the latter.
                if hasattr(model.q_mvar, "value") and not isinstance(model.q_mvar, (int, float)):
                    model.q_mvar.value = float(q_mvar_value)
                else:
                    model.q_mvar = float(q_mvar_value)
                self._dirty = True

            actions["set_q"] = set_q

        if isinstance(self._net.child_by_id(child_id).model, ExtHydrGrid):

            def set_pressure(pressure_pu_value: float) -> None:
                self._net.child_by_id(child_id).model.pressure_pu = float(
                    pressure_pu_value
                )
                self._dirty = True

            actions["set_pressure"] = set_pressure

        self._actions[aid] = actions

    def _install_node(self, aid: str, node_id: Any) -> None:
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


def schedule_failure(behavior: RestorationEnvironmentBehavior, world, failure: Failure) -> None:
    """Schedule *failure* on *behavior*."""
    behavior.schedule_failure(world, failure)


def apply_failures(net, failures: list[Failure]) -> None:
    """Apply *failures* directly to *net* without going through the behavior."""
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
    behavior.set_on_branch_failure(callback)


def on_node_failure(
    callback: Callable, behavior: RestorationEnvironmentBehavior
) -> None:
    behavior.set_on_node_failure(callback)


def on_custom_failure(
    callback: Callable, behavior: RestorationEnvironmentBehavior
) -> None:
    behavior.set_on_custom_failure(callback)


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

    Each connected component becomes a fully-connected topology cluster.
    Optionally separate clusters are created per energy sector string
    (e.g. ``["electricity", "gas"]``).

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
    for component in components:
        id_list: list[tuple[Any, list[str]]] = []
        added: list[str] = []

        for component_id in component:
            node = monee_net.node_by_id(component_id)

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

            for prev_nid in added_topo_ids:
                topology.add_edge(nid, prev_nid, State.NORMAL)

            added_topo_ids.append(nid)


def topology_based_on_sector_grid(
    monee_net,
    topology: Topology,
    world,
    *,
    sector: str,
    include_nodes: bool = False,
    include_childs: bool = True,
    include_branches: list[str] | None = None,
) -> None:
    """Populate *topology* with the physical subgraph of a single energy sector.

    Each monee node whose grid string matches *sector* becomes a topology
    node holding the node-agent (when *include_nodes* is true), all
    child-device agents at that node (when *include_childs* is true), and
    all branch-device agents matching *include_branches* (typically
    ``["HeatExchanger"]`` so heat exchangers appear as point-devices in
    the heat sector).

    Edges are added for same-sector branches that are **not** coupling
    points and whose model name does not contain any of the
    *include_branches* substrings — those branches represent point-devices
    rather than network connections.

    The resulting topology is the natural communication overlay for
    sector-local self-organisation (e.g. label-propagation community
    formation), preserving physical adjacency without merging sectors at
    coupling points.

    Parameters
    ----------
    monee_net:
        The monee Network.
    topology:
        The mango :class:`~mango.express.topology.Topology` to populate.
    world:
        The :class:`~mango.simulation.world.SimulationWorld` containing
        the registered agents.
    sector:
        Substring matched against the node's grid object's string
        representation (e.g. ``"power"``, ``"gas"``, ``"water"``).
    include_nodes:
        If true, the node-agent itself (when registered) is attached to
        its topology node.  Defaults to false because energy-balance
        roles are typically installed only on child / branch agents.
    include_childs:
        If true, all child-device agents at the node are attached.
        Defaults to true.
    include_branches:
        Branch model-name substrings whose agents are attached to their
        incident node as point-devices.  Edges spanning these branches
        are *not* added.
    """
    if include_branches is None:
        include_branches = []

    monee_to_topo: dict[Any, int] = {}
    added_branch_tids: set = set()

    def _matches(node) -> bool:
        grid = node.grid
        if isinstance(grid, list):
            return False
        grid_str = str(grid) if not isinstance(grid, str) else grid
        return sector in grid_str

    def _is_point_device(branch) -> bool:
        type_name = type(branch.model).__name__
        return any(sub in type_name for sub in include_branches)

    for node in monee_net.nodes:
        if not _matches(node):
            continue

        agents = []
        if include_nodes and node.tid in world._agents:
            agents.append(world._agents[node.tid])
        if include_childs:
            for child in monee_net.childs_by_ids(node.child_ids):
                if child.tid in world._agents:
                    agents.append(world._agents[child.tid])
        for branch in monee_net.branches_connected_to(node.id):
            if not _is_point_device(branch):
                continue
            if branch.tid in added_branch_tids:
                continue
            if branch.tid in world._agents:
                agents.append(world._agents[branch.tid])
                added_branch_tids.add(branch.tid)

        # Always add the node — even agentless transit nodes are needed
        # to keep the physical graph connected for downstream community
        # detection.  Empty-agent nodes get filtered later when groups
        # are formed.
        topo_id = topology.add_node(*agents)
        monee_to_topo[node.id] = topo_id

    for branch in monee_net.branches:
        if branch.model.is_cp():
            continue
        if _is_point_device(branch):
            continue
        from_topo = monee_to_topo.get(branch.from_node_id)
        to_topo = monee_to_topo.get(branch.to_node_id)
        if from_topo is None or to_topo is None or from_topo == to_topo:
            continue
        if topology.graph.has_edge(from_topo, to_topo):
            continue
        state = (
            State.NORMAL
            if (branch.active and getattr(branch.model, "on_off", 1) == 1)
            else State.INACTIVE
        )
        topology.add_edge(from_topo, to_topo, state)
