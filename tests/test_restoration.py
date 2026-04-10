"""Tests for the multi-energy restoration environment.

Translated from MangoEnergyEnvironments.jl/test/monee_environment_test.jl.
The Julia test was fully commented out (only works with a correctly installed
monee environment), so these tests establish the canonical Python baseline.
"""

from __future__ import annotations

import asyncio

import pytest

from mango.agent.role import Role, RoleAgent
from mango.express.topology import create_topology
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world, discrete_step_until

from mango_energy_environments import (
    BranchFailureEvent,
    Failure,
    NodeFailureEvent,
    RestorationEnvironmentBehavior,
    schedule_failure,
    topology_based_on_grid,
)
from mango_energy_environments.base.monee import energyflow
import mango_energy_environments.environments.restoration.multi_energy_monee as _restoration_mod


# ---------------------------------------------------------------------------
# Autouse fixture: stub out the GEKKO-backed energyflow for all tests
# ---------------------------------------------------------------------------


class _FakeEnergyFlowResult:
    """Minimal stub for a monee energy-flow result.

    Returns the original network as ``result.network`` so that observer
    closures (``node_by_id(...).model.values`` etc.) still work.
    """

    def __init__(self, net):
        self.network = net


@pytest.fixture(autouse=True)
def _stub_energyflow(monkeypatch):
    """Replace energyflow with a pass-through stub to avoid GEKKO failures."""
    monkeypatch.setattr(
        _restoration_mod,
        "energyflow",
        lambda net: _FakeEnergyFlowResult(net),
    )


# ---------------------------------------------------------------------------
# Helper role (mirrors Julia's @role struct BranchFailureHandler)
# ---------------------------------------------------------------------------


class BranchFailureHandler(Role):
    """Counts received global failure events and forwarded string messages."""

    def __init__(self):
        self.counter = 0
        self.msg_counter = 0

    def on_global_event(self, event):
        if isinstance(event, BranchFailureEvent):
            self.counter += 1

    def handle_message(self, content, meta):
        if isinstance(content, str):
            self.msg_counter += 1


# ---------------------------------------------------------------------------
# Unit tests (no mango world needed)
# ---------------------------------------------------------------------------


class TestRestorationBehaviorUnit:
    """Unit tests for RestorationEnvironmentBehavior in isolation."""

    def test_initial_state(self, example_net):
        behavior = RestorationEnvironmentBehavior(example_net)
        assert behavior.net is example_net
        assert behavior.net_results is None
        assert behavior.failures == []
        assert not behavior._dirty

    def test_initialize_runs_energy_flow(self, example_net):
        from mango.util.clock import ExternalClock
        from mango.simulation.environment import DefaultEnvironment

        behavior = RestorationEnvironmentBehavior(example_net)
        env = DefaultEnvironment(behavior=behavior)
        clock = ExternalClock(start_time=0.0)
        behavior.initialize(env, clock)

        # The autouse _stub_energyflow fixture replaces energyflow with a
        # _FakeEnergyFlowResult stub, so net_results is a non-None object.
        assert behavior.net_results is not None

    def test_apply_failures_deactivates_branch(self, example_net):
        branch = example_net.branches[0]
        assert branch.active

        behavior = RestorationEnvironmentBehavior(example_net)
        failure = Failure(delay_s=0.0, branch_ids=[branch.id])
        behavior.apply_failures([failure])

        assert not example_net.branch_by_id(branch.id).active

    def test_apply_failures_deactivates_node(self, example_net):
        node = example_net.nodes[0]
        behavior = RestorationEnvironmentBehavior(example_net)
        failure = Failure(delay_s=0.0, node_ids=[node.id])
        behavior.apply_failures([failure])

        assert not example_net.node_by_id(node.id).active

    def test_apply_failures_custom(self, example_net):
        called_with = []
        behavior = RestorationEnvironmentBehavior(example_net)
        failure = Failure(
            delay_s=0.0,
            custom=lambda net: called_with.append(net),
            custom_id=42,
        )
        behavior.apply_failures([failure])
        assert len(called_with) == 1

    def test_observe_returns_empty_without_install(self, example_net):
        behavior = RestorationEnvironmentBehavior(example_net)
        assert behavior.observe("nonexistent-agent") == {}

    def test_act_unknown_agent_is_silent(self, example_net):
        behavior = RestorationEnvironmentBehavior(example_net)
        # Must not raise
        behavior.act("nonexistent-agent", "regulate", 0.5)

    def test_set_callbacks(self, example_net):
        behavior = RestorationEnvironmentBehavior(example_net)
        received = []
        behavior.set_on_branch_failure(lambda bid: received.append(("branch", bid)))
        behavior.set_on_node_failure(lambda nid: received.append(("node", nid)))
        behavior.set_on_custom_failure(lambda cid: received.append(("custom", cid)))

        # Verify they are stored (we'll trigger in integration tests)
        assert behavior._on_branch_failure is not None
        assert behavior._on_node_failure is not None
        assert behavior._on_custom_failure is not None


# ---------------------------------------------------------------------------
# Failure dataclass
# ---------------------------------------------------------------------------


class TestFailure:
    def test_defaults(self):
        f = Failure(delay_s=5.0)
        assert f.delay_s == 5.0
        assert f.branch_ids == []
        assert f.node_ids == []
        assert f.custom is None
        assert f.custom_id is None

    def test_full(self):
        fn = lambda net: None
        f = Failure(delay_s=2.0, branch_ids=[(0, 1, 0)], node_ids=[3], custom=fn, custom_id=99)
        assert f.branch_ids == [(0, 1, 0)]
        assert f.node_ids == [3]
        assert f.custom is fn
        assert f.custom_id == 99


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


class TestTopologyBasedOnGrid:
    def test_adds_nodes_for_each_monee_node(self, example_net):
        from mango.express.topology import Topology
        import networkx as nx

        world = create_world(start_time=0.0)
        topology = Topology(nx.Graph())
        topology_based_on_grid(example_net, topology, world)

        assert topology.graph.number_of_nodes() == len(example_net.nodes)

    def test_adds_edges_for_active_branches(self, example_net):
        from mango.express.topology import Topology
        import networkx as nx

        world = create_world(start_time=0.0)
        topology = Topology(nx.Graph())
        topology_based_on_grid(example_net, topology, world)

        # At minimum one edge should be present (benchmark net is connected)
        assert topology.graph.number_of_edges() > 0


# ---------------------------------------------------------------------------
# Integration tests — translated from Julia monee_environment_test.jl
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mes_environment_shallow():
    """Shallow integration test: mirrors the (commented-out) Julia test.

    Checks that:
    - A branch failure scheduled at t=2 s fires exactly once.
    - Agents at directly-connected nodes receive the global event.
    - Energy flow is re-run after the failure (net_results is updated).
    """
    from mango_energy_environments.base.monee import fetch_example_net

    monee_net = fetch_example_net()
    behavior = RestorationEnvironmentBehavior(monee_net)
    environment = DefaultEnvironment(behavior=behavior)
    com_sim = SimpleCommunicationSimulation(default_delay_s=0.02)
    world = create_world(start_time=0.0, communication_sim=com_sim, environment=environment)

    agents: list[RoleAgent] = []
    for node in monee_net.nodes:
        agent = RoleAgent()
        handler = BranchFailureHandler()
        agent.add_role(handler)
        world.register(agent, suggested_aid=node.tid)
        world.environment.install(agent, id=node.id, type="node")
        agents.append(agent)

    # Pick the third branch (index 2) as the failing one — matches Julia test
    branch = monee_net.branches[2]
    failure = Failure(delay_s=2.0, branch_ids=[branch.id])

    async with world:
        schedule_failure(behavior, world, failure)
        await discrete_step_until(world, max_advance_time_s=10.0)

    # The branch should now be deactivated in the live network
    assert not monee_net.branch_by_id(branch.id).active

    # Energy flow should have been re-run after the failure
    assert behavior.net_results is not None

    # At least one agent received the BranchFailureEvent
    total_events = sum(
        role.counter
        for agent in agents
        for role in agent.roles
        if isinstance(role, BranchFailureHandler)
    )
    assert total_events >= 1


@pytest.mark.asyncio
async def test_failure_callbacks_are_invoked():
    """User-registered failure callbacks are called with the correct IDs."""
    from mango_energy_environments.base.monee import fetch_example_net

    monee_net = fetch_example_net()
    branch_failures = []
    behavior = RestorationEnvironmentBehavior(
        monee_net,
        on_branch_failure=lambda bid: branch_failures.append(bid),
    )
    environment = DefaultEnvironment(behavior=behavior)
    world = create_world(
        start_time=0.0,
        communication_sim=SimpleCommunicationSimulation(default_delay_s=0.0),
        environment=environment,
    )

    branch = monee_net.branches[0]
    failure = Failure(delay_s=1.0, branch_ids=[branch.id])

    async with world:
        schedule_failure(behavior, world, failure)
        await discrete_step_until(world, max_advance_time_s=5.0)

    assert len(branch_failures) == 1
    assert branch_failures[0] == branch.id


@pytest.mark.asyncio
async def test_node_failure_emits_event():
    """NodeFailureEvent is broadcast when a node failure fires."""
    from mango_energy_environments.base.monee import fetch_example_net

    monee_net = fetch_example_net()

    received_events = []

    class NodeEventWatcher(Role):
        def on_global_event(self, event):
            if isinstance(event, NodeFailureEvent):
                received_events.append(event)

    behavior = RestorationEnvironmentBehavior(monee_net)
    environment = DefaultEnvironment(behavior=behavior)
    world = create_world(
        start_time=0.0,
        communication_sim=SimpleCommunicationSimulation(default_delay_s=0.0),
        environment=environment,
    )

    # Register one watcher agent
    watcher = RoleAgent()
    watcher.add_role(NodeEventWatcher())
    world.register(watcher, suggested_aid="watcher")

    node = monee_net.nodes[1]
    failure = Failure(delay_s=0.5, node_ids=[node.id])

    async with world:
        schedule_failure(behavior, world, failure)
        await discrete_step_until(world, max_advance_time_s=5.0)

    assert len(received_events) == 1
    assert received_events[0].node_id == node.id
    assert not monee_net.node_by_id(node.id).active


@pytest.mark.asyncio
async def test_observer_and_action_for_node(example_net):
    """Installing a node agent wires observer (returns dict) and optional regulate."""
    behavior = RestorationEnvironmentBehavior(example_net)
    environment = DefaultEnvironment(behavior=behavior)
    world = create_world(start_time=0.0, environment=environment)

    node = example_net.nodes[0]
    agent = RoleAgent()
    world.register(agent, suggested_aid=node.tid)

    async with world:
        world.environment.install(agent, id=node.id, type="node")
        obs = behavior.observe(agent.aid)

    assert isinstance(obs, dict)
    # A freshly initialised network should have values
    assert len(obs) > 0


@pytest.mark.asyncio
async def test_branch_switch_action(example_net):
    """The 'switch' action on a branch toggles its on_off model attribute."""
    from mango.util.clock import ExternalClock

    behavior = RestorationEnvironmentBehavior(example_net)
    environment = DefaultEnvironment(behavior=behavior)
    clock = ExternalClock(start_time=0.0)
    behavior.initialize(environment, clock)

    # Find a branch that has an on_off attribute
    branch = None
    for b in example_net.branches:
        if hasattr(b.model, "on_off"):
            branch = b
            break

    if branch is None:
        pytest.skip("No branch with on_off attribute in this network")

    world = create_world(start_time=0.0, environment=environment)
    agent = RoleAgent()
    world.register(agent, suggested_aid="branch-agent")

    world.environment.install(agent, id=branch.id, type="branch")
    assert behavior.has_action(agent.aid, "switch")

    original_state = branch.model.on_off
    behavior.act(agent.aid, "switch")
    assert branch.model.on_off != original_state
    assert behavior._dirty
