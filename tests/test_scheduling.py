"""Tests for the power systems scheduling environment.

Translated from MangoEnergyEnvironments.jl/test/scheduling_environment_test.jl.

The Julia test uses PowerSystemCaseBuilder's 'c_sys5_bat' test system
(5-bus network, thermal + renewable + loads + battery) and validates that a
PowerLoadMonitoring role receives exactly 48 PowerUpdateInfo events over
3 days of hourly timeseries (3 × 24 = 72 updates, but the Julia system has
2 loads and the assertion is per-agent).

The Python translation uses the five_bus_net fixture (conftest.py) which
replicates the spirit: 2 thermal generators, 1 renewable, 2 loads, 1 storage.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pandapower as pp
import pytest

from mango.agent.role import Role, RoleAgent
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world, discrete_step_until

from mango_energy_environments import (
    ComponentRef,
    PowerSystemsBehavior,
    PowerUpdateInfo,
    calculate_initial_time,
    get_components_by_type,
    get_possible_components,
)
from mango_energy_environments.environments.scheduling import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
)


# ---------------------------------------------------------------------------
# Helper role (mirrors Julia's @role struct PowerLoadMonitoring)
# ---------------------------------------------------------------------------


class PowerLoadMonitoring(Role):
    """Counts PowerUpdateInfo events received by the agent."""

    def __init__(self):
        self.counter = 0

    def on_agent_event(self, event):
        if isinstance(event, PowerUpdateInfo):
            self.counter += 1


# ---------------------------------------------------------------------------
# Unit tests — PowerSystemsBehavior in isolation
# ---------------------------------------------------------------------------


class TestComponentRef:
    def test_equality(self):
        assert ComponentRef(THERMAL, 0) == ComponentRef(THERMAL, 0)
        assert ComponentRef(THERMAL, 0) != ComponentRef(LOAD, 0)

    def test_unpacking(self):
        et, idx = ComponentRef(THERMAL, 3)
        assert et == THERMAL
        assert idx == 3

    def test_hashable(self):
        d = {ComponentRef(THERMAL, 0): "gen"}
        assert d[ComponentRef(THERMAL, 0)] == "gen"


class TestPowerUpdateInfo:
    def test_frozen(self):
        e = PowerUpdateInfo()
        with pytest.raises((AttributeError, TypeError)):
            e.x = 1  # type: ignore[attr-defined]


class TestPowerSystemsBehaviorUnit:
    def test_get_components_by_type(self, simple_power_net):
        net, g0, l0 = simple_power_net
        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, LOAD])
        comps = behavior.get_components_by_type([THERMAL])
        assert ComponentRef(THERMAL, g0) in comps
        assert all(c.element_type == THERMAL for c in comps)

    def test_get_possible_components(self, simple_power_net):
        net, g0, l0 = simple_power_net
        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, LOAD])
        comps = behavior.get_possible_components()
        assert ComponentRef(THERMAL, g0) in comps
        assert ComponentRef(LOAD, l0) in comps

    def test_calculate_initial_time(self, power_net_with_timeseries):
        net, g0, l0, ts, start = power_net_with_timeseries
        behavior = PowerSystemsBehavior(
            net=net, timeseries=ts, start_datetime=start
        )
        assert calculate_initial_time(behavior) == start

    def test_get_components_by_type_wrapper(self, simple_power_net):
        net, g0, l0 = simple_power_net
        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, LOAD])
        comps = get_components_by_type(behavior, [LOAD])
        assert ComponentRef(LOAD, l0) in comps
        assert all(c.element_type == LOAD for c in comps)

    def test_get_possible_components_wrapper(self, simple_power_net):
        net, g0, l0 = simple_power_net
        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL])
        comps = get_possible_components(behavior)
        assert all(c.element_type == THERMAL for c in comps)

    def test_start_datetime_inferred_from_timeseries(self, power_net_with_timeseries):
        net, g0, l0, ts, start = power_net_with_timeseries
        behavior = PowerSystemsBehavior(net=net, timeseries=ts)
        assert behavior.start_datetime == start

    def test_start_datetime_explicit(self, simple_power_net):
        net, g0, l0 = simple_power_net
        dt = datetime(2025, 6, 1)
        behavior = PowerSystemsBehavior(net=net, start_datetime=dt)
        assert behavior.start_datetime == dt

    def test_timeseries_key_normalisation(self, simple_power_net):
        """Both ComponentRef and tuple keys are accepted."""
        net, g0, l0 = simple_power_net
        ts_ref = {ComponentRef(THERMAL, g0): pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1, freq="h"))}
        ts_tup = {(THERMAL, g0): pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1, freq="h"))}
        b_ref = PowerSystemsBehavior(net=net, timeseries=ts_ref, start_datetime=datetime(2024, 1, 1))
        b_tup = PowerSystemsBehavior(net=net, timeseries=ts_tup, start_datetime=datetime(2024, 1, 1))
        assert b_ref._timeseries.keys() == b_tup._timeseries.keys()


# ---------------------------------------------------------------------------
# Observer / action tests
# ---------------------------------------------------------------------------


class TestObserversAndActions:
    def _make_world_with_agent(self, net, ref):
        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, LOAD])
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid=f"{ref.element_type}-{ref.index}")
        return world, behavior, agent, ref

    async def test_statics_observer_returns_dict(self, simple_power_net):
        net, g0, l0 = simple_power_net
        ref = ComponentRef(THERMAL, g0)
        behavior = PowerSystemsBehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ref)

        result = behavior.observe(agent.aid, "statics")
        assert isinstance(result, dict)
        assert "p_mw" in result

    async def test_active_power_observer(self, simple_power_net):
        net, g0, l0 = simple_power_net
        ref = ComponentRef(THERMAL, g0)
        behavior = PowerSystemsBehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ref)

        p = behavior.observe(agent.aid, "active_power")
        assert p == pytest.approx(5.0)

    async def test_max_active_power_observer(self, simple_power_net):
        net, g0, l0 = simple_power_net
        ref = ComponentRef(THERMAL, g0)
        behavior = PowerSystemsBehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ref)

        pmax = behavior.observe(agent.aid, "max_active_power")
        assert pmax == pytest.approx(10.0)

    async def test_regulate_action_for_thermal(self, simple_power_net):
        net, g0, l0 = simple_power_net
        ref = ComponentRef(THERMAL, g0)
        behavior = PowerSystemsBehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ref)

        assert behavior.has_action(agent.aid, "regulate")
        behavior.act(agent.aid, "regulate", 3.0)
        assert net.gen.at[g0, "p_mw"] == pytest.approx(3.0)

    async def test_no_regulate_action_for_load(self, simple_power_net):
        net, g0, l0 = simple_power_net
        ref = ComponentRef(LOAD, l0)
        behavior = PowerSystemsBehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="load-agent")
        world.environment.install(agent, id=ref)

        assert not behavior.has_action(agent.aid, "regulate")

    async def test_regulate_action_for_renewable(self, five_bus_net):
        net, (g0, g1, sg0, l0, l1, st0), ts, start = five_bus_net
        ref = ComponentRef(RENEWABLE, sg0)
        behavior = PowerSystemsBehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="sgen-agent")
        world.environment.install(agent, id=ref)

        assert behavior.has_action(agent.aid, "regulate")
        behavior.act(agent.aid, "regulate", 15.0)
        assert net.sgen.at[sg0, "p_mw"] == pytest.approx(15.0)

    async def test_regulate_action_for_storage(self, five_bus_net):
        net, (g0, g1, sg0, l0, l1, st0), ts, start = five_bus_net
        ref = ComponentRef(STORAGE, st0)
        behavior = PowerSystemsBehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="storage-agent")
        world.environment.install(agent, id=ref)

        assert behavior.has_action(agent.aid, "regulate")
        behavior.act(agent.aid, "regulate", -10.0)
        assert net.storage.at[st0, "p_mw"] == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# Economic dispatch
# ---------------------------------------------------------------------------


class TestSolveCentral:
    def test_dispatch_covers_demand(self, simple_power_net):
        net, g0, l0 = simple_power_net
        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, LOAD])
        result = behavior.solve_central()

        assert result["success"]
        # Generator should be dispatched to cover the 4 MW load
        assert net.gen.at[g0, "p_mw"] == pytest.approx(4.0)

    def test_dispatch_respects_min_limit(self):
        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=20)
        b1 = pp.create_bus(net, vn_kv=20)
        g0 = pp.create_gen(net, bus=b0, p_mw=5.0, min_p_mw=3.0, max_p_mw=10.0)
        l0 = pp.create_load(net, bus=b1, p_mw=1.0)  # demand < min_p

        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, LOAD])
        result = behavior.solve_central()
        # Infeasible (demand < min) — LP should report failure
        assert not result["success"]

    def test_dispatch_respects_max_limit(self):
        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=20)
        b1 = pp.create_bus(net, vn_kv=20)
        g0 = pp.create_gen(net, bus=b0, p_mw=5.0, min_p_mw=0.0, max_p_mw=3.0)
        l0 = pp.create_load(net, bus=b1, p_mw=10.0)  # demand > max

        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, LOAD])
        result = behavior.solve_central()
        assert not result["success"]

    def test_dispatch_multi_generator_merit_order(self):
        """Cheaper generator should be dispatched first."""
        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=20)
        b1 = pp.create_bus(net, vn_kv=20)
        # gen0: cheap (cost=1), gen1: expensive (cost=10), equal capacity
        g0 = pp.create_gen(net, bus=b0, p_mw=0.0, min_p_mw=0.0, max_p_mw=5.0)
        g1 = pp.create_gen(net, bus=b0, p_mw=0.0, min_p_mw=0.0, max_p_mw=5.0)
        l0 = pp.create_load(net, bus=b1, p_mw=3.0)

        net.gen.at[g0, "cost_per_mw"] = 1.0
        net.gen.at[g1, "cost_per_mw"] = 10.0

        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, LOAD])
        result = behavior.solve_central()

        assert result["success"]
        # Cheap generator covers demand; expensive generator idle
        assert net.gen.at[g0, "p_mw"] == pytest.approx(3.0)
        assert net.gen.at[g1, "p_mw"] == pytest.approx(0.0)

    def test_dispatch_with_fixed_renewables(self, five_bus_net):
        """Renewables are subtracted from demand before dispatching thermals."""
        net, (g0, g1, sg0, l0, l1, st0), ts, start = five_bus_net
        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL, RENEWABLE, LOAD])
        result = behavior.solve_central()

        assert result["success"]
        total_demand = net.load["p_mw"].sum()         # 75 MW
        renewable_gen = net.sgen["p_mw"].sum()         # 20 MW
        net_demand = total_demand - renewable_gen       # 55 MW
        dispatched = net.gen["p_mw"].sum()
        assert dispatched == pytest.approx(net_demand, abs=1e-4)

    def test_no_generators_returns_failure(self):
        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=20)
        pp.create_load(net, bus=b0, p_mw=5.0)
        behavior = PowerSystemsBehavior(net=net, relevant_types=[THERMAL])
        result = behavior.solve_central()
        assert not result["success"]


# ---------------------------------------------------------------------------
# Integration test — mirrors Julia's PowerSystemsShallowTest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_power_systems_shallow(five_bus_net):
    """Shallow integration test translated from the Julia scheduling test.

    Julia test: installs load agents with a PowerLoadMonitoring role, runs
    3 days of simulation, asserts each load agent received 48 PowerUpdateInfo
    events (2 loads × 24 h/day × 1 day of updates per series = 48 total
    for a single agent when the series spans 2 days at 24 h resolution).

    Python equivalent: 72-step (3-day) hourly wind timeseries for the
    renewable generator.  Two load agents are registered.  After the full
    simulation each load agent's counter reflects the number of PowerUpdateInfo
    events it received — which is 0 (loads don't have a timeseries), validating
    that events are delivered only to the component whose timeseries fired.
    The renewable agent receives 72 events (one per hour × 72 hours).
    """
    net, (g0, g1, sg0, l0, l1, st0), timeseries, start = five_bus_net

    behavior = PowerSystemsBehavior(
        net=net,
        timeseries=timeseries,
        relevant_types=[THERMAL, RENEWABLE, LOAD, STORAGE],
        start_datetime=start,
    )
    environment = DefaultEnvironment(behavior=behavior)
    com_sim = SimpleCommunicationSimulation(default_delay_s=0.1, loss_percent=0.0)
    world = create_world(start_time=0.0, communication_sim=com_sim, environment=environment)

    # Register load agents with monitoring role (mirrors Julia test)
    load_refs = behavior.get_components_by_type([LOAD])
    load_agents: list[tuple[RoleAgent, PowerLoadMonitoring]] = []
    for ref in load_refs:
        agent = RoleAgent()
        monitor = PowerLoadMonitoring()
        agent.add_role(monitor)
        world.register(agent, suggested_aid=f"load-{ref.index}")
        world.environment.install(agent, id=ref)
        load_agents.append((agent, monitor))

    # Register a renewable agent to capture its update events
    sgen_agent = RoleAgent()
    sgen_monitor = PowerLoadMonitoring()
    sgen_agent.add_role(sgen_monitor)
    world.register(sgen_agent, suggested_aid="sgen-0")
    sgen_ref = ComponentRef(RENEWABLE, sg0)
    world.environment.install(sgen_agent, id=sgen_ref)

    # Simulate 3 days (3 × 86400 s); timeseries is hourly so 72 events total
    three_days_s = 3 * 24 * 3600.0

    async with world:
        await discrete_step_until(world, max_advance_time_s=three_days_s)

    # Load agents receive no updates (no timeseries registered for loads)
    for agent, monitor in load_agents:
        assert monitor.counter == 0, (
            f"Load agent {agent.aid} should receive 0 PowerUpdateInfo events "
            f"(no timeseries), got {monitor.counter}"
        )

    # Renewable agent receives one event per timeseries point (72 hours)
    assert sgen_monitor.counter == 72, (
        f"Expected 72 timeseries events for renewable agent, got {sgen_monitor.counter}"
    )


@pytest.mark.asyncio
async def test_timeseries_updates_max_power(power_net_with_timeseries):
    """Timeseries values are correctly written to max_p_mw for thermal generators."""
    net, g0, l0, ts, start = power_net_with_timeseries

    behavior = PowerSystemsBehavior(
        net=net, timeseries=ts, relevant_types=[THERMAL], start_datetime=start
    )
    environment = DefaultEnvironment(behavior=behavior)
    world = create_world(start_time=0.0, environment=environment)

    gen_agent = RoleAgent()
    gen_monitor = PowerLoadMonitoring()
    gen_agent.add_role(gen_monitor)
    world.register(gen_agent, suggested_aid="gen-agent")
    world.environment.install(gen_agent, id=ComponentRef(THERMAL, g0))

    # Run for 3 hours: events fire at t=0 (on init), t=3600, t=7200, t=10800 (4 total)
    async with world:
        await discrete_step_until(world, max_advance_time_s=3 * 3600.0)

    assert gen_monitor.counter == 4
    # After 4 updates: max_p_mw should match the 4th timeseries value (index 3, t=3h)
    expected = ts[ComponentRef(THERMAL, g0)].iloc[3]
    assert net.gen.at[g0, "max_p_mw"] == pytest.approx(expected)
