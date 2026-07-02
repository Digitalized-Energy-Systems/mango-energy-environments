"""Unit tests for :class:`PyPSABehavior` and :func:`extract_timeseries`."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from mango import RoleAgent
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world, discrete_step_until

from mango_energy_environments.environments.scheduling import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    ComponentRef,
    PowerUpdateInfo,
    PyPSABehavior,
    calculate_initial_time,
    extract_timeseries,
    get_components_by_type,
    get_possible_components,
)


class TestComponentRef:
    def test_equality(self):
        assert ComponentRef(THERMAL, "g0") == ComponentRef(THERMAL, "g0")
        assert ComponentRef(THERMAL, "g0") != ComponentRef(LOAD, "g0")

    def test_unpack(self):
        et, cid = ComponentRef(THERMAL, "g0")
        assert et == THERMAL
        assert cid == "g0"

    def test_hashable(self):
        d = {ComponentRef(THERMAL, "g0"): 1}
        assert d[ComponentRef(THERMAL, "g0")] == 1


class TestPyPSABehaviorUnit:
    def test_get_components_by_type_splits_thermal_and_renewable(
        self, five_bus_pypsa_net
    ):
        net, ts, start = five_bus_pypsa_net
        behavior = PyPSABehavior(net=net, timeseries=ts, start_datetime=start)

        thermal = behavior.get_components_by_type([THERMAL])
        renewable = behavior.get_components_by_type([RENEWABLE])

        assert all(c.element_type == THERMAL for c in thermal)
        assert all(c.element_type == RENEWABLE for c in renewable)
        assert {c.component_id for c in thermal} == {"thermal0", "thermal1"}
        assert {c.component_id for c in renewable} == {"wind0"}

    def test_get_possible_components_returns_all_relevant(self, five_bus_pypsa_net):
        net, ts, start = five_bus_pypsa_net
        behavior = PyPSABehavior(
            net=net,
            timeseries=ts,
            start_datetime=start,
            relevant_types=[THERMAL, LOAD],
        )
        comps = behavior.get_possible_components()
        assert {c.element_type for c in comps} == {THERMAL, LOAD}

    def test_start_datetime_inferred_from_timeseries(self, pypsa_net_with_timeseries):
        net, g0, l0, ts, start = pypsa_net_with_timeseries
        behavior = PyPSABehavior(net=net, timeseries=ts)
        assert behavior.start_datetime == start

    def test_calculate_initial_time_wrapper(self, pypsa_net_with_timeseries):
        net, g0, l0, ts, start = pypsa_net_with_timeseries
        behavior = PyPSABehavior(net=net, timeseries=ts, start_datetime=start)
        assert calculate_initial_time(behavior) == start

    def test_get_components_by_type_wrapper(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        comps = get_components_by_type(behavior, [LOAD])
        assert ComponentRef(LOAD, l0) in comps

    def test_get_possible_components_wrapper(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net, relevant_types=[THERMAL])
        comps = get_possible_components(behavior)
        assert all(c.element_type == THERMAL for c in comps)

    def test_timeseries_key_normalisation(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        index = pd.date_range("2024-01-01", periods=1, freq="h")
        ts_ref = {ComponentRef(THERMAL, g0): pd.Series([1.0], index=index)}
        ts_tup = {(THERMAL, g0): pd.Series([1.0], index=index)}
        b_ref = PyPSABehavior(
            net=net, timeseries=ts_ref, start_datetime=datetime(2024, 1, 1)
        )
        b_tup = PyPSABehavior(
            net=net, timeseries=ts_tup, start_datetime=datetime(2024, 1, 1)
        )
        assert b_ref._timeseries.keys() == b_tup._timeseries.keys()


class TestObserversAndActions:
    async def test_statics_observer_returns_dict(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ComponentRef(THERMAL, g0))

        result = behavior.observe(agent.aid, "statics")
        assert isinstance(result, dict)
        assert "p_nom" in result

    async def test_active_power_observer(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ComponentRef(THERMAL, g0))

        assert behavior.observe(agent.aid, "active_power") == pytest.approx(5.0)

    async def test_max_active_power_observer(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ComponentRef(THERMAL, g0))

        assert behavior.observe(agent.aid, "max_active_power") == pytest.approx(10.0)

    async def test_regulate_action_for_thermal(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ComponentRef(THERMAL, g0))

        assert behavior.has_action(agent.aid, "regulate")
        behavior.act(agent.aid, "regulate", 3.0)
        assert net.generators.at[g0, "p_set"] == pytest.approx(3.0)

    async def test_no_regulate_action_for_load(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="load-agent")
        world.environment.install(agent, id=ComponentRef(LOAD, l0))

        assert not behavior.has_action(agent.aid, "regulate")


class TestTimeseriesScheduling:
    async def test_renewable_timeseries_fires_events(self, five_bus_pypsa_net):
        net, timeseries, start = five_bus_pypsa_net
        behavior = PyPSABehavior(
            net=net,
            timeseries=timeseries,
            start_datetime=start,
            relevant_types=[THERMAL, RENEWABLE, LOAD, STORAGE],
        )
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)

        events: list[PowerUpdateInfo] = []

        class Capture(RoleAgent):
            def on_agent_event(self, event):
                super().on_agent_event(event)
                if isinstance(event, PowerUpdateInfo):
                    events.append(event)

        agent = Capture()
        world.register(agent, suggested_aid="wind0")
        world.environment.install(agent, id=ComponentRef(RENEWABLE, "wind0"))

        async with world:
            await discrete_step_until(world, 3 * 24 * 3600.0)

        # 72 hourly points → 72 events.
        assert len(events) == 72


class TestExtractTimeseries:
    def test_no_timeseries_returns_empty(self):
        import pypsa

        net = pypsa.Network()
        net.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        net.add("Bus", "b0")
        net.add("Generator", "g0", bus="b0", carrier="gas", p_nom=10.0)
        assert extract_timeseries(net) == {}

    def test_p_max_pu_extracted_as_renewable(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=3, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Generator", "w0", bus="b0", carrier="wind", p_nom=10.0)
        net.generators_t.p_max_pu = pd.DataFrame({"w0": [0.5, 0.6, 0.7]}, index=snaps)
        ts = extract_timeseries(net)
        assert ComponentRef(RENEWABLE, "w0") in ts
        assert list(ts[ComponentRef(RENEWABLE, "w0")].values) == [0.5, 0.6, 0.7]

    def test_loads_t_extracted(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=2, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Load", "l0", bus="b0", p_set=3.0)
        net.loads_t.p_set = pd.DataFrame({"l0": [3.0, 4.0]}, index=snaps)
        ts = extract_timeseries(net)
        assert ComponentRef(LOAD, "l0") in ts

    def test_storage_units_t_extracted(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=2, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("StorageUnit", "s0", bus="b0", p_nom=5.0, max_hours=2.0)
        net.storage_units_t.p_set = pd.DataFrame({"s0": [0.0, 1.0]}, index=snaps)
        ts = extract_timeseries(net)
        assert ComponentRef(STORAGE, "s0") in ts


class TestPyPSABehaviorFromScenario:
    def test_from_scenario_carries_start_and_timeseries(self, five_bus_pypsa_net):
        net, timeseries, start = five_bus_pypsa_net
        scenario = SimpleNamespace(net=net, timeseries=timeseries, start=start)
        behavior = PyPSABehavior.from_scenario(scenario)

        assert behavior.net is scenario.net
        assert behavior.start_datetime == scenario.start

    def test_from_network_auto_extracts_timeseries(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=3, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Generator", "w0", bus="b0", carrier="wind", p_nom=10.0)
        net.generators_t.p_max_pu = pd.DataFrame({"w0": [0.5, 0.6, 0.7]}, index=snaps)

        behavior = PyPSABehavior.from_network(net)
        assert ComponentRef(RENEWABLE, "w0") in behavior._timeseries

    def test_from_network_auto_timeseries_disabled(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=3, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Generator", "w0", bus="b0", carrier="wind", p_nom=10.0)
        net.generators_t.p_max_pu = pd.DataFrame({"w0": [0.5, 0.6, 0.7]}, index=snaps)

        behavior = PyPSABehavior.from_network(
            net, auto_timeseries=False, start_datetime=datetime(2024, 1, 1)
        )
        assert behavior._timeseries == {}
