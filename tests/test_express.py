"""Tests for the high-level convenience API in ``express.py``.

Covers the world factories (``create_restoration_world`` and friends) and
``enable_poisson_com_for_monee`` — previously untested despite being the
package's advertised quick-start API.
"""

from __future__ import annotations

import pytest
from mango.agent.role import RoleAgent
from mango.simulation.communication import (
    DelayProviderCommunicationSimulation,
    SimpleCommunicationSimulation,
)

from mango_energy_environments import (
    RestorationEnvironmentBehavior,
    create_cigre_benchmark_restoration_world,
    create_restoration_world,
    create_small_benchmark_restoration_world,
    enable_poisson_com_for_monee,
)
from mango_energy_environments.express import _poisson_sample


class TestCreateRestorationWorld:
    """create_restoration_world builds a world wired to the given net."""

    def test_behavior_wraps_the_given_net(self, example_net):
        world = create_restoration_world(example_net, with_communication=False)
        behavior = world.environment.behavior
        assert isinstance(behavior, RestorationEnvironmentBehavior)
        assert behavior.net is example_net

    def test_with_communication_true_installs_delay_provider(self, example_net):
        world = create_restoration_world(example_net, with_communication=True)
        assert isinstance(world.communication_sim, DelayProviderCommunicationSimulation)

    def test_with_communication_false_keeps_static_delay(self, example_net):
        world = create_restoration_world(example_net, with_communication=False)
        assert isinstance(world.communication_sim, SimpleCommunicationSimulation)

    def test_static_delay_s_is_applied_without_communication(self, example_net):
        world = create_restoration_world(
            example_net, with_communication=False, static_delay_s=0.5
        )
        assert world.communication_sim.default_delay_s == 0.5


class TestBenchmarkWorldFactories:
    """The small and CIGRE benchmark factories build valid worlds end-to-end."""

    def test_small_benchmark_world_has_restoration_behavior(self):
        world = create_small_benchmark_restoration_world(with_communication=False)
        assert isinstance(world.environment.behavior, RestorationEnvironmentBehavior)
        assert len(world.environment.behavior.net.nodes) > 0

    @pytest.mark.xfail(
        reason=(
            "fetch_cigre_net() is broken with the currently installed "
            "pandapower/monee versions: monee.io.from_pandapower.from_pandapower_net "
            "calls pandapower.converter.to_mpc, which no longer exists on the "
            "installed pandapower. Pre-existing environment incompatibility, "
            "not a regression from this test."
        ),
        raises=AttributeError,
        strict=False,
    )
    def test_cigre_benchmark_world_has_restoration_behavior(self):
        world = create_cigre_benchmark_restoration_world(with_communication=False)
        assert isinstance(world.environment.behavior, RestorationEnvironmentBehavior)
        assert len(world.environment.behavior.net.nodes) > 0

    @pytest.mark.xfail(
        reason="fetch_cigre_net() is broken; see test_cigre_benchmark_world_has_restoration_behavior.",
        raises=AttributeError,
        strict=False,
    )
    def test_small_and_cigre_nets_differ_in_size(self):
        """Sanity check that the two benchmark networks are actually different."""
        small_world = create_small_benchmark_restoration_world(with_communication=False)
        cigre_world = create_cigre_benchmark_restoration_world(with_communication=False)
        small_n = len(small_world.environment.behavior.net.nodes)
        cigre_n = len(cigre_world.environment.behavior.net.nodes)
        assert small_n != cigre_n


class TestEnablePoissonComForMonee:
    """enable_poisson_com_for_monee replaces the world's communication_sim."""

    def _world_with_node_agents(self, example_net):
        world = create_restoration_world(example_net, with_communication=False)
        for node in example_net.nodes:
            agent = RoleAgent()
            world.register(agent, suggested_aid=node.tid)
        return world

    async def test_installs_delay_provider_communication_sim(self, example_net):
        world = self._world_with_node_agents(example_net)
        enable_poisson_com_for_monee(world, example_net)
        assert isinstance(world.communication_sim, DelayProviderCommunicationSimulation)

    async def test_default_delay_provider_is_non_negative(self, example_net):
        world = self._world_with_node_agents(example_net)
        enable_poisson_com_for_monee(world, example_net, base_delay_per_message=5.0)
        assert world.communication_sim.default_delay_s_provider() >= 0.0

    async def test_directed_edge_delays_cover_all_agent_pairs(self, example_net):
        world = self._world_with_node_agents(example_net)
        enable_poisson_com_for_monee(world, example_net, base_delay_per_message=5.0)
        aids = list(world._agents.keys())
        edge_dict = world.communication_sim.delay_s_directed_edge_dict
        for sender in aids:
            for receiver in aids:
                if sender != receiver:
                    assert (sender, receiver) in edge_dict
                    assert edge_dict[(sender, receiver)]() >= 0.0

    def test_no_agents_registered_still_succeeds(self, example_net):
        """With no agents installed, the topology graph is empty but the
        function must not raise."""
        world = create_restoration_world(example_net, with_communication=False)
        enable_poisson_com_for_monee(world, example_net)
        assert isinstance(world.communication_sim, DelayProviderCommunicationSimulation)


class TestPoissonSample:
    """_poisson_sample: internal helper backing the delay providers."""

    def test_zero_lambda_returns_zero(self):
        assert _poisson_sample(0.0) == 0.0

    def test_positive_lambda_returns_non_negative(self):
        for _ in range(50):
            assert _poisson_sample(10.0) >= 0.0

    def test_negative_lambda_returns_zero(self):
        assert _poisson_sample(-5.0) == 0.0


class TestReExports:
    """express.py re-exports the base.monee helpers advertised in __all__."""

    def test_monee_helpers_are_importable(self):
        from mango_energy_environments.express import (
            calc_general_resilience_performance,
            connected_components,
            edge_centrality,
            energyflow,
            fetch_cigre_net,
            fetch_example_net,
            lower,
            solve_load_shedding_optimization,
            solve_load_shedding_optimization_relaxed,
            upper,
        )

        assert callable(energyflow)
        assert callable(upper)
        assert callable(lower)
        assert callable(edge_centrality)
        assert callable(connected_components)
        assert callable(fetch_example_net)
        assert callable(fetch_cigre_net)
        assert callable(solve_load_shedding_optimization)
        assert callable(solve_load_shedding_optimization_relaxed)
        assert callable(calc_general_resilience_performance)
