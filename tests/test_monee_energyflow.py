"""Tests that run the real (un-stubbed) monee energy flow.

``test_restoration.py`` stubs ``energyflow`` for every test, which hid that
``base.monee.energyflow`` passed ``solver="gurobi"`` — a string monee >= 1.1
rejects (``AttributeError: 'str' object has no attribute 'solve'``).
"""

from __future__ import annotations

import monee
from mango.agent.role import RoleAgent
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world, discrete_step_until

from mango_energy_environments import (
    Failure,
    RestorationEnvironmentBehavior,
    energyflow,
    schedule_failure,
)


class TestEnergyflow:
    def test_default_solver_returns_result_network(self, example_net):
        result = energyflow(example_net)

        assert result.network is not None
        assert len(result.network.nodes) > 0

    def test_explicit_solver_instance_is_used(self, example_net):
        result = energyflow(example_net, solver=monee.solver.GEKKOSolver())

        assert result.network is not None


class TestRestorationWorldWithRealEnergyflow:
    async def test_world_runs_through_branch_failure(self, example_net):
        behavior = RestorationEnvironmentBehavior(example_net)
        world = create_world(
            start_time=0.0,
            communication_sim=SimpleCommunicationSimulation(default_delay_s=0.02),
            environment=DefaultEnvironment(behavior=behavior),
        )
        for node in example_net.nodes:
            agent = RoleAgent()
            world.register(agent, suggested_aid=node.tid)
            world.environment.install(agent, id=node.id, type="node")

        branch = example_net.branches[2]
        async with world:
            schedule_failure(
                behavior, world, Failure(delay_s=2.0, branch_ids=[branch.id])
            )
            await discrete_step_until(world, max_advance_time_s=10.0)

        assert not behavior.net.branch_by_id(branch.id).active
