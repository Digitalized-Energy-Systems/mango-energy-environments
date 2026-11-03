import numpy as np
import pandas as pd
import pytest
from mango import RoleAgent
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world

from mango_energy_environments.environments.copper_plate import CopperPlateBehavior
from mango_energy_environments.network import (
    LOAD,
    RENEWABLE,
    THERMAL,
    Component,
    Network,
    connection_graph,
)

T = 3


def flat(v):
    return np.full(T, float(v))


@pytest.fixture
def network():
    return Network(
        time_index=pd.date_range("2024-01-01", periods=T, freq="h"),
        components={
            "g0": Component(THERMAL, "b0", flat(0), flat(10), cost=(20.0, 0.1)),
            "l0": Component(LOAD, "b0", flat(4), np.array([4.0, 6.0, 8.0])),
            "w1": Component(RENEWABLE, "b1", flat(0), np.array([1.0, 2.0, 3.0])),
            "g2": Component(THERMAL, "b2", flat(0), flat(5)),
        },
        lines=[("b0", "b1")],
    )


def test_connection_graph(network):
    g = connection_graph(network)
    assert set(map(frozenset, g.edges)) == {
        frozenset(e) for e in [("g0", "l0"), ("g0", "w1"), ("l0", "w1")]
    }
    assert "g2" in g and g.degree("g2") == 0  # bus b2 has no line


async def test_observe_act_balance_in_world(network):
    behavior = CopperPlateBehavior(network)
    world = create_world(
        start_time=0.0, environment=DefaultEnvironment(behavior=behavior)
    )
    gen = world.register(RoleAgent(), suggested_aid="gen", id="g0")
    load = world.register(RoleAgent(), suggested_aid="load", id="l0")
    world.register(RoleAgent(), suggested_aid="other")  # no component: ignored

    assert behavior.observe(gen.aid, "statics") == {
        "kind": THERMAL,
        "bus": "b0",
        "cost": (20.0, 0.1),
    }
    np.testing.assert_array_equal(behavior.observe(load.aid, "demand"), [4.0, 6.0, 8.0])

    behavior.act(gen.aid, "set_schedule", [4.0, 5.0, 9.0])
    np.testing.assert_array_equal(behavior.balance(), [0.0, -1.0, 1.0])

    with pytest.raises(ValueError):
        behavior.act(gen.aid, "set_schedule", [1.0])
    with pytest.raises(KeyError):
        world.register(RoleAgent(), suggested_aid="bad", id="nope")
