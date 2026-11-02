import numpy as np
import pandas as pd
import pytest
from mango import RoleAgent
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world

from mango_energy_environments.environments.copper_plate import (
    Component,
    CopperPlateBehavior,
    CopperPlateNetwork,
    connection_graph,
)
from mango_energy_environments.environments.scheduling import (
    LOAD,
    RENEWABLE,
    THERMAL,
)

T = 3


def flat(v):
    return np.full(T, float(v))


@pytest.fixture
def network():
    return CopperPlateNetwork(
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


def test_from_pypsa(caplog):
    pypsa = pytest.importorskip("pypsa")
    from mango_energy_environments.environments.copper_plate import from_pypsa
    from mango_energy_environments.environments.scheduling import STORAGE

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=T, freq="h"))
    for b in ("b0", "b1", "b2"):
        n.add("Bus", b)
    n.add("Line", "l01", bus0="b0", bus1="b1", x=0.1, s_nom=100)
    n.add("Link", "k12", bus0="b1", bus1="b2", p_nom=50)
    n.add(
        "Generator",
        "coal",
        bus="b0",
        p_nom=10,
        marginal_cost=20,
        marginal_cost_quadratic=0.1,
    )
    n.add("Generator", "bio", bus="b0", p_nom=4, marginal_cost=15, carrier="biomass")
    n.add(
        "Generator", "pv", bus="b1", p_nom=5, p_max_pu=[0.0, 0.5, 1.0], carrier="solar"
    )
    n.add("Load", "d", bus="b1", p_set=[1.0, 2.0, 3.0])
    n.add("StorageUnit", "bat", bus="b2", p_nom=2, max_hours=2, p_min_pu=-1)
    n.add("Store", "tank", bus="b2")

    cp = from_pypsa(n)

    kinds = {cid: c.kind for cid, c in cp.components.items()}
    assert kinds == {
        "coal": THERMAL,
        "bio": THERMAL,
        "pv": RENEWABLE,
        "d": LOAD,
        "bat": STORAGE,
    }
    np.testing.assert_array_equal(cp.components["pv"].p_max, [0.0, 2.5, 5.0])
    np.testing.assert_array_equal(cp.components["coal"].p_max, flat(10))
    np.testing.assert_array_equal(cp.components["d"].p_max, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(cp.components["bat"].p_min, flat(-2))
    assert cp.components["coal"].cost == (20.0, 0.1)
    assert cp.components["bat"].params["max_hours"] == 2.0
    assert cp.components["bat"].params["p_nom"] == 2.0
    assert cp.lines == [("b0", "b1"), ("b1", "b2")]
    assert len(cp.time_index) == T
    assert "1 Store ignored" in caplog.text


def test_from_pypsa_drops_buses_without_components(caplog):
    pypsa = pytest.importorskip("pypsa")
    from mango_energy_environments.environments.copper_plate import from_pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=T, freq="h"))
    for b in ("b0", "b1", "b2"):
        n.add("Bus", b)
    n.add("Line", "l01", bus0="b0", bus1="b1", x=0.1, s_nom=100)
    n.add("Line", "l12", bus0="b1", bus1="b2", x=0.1, s_nom=100)
    n.add("Generator", "g", bus="b0", p_nom=10)
    n.add("Load", "d", bus="b2", p_set=1.0)

    cp = from_pypsa(n)

    assert cp.lines == []
    assert "1 buses without components dropped: ['b1']" in caplog.text
