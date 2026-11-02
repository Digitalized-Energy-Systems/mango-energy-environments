import numpy as np
import pandas as pd
import pytest

from mango_energy_environments.pypsa_preparation import (
    add_quadratic_cost,
    drop_unusable_units,
    set_storage_defaults,
)

pypsa = pytest.importorskip("pypsa")


@pytest.fixture
def net():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "coal", bus="b", p_nom=100, marginal_cost=20)
    n.add("Generator", "pv", bus="b", p_nom=50, p_max_pu=[0.2, 0.8], marginal_cost=0.01)
    n.add("Generator", "empty", bus="b", p_nom=0, marginal_cost=10)
    n.add("StorageUnit", "phs", bus="b", p_nom=10, carrier="PHS", max_hours=8)
    n.add("StorageUnit", "bat", bus="b", p_nom=4, carrier="battery", max_hours=1)
    n.add("StorageUnit", "dam", bus="b", p_nom=30, carrier="hydro", max_hours=100)
    return n


def prepare(n):
    drop_unusable_units(n, storage_carriers=("hydro",))
    add_quadratic_cost(n, k=0.11)
    set_storage_defaults(n, {"PHS": 6.0, "battery": 2.0}, soc_initial=0.5)


def test_prepare(net):
    prepare(net)
    assert list(net.generators.index) == ["coal", "pv"]
    assert list(net.storage_units.index) == ["phs", "bat"]
    c2 = net.generators.marginal_cost_quadratic
    assert c2["coal"] == pytest.approx(0.11 * 20 / 100)
    assert c2["pv"] == 0.0
    np.testing.assert_allclose(net.storage_units.max_hours, [6.0, 2.0])
    np.testing.assert_allclose(net.storage_units.state_of_charge_initial, [30.0, 4.0])


def test_prepare_is_idempotent(net):
    prepare(net)
    before = (net.generators.copy(), net.storage_units.copy())
    prepare(net)
    pd.testing.assert_frame_equal(net.generators, before[0])
    pd.testing.assert_frame_equal(net.storage_units, before[1])


def test_linear_cost_model(net):
    add_quadratic_cost(net, k=0.0)
    assert (net.generators.marginal_cost_quadratic == 0).all()


def test_scale_to_targets():
    from mango_energy_environments.pypsa_preparation import scale_to_targets

    n = pypsa.Network()
    n.add("Bus", "b")
    n.add("Generator", "w1", bus="b", carrier="onwind", p_nom=10, p_nom_max=40)
    n.add("Generator", "w2", bus="b", carrier="onwind", p_nom=10, p_nom_max=20)
    n.add("Generator", "pv", bus="b", carrier="solar", p_nom=50)
    n.add("StorageUnit", "s1", bus="b", carrier="battery", p_nom=1, p_nom_max=np.inf)
    n.add("StorageUnit", "s2", bus="b", carrier="battery", p_nom=1, p_nom_max=np.inf)
    scale_to_targets(n, {"onwind": 60, "solar": 30, "battery": 10})
    # +40 MW spread by headroom 30:10
    assert n.generators.p_nom["w1"] == pytest.approx(40)
    assert n.generators.p_nom["w2"] == pytest.approx(20)
    assert n.generators.p_nom["pv"] == 50  # already above target
    # no finite headroom: split evenly
    np.testing.assert_allclose(n.storage_units.p_nom, [5, 5])


def test_keep_buses():
    from mango_energy_environments.pypsa_preparation import keep_buses

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    for b in ("in1", "in2", "out"):
        n.add("Bus", b)
    n.add("Line", "inner", bus0="in1", bus1="in2", x=0.1)
    n.add("Line", "tie", bus0="in2", bus1="out", x=0.1)
    n.add("Link", "dc", bus0="out", bus1="in1")
    n.add("Generator", "g_in", bus="in1", p_nom=1, p_max_pu=[0.5, 1.0])
    n.add("Generator", "g_out", bus="out", p_nom=1, p_max_pu=[0.5, 1.0])
    n.add("Load", "l_out", bus="out", p_set=[1.0, 1.0])
    keep_buses(n, ["in1", "in2"])
    assert list(n.buses.index) == ["in1", "in2"]
    assert list(n.lines.index) == ["inner"]
    assert n.links.empty and n.loads.empty
    assert list(n.generators.index) == ["g_in"]
    assert list(n.generators_t.p_max_pu.columns) == ["g_in"]
