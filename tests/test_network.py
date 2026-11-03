import copy
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mango_energy_environments.network import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    add_quadratic_cost,
    drop_unusable_units,
    from_pypsa,
    keep_buses,
    load_network,
    save_network,
    scale_to_targets,
    set_storage_defaults,
)

pypsa = pytest.importorskip("pypsa")

T = 3


def test_from_pypsa(caplog):
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
    np.testing.assert_array_equal(cp.components["coal"].p_max, np.full(T, 10.0))
    np.testing.assert_array_equal(cp.components["d"].p_max, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(cp.components["bat"].p_min, np.full(T, -2.0))
    assert cp.components["coal"].cost == (20.0, 0.1)
    assert cp.components["coal"].params["p_nom"] == 10.0
    assert cp.components["bat"].params["max_hours"] == 2.0
    assert cp.components["bat"].params["p_nom"] == 2.0
    assert cp.lines == [("b0", "b1"), ("b1", "b2")]
    assert len(cp.time_index) == T
    assert "1 Store ignored" in caplog.text


def test_from_pypsa_drops_buses_without_components(caplog):
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
    return from_pypsa(n)


def prepare(net):
    drop_unusable_units(net, storage_carriers=("hydro",))
    add_quadratic_cost(net, k=0.11)
    set_storage_defaults(net, {"PHS": 6.0, "battery": 2.0}, soc_initial=0.5)


def test_prepare(net):
    prepare(net)
    c = net.components
    assert list(c) == ["coal", "pv", "phs", "bat"]
    assert c["coal"].cost == (20.0, pytest.approx(0.11 * 20 / 100))
    assert c["pv"].cost[1] == 0.0
    storage = [c["phs"].params, c["bat"].params]
    assert [p["max_hours"] for p in storage] == [6.0, 2.0]
    assert [p["state_of_charge_initial"] for p in storage] == [30.0, 4.0]


def test_prepare_is_idempotent(net):
    prepare(net)
    before = copy.deepcopy(net.components)
    prepare(net)
    assert {k: (c.cost, c.params) for k, c in net.components.items()} == {
        k: (c.cost, c.params) for k, c in before.items()
    }


def test_linear_cost_model(net):
    add_quadratic_cost(net, k=0.0)
    assert all(c.cost[1] == 0 for c in net.components.values())


def test_scale_to_targets():
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


def test_save_load_roundtrip(tmp_path, net):
    prepare(net)
    net.lines = [("b", "b")]
    path = tmp_path / "net.nc"
    save_network(net, path, meta={"k": 0.11})
    back = load_network(path)

    assert back.time_index.equals(net.time_index)
    assert back.lines == net.lines
    assert list(back.components) == list(net.components)
    for cid, c in net.components.items():
        b = back.components[cid]
        assert (b.kind, b.bus, b.cost, b.params) == (c.kind, c.bus, c.cost, c.params)
        np.testing.assert_array_equal(b.p_min, c.p_min)
        np.testing.assert_array_equal(b.p_max, c.p_max)
    with xr.open_dataset(path) as ds:
        assert json.loads(ds.attrs["meta"]) == {"k": 0.11}


def test_load_rejects_other_nc(tmp_path):
    xr.Dataset({"x": ("i", [1.0])}).to_netcdf(tmp_path / "other.nc")
    with pytest.raises(ValueError, match="not a saved Network"):
        load_network(tmp_path / "other.nc")
