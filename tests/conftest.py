"""Shared pytest fixtures for mango-energy-environments tests."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pandapower as pp
import pytest

from mango_energy_environments.environments.scheduling import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    ComponentRef,
)


# ---------------------------------------------------------------------------
# monee fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def example_net():
    """Small monee benchmark multi-energy network."""
    from mango_energy_environments.base.monee import fetch_example_net

    return fetch_example_net()


# ---------------------------------------------------------------------------
# pandapower fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_power_net():
    """Minimal two-bus pandapower network with one generator and one load."""
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20, name="Bus0")
    b1 = pp.create_bus(net, vn_kv=20, name="Bus1")
    g0 = pp.create_gen(net, bus=b0, p_mw=5.0, min_p_mw=0.0, max_p_mw=10.0, name="Gen0")
    l0 = pp.create_load(net, bus=b1, p_mw=4.0, name="Load0")
    return net, g0, l0


@pytest.fixture()
def power_net_with_timeseries(simple_power_net):
    """Two-bus network with 24-hour hourly timeseries on the generator."""
    net, g0, l0 = simple_power_net
    start = datetime(2024, 1, 1)
    index = pd.date_range(start, periods=24, freq="h")
    timeseries = {
        ComponentRef(THERMAL, g0): pd.Series(
            [5.0 + i * 0.2 for i in range(24)], index=index
        ),
    }
    return net, g0, l0, timeseries, start


@pytest.fixture()
def five_bus_net():
    """Five-bus network mirroring the Julia 'c_sys5_bat' test system spirit:
    two thermal generators, one renewable, two loads, one storage unit."""
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=20, name=f"Bus{i}") for i in range(5)]

    g0 = pp.create_gen(
        net, bus=buses[0], p_mw=50.0, min_p_mw=10.0, max_p_mw=100.0, name="Thermal0"
    )
    g1 = pp.create_gen(
        net, bus=buses[1], p_mw=30.0, min_p_mw=5.0, max_p_mw=60.0, name="Thermal1"
    )
    sg0 = pp.create_sgen(
        net, bus=buses[2], p_mw=20.0, max_p_mw=40.0, name="Wind0"
    )
    l0 = pp.create_load(net, bus=buses[3], p_mw=45.0, name="Load0")
    l1 = pp.create_load(net, bus=buses[4], p_mw=30.0, name="Load1")
    st0 = pp.create_storage(
        net, bus=buses[0], p_mw=0.0, min_p_mw=-20.0, max_p_mw=20.0,
        max_e_mwh=80.0, name="Batt0"
    )

    # Hourly timeseries for renewable (per-unit availability × max_p_mw)
    start = datetime(2024, 1, 1)
    index = pd.date_range(start, periods=72, freq="h")  # 3 days → 72 steps
    ts_wind = pd.Series(
        [0.3 + 0.4 * abs((i % 24 - 12) / 12) for i in range(72)], index=index
    )
    timeseries = {ComponentRef(RENEWABLE, sg0): ts_wind}

    return net, (g0, g1, sg0, l0, l1, st0), timeseries, start
