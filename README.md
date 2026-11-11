![lifecycle](https://img.shields.io/badge/lifecycle-experimental-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/actions/workflows/test.yml/badge.svg)](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/actions/workflows/test.yml)

# mango-energy-environments

Energy system simulation environments for [mango-agents](https://github.com/OFFIS-DAI/mango), the Python multi-agent systems framework.

**Networks** (`mango_energy_environments.network`) are prepared independently of any environment:

```
source network (PyPSA, pandapower/SimBench, ...) → from_<source>() → prepare steps → Network → any environment
```

- `Network` = time index + `Component`s (kind, bus, `p_min`/`p_max` arrays, cost c₁/c₂, `params`)
  + lines (bus pairs); `connection_graph(net)` gives the component graph (→ mango `graph_topology`).
- `from_pypsa(pypsa.Network(path))` converts a PyPSA network (install extra `[pypsa]`):
  renewable = generator with an hourly `p_max_pu` profile; lines and links both connect agents.
- `from_pandapower(pp_net, profiles)` converts a pandapower net + absolute MW profiles per table
  (`"load"`, `"sgen"`): gen → thermal 0…`max_p_mw` (no unit commitment), sgen → renewable
  (fuel-fired biomass sgen → thermal 0…profile, as in `from_pypsa`), costs
  from `poly_cost`; lines and trafos (unless an open switch cuts them) and closed bus–bus switches
  connect buses. NaN profiles raise. `load_simbench(code)` → `(pp_net, hourly profiles)` for a
  SimBench grid (local-time labels replaced by a regular 15-min index before averaging).
- Both converters drop buses without components; lines through them are bridged
  (`bridge_lines`), so dropping a transit bus does not split the grid.
- Source-independent preparation on the `Network` (`network.prepare`): `drop_unusable_units`,
  `add_quadratic_cost(net, k, band, band_kinds)` (c₂ = k·c₁/p_nom on thermal units with c₁ > 0;
  the other units of a kind in `band_kinds` — default renewables, storage and free thermal — get
  c₂ = band/(2·p_nom), the rest c₂ = 0; `k = band = 0` = linear),
  `set_storage_defaults(net, max_hours, soc_initial, marginal_cost)` (per-carrier storage c₁).
- PyPSA-only step for building dataset files (`network.pypsa`): `keep_buses(n, buses)` (cut out a
  region).
- `save_network(net, path, meta)` / `load_network(path)`: a prepared `Network` as NetCDF (`.nc`;
  arrays zlib-compressed, per-component params as JSON, `meta` e.g. the recipe as JSON attribute),
  so a network is prepared once and reused by every run.
- `to_pandapower(net)` → `(pp_net, timeseries, refs)` runs a prepared `Network` in the pandapower
  scheduling environment: `PowerSystemsBehavior(pp_net, timeseries)`, register each agent with
  `id=refs[component_id]`. No lines are converted (a `Network` has no impedances; the
  environment runs no power flow); storage bounds follow pandapower's sign (positive = charging).

Three environments are available:

- **Multi-energy restoration** — based on [`monee`](https://pypi.org/project/monee/)
- **Power systems scheduling** — based on [`pandapower`](https://pandapower.readthedocs.io/)
- **Copper plate** — runs a prepared `Network` without physics: energy moves freely between all
  components (one balance pool per time step); lines only decide which agents are connected.
  Agents observe whole-horizon arrays (`statics`, `p_min`, `p_max`/`demand`) and act with
  `set_schedule`; `CopperPlateBehavior.balance()` gives the residual per step. An agent is bound
  to one component (`register(..., id=cid)`) or to all components at a bus (`bus=b`); a bus
  agent passes the component id (`observe(aid, name, cid)`, `act(aid, "set_schedule", s, cid)`)
  and lists its own with `observe(aid, "components")`.

---

## Concept

Each environment is an `EnvironmentBehavior` passed to `create_world(environment=DefaultEnvironment(behavior=...))`.  Agents register with the physical simulation through *observers* (read current state) and *actions* (mutate state).

---

## Scope of this package

This package is one of three with separate responsibilities:

- **mango-energy-environments** — simulation and representation of energy networks for mango, in
  two modes: **physical** (runs on the real frameworks — pandapower, monee — incl. their physical
  calculations) and **copper-plate** (physics skipped; lines only decide which agents are connected,
  energy moves freely between any actors; used by the benchmark). Networks, components, time series,
  observe/act, binding agents to network elements (one component or a whole bus), network sources (e.g. PyPSA) converted into backend networks,
  generic preparation steps. Existing functionality (monee restoration, pandapower scheduling) is
  kept. No distributed-algorithm or benchmark logic.
- **distributed-resource-optimization** — only the logic of the distributed algorithms, as generic
  as possible (usable for non-energy problems too) and built on mango as far as possible, including
  the generic mango carrier/role. No PyPSA, network or benchmark logic.
- **energy-scheduling-benchmark** — builds the benchmark from the other two: the economic-dispatch
  problem formulation, storage model, central-dispatch optimum, benchmark roles (leader, bus agents),
  concrete datasets, runs and comparisons.

Dependencies point one way only: energy-scheduling-benchmark → the other two; mango-energy-environments
and distributed-resource-optimization never import each other.

## Installation

```bash
pip install mango-energy-environments
```

Or in development mode from the repository root:

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.11, `mango-agents`, `monee`, `pandapower`, `networkx`, `pandas`, `numpy`.
The restoration environment solves with **Gurobi** (`base/monee.py`, solver fixed) and needs a
Gurobi installation + license; the copper-plate and pandapower environments do not.

---

## Multi-Energy Restoration Environment

The restoration environment wraps a `monee` multi-energy network.  It:

- Runs energy flow via `monee.run_energy_flow` after every state change.
- Supports scheduled failures (branches, nodes, custom callables) that fire at user-defined simulation times.
- Installs per-agent *observers* (current physical values) and *actions* (`regulate`, `switch`) when `world.environment.install(agent, id=..., type=...)` is called.
- Provides helpers to build a mango `Topology` that mirrors the physical network graph.

### Example

```python
import asyncio
from mango.agent.role import Role, RoleAgent
from mango.simulation.world import create_world, discrete_step_until
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.express.topology import create_topology

from mango_energy_environments import (
    RestorationEnvironmentBehavior,
    BranchFailureEvent,
    Failure,
    topology_based_on_grid,
    schedule_failure,
    fetch_example_net,
)


class BranchFailureHandler(Role):
    def __init__(self):
        self.counter = 0
        self.msg_counter = 0

    def on_global_event(self, event):
        if isinstance(event, BranchFailureEvent):
            self.counter += 1
            for neighbor in self.context.neighbors():
                self.context.send_message("Failure attention!", neighbor)

    def handle_message(self, content, meta):
        if isinstance(content, str):
            self.msg_counter += 1


async def run():
    monee_net = fetch_example_net()
    behavior = RestorationEnvironmentBehavior(monee_net)
    environment = DefaultEnvironment(behavior=behavior)
    com_sim = SimpleCommunicationSimulation(default_delay_s=0.02)

    world = create_world(
        start_time=0.0, communication_sim=com_sim, environment=environment
    )

    for node in monee_net.nodes:
        agent = RoleAgent()
        agent.add_role(BranchFailureHandler())
        world.register(agent, suggested_aid=node.tid)
        world.environment.install(agent, id=node.id, type="node")

    with create_topology() as topology:
        topology_based_on_grid(monee_net, topology, world)

    branch = monee_net.branches[2]
    failure = Failure(delay_s=2.0, branch_ids=[branch.id])

    async with world:
        schedule_failure(behavior, world, failure)
        await discrete_step_until(world, max_advance_time_s=10.0)


asyncio.run(run())
```

---

## Power Systems Scheduling Environment

The scheduling environment wraps a `pandapower` network and replays time-series data as the simulation progresses.  It provides:

- Automatic scheduling of per-component time-series updates at the correct simulation times.
- Per-agent observers: `"active_power"`, `"max_active_power"`, `"statics"`.
- Per-agent regulate action for thermal generators, renewables, and storage.

### Example

```python
import asyncio
from datetime import datetime

import pandas as pd
import pandapower as pp

from mango.agent.role import Role, RoleAgent
from mango.simulation.world import create_world, discrete_step_until
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.communication import SimpleCommunicationSimulation

from mango_energy_environments import (
    PowerSystemsBehavior,
    PowerUpdateInfo,
    ComponentRef,
)
from mango_energy_environments.environments.scheduling import THERMAL, LOAD


class LoadMonitor(Role):
    def __init__(self):
        self.update_count = 0

    def on_agent_event(self, event):
        if isinstance(event, PowerUpdateInfo):
            self.update_count += 1


async def run():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20)
    b1 = pp.create_bus(net, vn_kv=20)
    g0 = pp.create_gen(net, bus=b0, p_mw=5.0, min_p_mw=0.0, max_p_mw=10.0)
    l0 = pp.create_load(net, bus=b1, p_mw=4.0)

    start = datetime(2024, 1, 1)
    ts_index = pd.date_range(start, periods=24, freq="h")
    timeseries = {
        ComponentRef(THERMAL, g0): pd.Series(
            [8.0 + i * 0.1 for i in range(24)], index=ts_index
        ),
    }

    behavior = PowerSystemsBehavior(
        net=net,
        timeseries=timeseries,
        relevant_types=[THERMAL, LOAD],
        start_datetime=start,
    )
    environment = DefaultEnvironment(behavior=behavior)
    world = create_world(
        start_time=0.0,
        communication_sim=SimpleCommunicationSimulation(default_delay_s=0.01),
        environment=environment,
    )

    for ref in behavior.get_components_by_type([LOAD]):
        agent = RoleAgent()
        agent.add_role(LoadMonitor())
        world.register(agent, suggested_aid=f"load-{ref.index}", id=ref)

    async with world:
        await discrete_step_until(world, max_advance_time_s=3 * 3600.0)


asyncio.run(run())
```

---

## Express API

High-level factory functions for common scenarios:

```python
from mango_energy_environments import (
    create_restoration_world,
    create_small_benchmark_restoration_world,
    create_cigre_benchmark_restoration_world,
    enable_poisson_com_for_monee,
    fetch_example_net,
    fetch_cigre_net,
    solve_load_shedding_optimization,
    calc_general_resilience_performance,
)

# One-liner world creation with Poisson communication delays
world = create_small_benchmark_restoration_world()
```

---

## Development

```bash
git clone <repo>
cd mango-energy-environments
pip install -e ".[dev]"
pytest tests/
```

---

## License

[MIT](LICENSE) — Copyright (c) 2024 Rico Schrage, Digitalized Energy Systems, Carl von Ossietzky Universität Oldenburg.
