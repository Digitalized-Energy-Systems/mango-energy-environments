![lifecycle](https://img.shields.io/badge/lifecycle-experimental-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/actions/workflows/test.yml/badge.svg)](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/actions/workflows/test.yml)

# mango-energy-environments

Energy system simulation environments for [mango-agents](https://github.com/OFFIS-DAI/mango), the Python multi-agent systems framework.

Two environments are available:

- **Multi-energy restoration** — based on [`monee`](https://pypi.org/project/monee/)
- **Power systems scheduling / economic dispatch** — two interchangeable backends behind one `SchedulingBehavior` contract:
  - `PowerSystemsBehavior`, based on [`pandapower`](https://pandapower.readthedocs.io/) + `scipy` (HiGHS), with built-in copper-plate dispatch
  - `PyPSABehavior`, based on [`PyPSA`](https://pypsa.org) (used by `energy-scheduling-benchmark`)

---

## Concept

Each environment is an `EnvironmentBehavior` passed to `create_world(environment=DefaultEnvironment(behavior=...))`.  Agents register with the physical simulation through *observers* (read current state) and *actions* (mutate state).

---

## Installation

```bash
pip install mango-energy-environments
```

Or in development mode from the repository root:

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.11, `mango-agents`, `monee`, `pandapower`, `pypsa`, `scipy`, `networkx`, `pandas`, `numpy`, `highspy`.

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
        super().__init__()
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

    world = create_world(start_time=0.0, communication_sim=com_sim, environment=environment)

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

Energy flow uses monee's default solver (a local GEKKO solver, no external
solver or licence needed). To use another one, call
`energyflow(net, solver=...)` with a monee solver instance such as
`monee.solver.PyomoSolver()`.

---

## Power Systems Scheduling Environment

The scheduling environment wraps a power network and replays time-series data as the simulation progresses.  Two backends implement the same `SchedulingBehavior` contract (component classification into `THERMAL`/`RENEWABLE`/`LOAD`/`STORAGE`, all addressed by `ComponentRef(element_type, component_id)`):

| Backend | Network model | Classification | Extras |
|---------|---------------|----------------|--------|
| `PowerSystemsBehavior` | `pandapower` | by pandapower table | `solve_central()` — copper-plate economic dispatch (lossless, no network constraints) via HiGHS |
| `PyPSABehavior` | `pypsa.Network` | generators by PyPSA `carrier` (`DEFAULT_RENEWABLE_CARRIERS`, overridable via `renewable_carriers=`) | `from_network(net)` auto-extracts timeseries from `generators_t`/`loads_t`/`storage_units_t` (`extract_timeseries`); `from_scenario(...)` |

Both provide:

- Automatic scheduling of per-component time-series updates at the correct simulation times.
- Per-agent observers: `"active_power"`, `"max_active_power"`, `"statics"` (`PyPSABehavior` additionally exposes `"cost"`).
- Per-agent regulate action for thermal generators, renewables, and storage.

### Example (pandapower)

```python
import asyncio
from datetime import datetime

import pandas as pd
import pandapower as pp

from mango.agent.role import Role, RoleAgent
from mango.simulation.world import create_world, discrete_step_until
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.communication import SimpleCommunicationSimulation

from mango_energy_environments import PowerSystemsBehavior, PowerUpdateInfo, ComponentRef
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
        ComponentRef(THERMAL, g0): pd.Series([8.0 + i * 0.1 for i in range(24)], index=ts_index),
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
        world.register(agent, suggested_aid=f"load-{ref.component_id}")
        world.environment.install(agent, id=ref)

    async with world:
        await discrete_step_until(world, max_advance_time_s=3 * 3600.0)

    result = behavior.solve_central()
    print(f"Dispatch: success={result['success']}, p_gen={net.gen.at[g0, 'p_mw']:.2f} MW")


asyncio.run(run())
```

### Example (PyPSA)

```python
import pandas as pd
import pypsa

from mango_energy_environments import LOAD, RENEWABLE, THERMAL, PyPSABehavior

net = pypsa.Network()
net.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
net.add("Bus", "b0")
net.add("Generator", "coal", bus="b0", p_nom=10.0, marginal_cost=30.0, carrier="coal")
net.add("Generator", "pv", bus="b0", p_nom=5.0, carrier="solar",
        p_max_pu=[max(0.0, 1 - abs(h - 12) / 6) for h in range(24)])
net.add("Load", "l0", bus="b0", p_set=[4.0 + 0.1 * h for h in range(24)])

# Timeseries are extracted from generators_t.p_max_pu / loads_t.p_set automatically;
# "solar" is in DEFAULT_RENEWABLE_CARRIERS, so "pv" is classified as RENEWABLE.
behavior = PyPSABehavior.from_network(net)
for ref in behavior.get_components_by_type([THERMAL, RENEWABLE, LOAD]):
    print(ref.element_type, ref.component_id)
```

The resulting behavior is used with `create_world(...)` / `world.environment.install(agent, id=ref)` exactly like the pandapower one above.

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
