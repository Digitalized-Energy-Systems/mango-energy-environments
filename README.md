![lifecycle](https://img.shields.io/badge/lifecycle-experimental-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/actions/workflows/test.yml/badge.svg)](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/actions/workflows/test.yml)

# mango-energy-environments

Energy system simulation environments for [mango-agents](https://github.com/OFFIS-DAI/mango), the Python multi-agent systems framework.

This is the Python port of [MangoEnergyEnvironments.jl](MangoEnergyEnvironments.jl/), replacing:

| Julia | Python |
|---|---|
| `Mango.jl` | [`mango-agents`](https://github.com/OFFIS-DAI/mango) |
| `PowerSystems.jl` + `PowerSimulations.jl` | [`pandapower`](https://pandapower.readthedocs.io/) + `scipy` (HiGHS) |
| `monee` via `PyCall` | [`monee`](https://pypi.org/project/monee/) (native) |

Currently two environments are available:

- **Multi-energy restoration** — based on `monee`
- **Power systems scheduling / economic dispatch** — based on `pandapower`

---

## Concept

The general idea of this package is providing concrete environment implementations from the energy domain that plug directly into the mango multi-agent simulation world.  Each environment is an `EnvironmentBehavior` passed to `create_world(environment=DefaultEnvironment(behavior=...))`.  Agents can then be registered and wired up to the physical simulation through *observers* (read current state) and *actions* (mutate state).

---

## Installation

```bash
pip install mango-energy-environments
```

Or in development mode from the repository root:

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.11, `mango-agents`, `monee`, `pandapower`, `scipy`, `networkx`, `pandas`, `numpy`, `highspy`.

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
)
from mango_energy_environments.base.monee import fetch_example_net


class BranchFailureHandler(Role):
    def __init__(self):
        self.counter = 0
        self.msg_counter = 0

    def on_global_event(self, event):
        if isinstance(event, BranchFailureEvent):
            self.counter += 1
            # notify topology neighbors
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

    # One agent per network node
    for node in monee_net.nodes:
        agent = RoleAgent()
        agent.add_role(BranchFailureHandler())
        world.register(agent, suggested_aid=node.tid)
        world.environment.install(agent, id=node.id, type="node")

    # Mirror physical topology to agent topology
    with create_topology() as topology:
        topology_based_on_grid(monee_net, topology, world)

    # Schedule a branch failure at t = 2 s
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
- `solve_central()` — copper-plate economic dispatch (lossless, no network constraints) via HiGHS.

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
    # Build a simple pandapower network
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20)
    b1 = pp.create_bus(net, vn_kv=20)
    g0 = pp.create_gen(net, bus=b0, p_mw=5.0, min_p_mw=0.0, max_p_mw=10.0)
    l0 = pp.create_load(net, bus=b1, p_mw=4.0)

    # Hourly time-series for the generator max output
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

    # Register a monitoring agent for each load
    for ref in behavior.get_components_by_type([LOAD]):
        agent = RoleAgent()
        agent.add_role(LoadMonitor())
        world.register(agent, suggested_aid=f"load-{ref.index}")
        world.environment.install(agent, id=ref)

    async with world:
        # Run 3 hours of simulation (3 × 3600 s)
        await discrete_step_until(world, max_advance_time_s=3 * 3600.0)

    # Solve economic dispatch for current state
    result = behavior.solve_central()
    print(f"Dispatch: success={result['success']}, p_gen={net.gen.at[g0, 'p_mw']:.2f} MW")


asyncio.run(run())
```

---

## Express API

High-level factory functions for common scenarios:

```python
from mango_energy_environments.express import (
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
