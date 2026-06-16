"""mango-energy-environments
============================

Energy system simulation environments for **mango-agents**.

Two environments are provided:

**Restoration** – :mod:`~mango_energy_environments.environments.restoration`
    Integrates a *monee* multi-energy network into a mango
    :class:`~mango.simulation.world.SimulationWorld`.  Manages energy-flow
    computation, scheduled failures, and per-agent observer/action wiring.

**Scheduling** – :mod:`~mango_energy_environments.environments.scheduling`
    Integrates a *pandapower* power network into mango.  Replays timeseries
    data on the simulation clock and offers copper-plate economic dispatch.

Quick start::

    from mango_energy_environments import (
        create_restoration_world,
        fetch_example_net,
    )

    net = fetch_example_net()
    world = create_restoration_world(net)
"""

from mango_energy_environments.environments.restoration import (
    BranchFailureEvent,
    CustomFailureEvent,
    Failure,
    NodeFailureEvent,
    RestorationEnvironmentBehavior,
    apply_failures,
    create_branch_aid,
    on_branch_failure,
    on_custom_failure,
    on_node_failure,
    schedule_failure,
    topology_based_on_grid,
    topology_based_on_grid_groups,
    topology_based_on_sector_grid,
)
from mango_energy_environments.environments.scheduling import (
    ComponentRef,
    PowerSystemsBehavior,
    PowerUpdateInfo,
    calculate_initial_time,
    get_components_by_type,
    get_possible_components,
)
from mango_energy_environments.base.monee import (
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
from mango_energy_environments.express import (
    create_cigre_benchmark_restoration_world,
    create_restoration_world,
    create_small_benchmark_restoration_world,
    enable_poisson_com_for_monee,
)

__all__ = [
    # Restoration
    "Failure",
    "BranchFailureEvent",
    "NodeFailureEvent",
    "CustomFailureEvent",
    "RestorationEnvironmentBehavior",
    "schedule_failure",
    "apply_failures",
    "on_branch_failure",
    "on_node_failure",
    "on_custom_failure",
    "create_branch_aid",
    "topology_based_on_grid",
    "topology_based_on_grid_groups",
    "topology_based_on_sector_grid",
    # Scheduling
    "PowerUpdateInfo",
    "ComponentRef",
    "PowerSystemsBehavior",
    "calculate_initial_time",
    "get_possible_components",
    "get_components_by_type",
    # Monee utilities
    "energyflow",
    "upper",
    "lower",
    "edge_centrality",
    "connected_components",
    "fetch_example_net",
    "fetch_cigre_net",
    "solve_load_shedding_optimization",
    "solve_load_shedding_optimization_relaxed",
    "calc_general_resilience_performance",
    # World factories
    "create_restoration_world",
    "create_small_benchmark_restoration_world",
    "create_cigre_benchmark_restoration_world",
    "enable_poisson_com_for_monee",
]
