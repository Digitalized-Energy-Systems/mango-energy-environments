"""High-level convenience API.

Factory functions that assemble complete simulation worlds from minimal
parameters.

Example::

    from mango_energy_environments import create_restoration_world, fetch_example_net

    net = fetch_example_net()
    world = create_restoration_world(net)
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import datetime

import networkx as nx

from mango.express.topology import Topology, create_topology
from mango.simulation.communication import DelayProviderCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import SimulationWorld, create_world

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
from mango_energy_environments.environments.restoration.multi_energy_monee import (
    RestorationEnvironmentBehavior,
    topology_based_on_grid,
)

__all__ = [
    # Re-exports from base
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
    # Communication simulation
    "enable_poisson_com_for_monee",
]

_DEFAULT_START_DATE = datetime(2024, 8, 1, 0, 0, 0)
_DEFAULT_STATIC_DELAY_S = 0.02


def create_restoration_world(
    monee_net,
    *,
    with_communication: bool = True,
    start_date: datetime = _DEFAULT_START_DATE,
    static_delay_s: float = _DEFAULT_STATIC_DELAY_S,
) -> SimulationWorld:
    """Create a fully configured :class:`~mango.simulation.world.SimulationWorld`
    for a multi-energy restoration scenario.

    Parameters
    ----------
    monee_net:
        A monee :class:`~monee.model.network.Network` instance.
    with_communication:
        When ``True`` (default), install Poisson-distributed communication
        delays based on the physical network topology.
    start_date:
        Simulation start date (stored for logging; the clock starts at 0.0 s).
    static_delay_s:
        Default static message delay used before Poisson communication is
        enabled (or permanently when *with_communication* is ``False``).
    """
    from mango.simulation.communication import SimpleCommunicationSimulation

    behavior = RestorationEnvironmentBehavior(monee_net)
    environment = DefaultEnvironment(behavior=behavior)
    com_sim = SimpleCommunicationSimulation(default_delay_s=static_delay_s)
    world = create_world(start_time=0.0, communication_sim=com_sim, environment=environment)

    if with_communication:
        enable_poisson_com_for_monee(world, monee_net)

    return world


def create_small_benchmark_restoration_world(
    *,
    with_communication: bool = True,
    start_date: datetime = _DEFAULT_START_DATE,
    static_delay_s: float = _DEFAULT_STATIC_DELAY_S,
) -> SimulationWorld:
    """Create a restoration world using the small monee benchmark network."""
    monee_net = fetch_example_net()
    return create_restoration_world(
        monee_net,
        with_communication=with_communication,
        start_date=start_date,
        static_delay_s=static_delay_s,
    )


def create_cigre_benchmark_restoration_world(
    *,
    with_communication: bool = True,
    start_date: datetime = _DEFAULT_START_DATE,
    static_delay_s: float = _DEFAULT_STATIC_DELAY_S,
) -> SimulationWorld:
    """Create a restoration world using the CIGRE MV multi-energy benchmark network."""
    monee_net = fetch_cigre_net()
    return create_restoration_world(
        monee_net,
        with_communication=with_communication,
        start_date=start_date,
        static_delay_s=static_delay_s,
    )


def enable_poisson_com_for_monee(
    world: SimulationWorld,
    monee_net,
    *,
    base_delay_per_message: float = 20.0,
) -> None:
    """Replace the world's communication simulation with Poisson-distributed delays.

    Delays are sampled from ``Poisson(base_delay_per_message * hops)`` where
    *hops* is the shortest-path distance (in edges) between two agents in the
    physical network topology.

    Branch agent IDs (``"branch-hi-lo"``) are mapped to their higher-numbered
    endpoint node's agent ID for routing distance computation.

    Parameters
    ----------
    world:
        The simulation world to update.
    monee_net:
        The monee network (provides the physical topology).
    base_delay_per_message:
        Mean delay per hop in seconds.
    """
    monee_to_topo: dict = {}
    aid_graph: nx.Graph = nx.Graph()

    for node in monee_net.nodes:
        aids = []
        if node.tid in world._agents:
            aids.append(node.tid)
        for child in monee_net.childs_by_ids(node.child_ids):
            if child.tid in world._agents:
                aids.append(child.tid)
        topo_node = len(aid_graph)
        aid_graph.add_node(topo_node, aids=aids, monee_id=node.id)
        monee_to_topo[node.id] = topo_node

    for branch in monee_net.branches:
        from_t = monee_to_topo.get(branch.from_node_id)
        to_t = monee_to_topo.get(branch.to_node_id)
        if from_t is not None and to_t is not None:
            aid_graph.add_edge(from_t, to_t)

    def _label_replacer(aid: str) -> str:
        if aid.startswith("branch-"):
            parts = aid.split("-")
            return f"node-{parts[1]}"
        return aid

    all_aids = list(world._agents.keys())

    aid_to_node: dict[str, int] = {}
    for topo_node, data in aid_graph.nodes(data=True):
        for aid in data.get("aids", []):
            aid_to_node[_label_replacer(aid)] = topo_node

    try:
        all_pairs = dict(nx.all_pairs_shortest_path_length(aid_graph))
    except nx.NetworkXError:
        all_pairs = {}

    def _delay_provider_for(sender: str | None, receiver: str) -> float:
        sender_key = _label_replacer(sender) if sender else None
        receiver_key = _label_replacer(receiver)
        if sender_key is None or sender_key not in aid_to_node:
            return _poisson_sample(base_delay_per_message)
        s_node = aid_to_node[sender_key]
        r_node = aid_to_node.get(receiver_key)
        if r_node is None or s_node == r_node:
            return _poisson_sample(base_delay_per_message)
        hops = all_pairs.get(s_node, {}).get(r_node, 1)
        return _poisson_sample(base_delay_per_message * hops)

    delay_dict: dict[tuple[str | None, str], Callable[[], float]] = {}
    for sender in [None] + all_aids:
        for receiver in all_aids:
            if sender != receiver:
                s = sender
                r = receiver
                delay_dict[(s, r)] = lambda _s=s, _r=r: _delay_provider_for(_s, _r)

    world.communication_sim = DelayProviderCommunicationSimulation(
        default_delay_s_provider=lambda: _poisson_sample(base_delay_per_message),
        delay_s_directed_edge_dict=delay_dict,
    )


def _poisson_sample(lam: float) -> float:
    return max(0.0, random.expovariate(1.0 / lam) if lam > 0 else 0.0)
