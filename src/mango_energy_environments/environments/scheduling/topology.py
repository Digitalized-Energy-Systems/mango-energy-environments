"""Communication topologies for scheduling agents on a PyPSA network.

:func:`build_communication_graph` returns an undirected ``networkx`` graph whose
nodes are :class:`ComponentRef` s and whose edges are direct communication links:

* ``complete`` — every component links to every other one.
* ``ring`` — the *participants* (the agents taking part in the distributed
  algorithm) form a cycle ordered by component id.
* ``grid`` — follows the electrical network: participants on one bus are fully
  connected, and every participant on bus A links to every participant on bus B
  when A and B are joined by a line, link or transformer. Buses without
  participants are passed through (their neighbours become neighbours), and
  disconnected islands are bridged so the graph stays connected.

In ``ring`` and ``grid`` the *attached* components (e.g. loads, which only report
to a leader) connect via their bus: to every participant on their own bus, or on
the nearest bus that has participants.

:class:`TopologyCommunicationSimulation` turns such a graph (relabelled to agent
ids) into message delays and losses: a message travels the shortest path, so it
takes ``hops × delay_s`` and is lost with probability ``1 − (1 − loss_rate)^hops``
(store-and-forward over the topology; relaying costs no agent time).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import networkx as nx
from mango.simulation.communication import (
    CommunicationSimulation,
    CommunicationSimulationResult,
    MessagePackage,
    PackageResult,
)

from .base import ComponentRef

logger = logging.getLogger(__name__)

TOPOLOGIES = ("complete", "ring", "grid")

#: PyPSA branch components that join two buses (``bus0`` / ``bus1``).
BRANCH_COMPONENTS = ("lines", "links", "transformers")


def bus_graph(net: Any) -> nx.Graph:
    """Undirected graph of *net*'s buses, joined by lines, links and transformers."""
    graph = nx.Graph()
    graph.add_nodes_from(str(b) for b in net.buses.index)
    for name in BRANCH_COMPONENTS:
        df = getattr(net, name, None)
        if df is None or df.empty:
            continue
        for b0, b1 in zip(df["bus0"], df["bus1"], strict=True):
            if str(b0) != str(b1):
                graph.add_edge(str(b0), str(b1))
    return graph


def _connect_all(graph: nx.Graph, a: Iterable[Hashable], b: Iterable[Hashable]) -> None:
    b = list(b)
    for u in a:
        for v in b:
            if u != v:
                graph.add_edge(u, v)


def _sort_key(ref: ComponentRef) -> tuple[str, str]:
    return (str(ref.component_id), str(ref.element_type))


def _participant_bus_graph(buses: nx.Graph, participant_buses: set[str]) -> nx.Graph:
    """Buses with participants; adjacent if a path joins them through buses without."""
    graph = nx.Graph()
    graph.add_nodes_from(participant_buses)
    for start in participant_buses:
        seen = {start}
        queue = deque([start])
        while queue:
            bus = queue.popleft()
            for nxt in buses.neighbors(bus):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if nxt in participant_buses:
                    graph.add_edge(start, nxt)
                else:
                    queue.append(nxt)
    return graph


def _nearest_participant_bus(
    buses: nx.Graph, bus: str, participant_buses: set[str]
) -> str:
    """The participant bus closest to *bus* (fewest hops, then lowest name)."""
    if bus in participant_buses:
        return bus
    if bus in buses:
        lengths = nx.single_source_shortest_path_length(buses, bus)
        reachable = [(d, b) for b, d in lengths.items() if b in participant_buses]
        if reachable:
            return min(reachable)[1]
    fallback = min(participant_buses)
    logger.warning(
        "Bus %r reaches no bus with participants; attaching it to bus %r", bus, fallback
    )
    return fallback


def build_communication_graph(
    kind: str,
    net: Any,
    participants: Sequence[ComponentRef],
    attached: Sequence[ComponentRef] = (),
    bus_of: Mapping[ComponentRef, str] | None = None,
) -> nx.Graph:
    """Communication graph of *participants* and *attached* components.

    :param kind: one of :data:`TOPOLOGIES`.
    :param net: the ``pypsa.Network`` (buses and branches; only used by
        ``ring``/``grid``).
    :param participants: components running the distributed algorithm.
    :param attached: components that only talk to others (e.g. loads).
    :param bus_of: bus per component; defaults to the ``bus`` column of the
        component's PyPSA table (generators, loads, storage units).
    :return: graph over the given :class:`ComponentRef` s. The subgraph of the
        participants is connected.
    """
    if kind not in TOPOLOGIES:
        raise ValueError(f"unknown topology {kind!r}; choose from {TOPOLOGIES}")
    participants = list(participants)
    attached = list(attached)
    graph = nx.Graph()
    graph.add_nodes_from(participants)
    graph.add_nodes_from(attached)
    if kind == "complete":
        _connect_all(graph, graph.nodes, list(graph.nodes))
        return graph

    if bus_of is None:
        bus_of = {ref: _pypsa_bus(net, ref) for ref in (*participants, *attached)}
    buses = bus_graph(net)
    by_bus: dict[str, list[ComponentRef]] = {}
    for ref in participants:
        by_bus.setdefault(str(bus_of[ref]), []).append(ref)

    if kind == "ring":
        ordered = sorted(participants, key=_sort_key)
        if len(ordered) == 2:
            graph.add_edge(ordered[0], ordered[1])
        elif len(ordered) > 2:
            nx.add_cycle(graph, ordered)
    else:  # grid
        for refs in by_bus.values():
            _connect_all(graph, refs, refs)
        pbuses = _participant_bus_graph(buses, set(by_bus))
        for a, b in pbuses.edges:
            _connect_all(graph, by_bus[a], by_bus[b])
        islands = sorted(
            (sorted(c) for c in nx.connected_components(pbuses)), key=lambda c: c[0]
        )
        if len(islands) > 1:
            logger.warning(
                "grid topology: %d disconnected bus islands; bridging them at their "
                "lowest-named buses",
                len(islands),
            )
        for left, right in zip(islands, islands[1:], strict=False):
            _connect_all(graph, by_bus[left[0]], by_bus[right[0]])

    if by_bus:
        pbus_set = set(by_bus)
        for ref in attached:
            bus = _nearest_participant_bus(buses, str(bus_of[ref]), pbus_set)
            _connect_all(graph, [ref], by_bus[bus])
    return graph


def _pypsa_bus(net: Any, ref: ComponentRef) -> str:
    from .base import LOAD, RENEWABLE, STORAGE, THERMAL

    table = {
        THERMAL: "generators",
        RENEWABLE: "generators",
        LOAD: "loads",
        STORAGE: "storage_units",
    }[ref.element_type]
    return str(getattr(net, table).at[ref.component_id, "bus"])


class TopologyCommunicationSimulation(CommunicationSimulation):
    """Message delay and loss over the shortest path in a communication graph.

    Without a graph (or for a pair it does not contain / connect) a message
    takes one hop, which is exactly mango's ``SimpleCommunicationSimulation``
    with the same *rng*: one draw per message, lost if it is below the loss
    probability.

    :param delay_s: delay per hop (seconds).
    :param loss_rate: probability that one hop loses the message.
    :param rng: random source for the loss draws.
    :param graph: undirected graph over agent ids; see :meth:`set_graph`.
    """

    def __init__(
        self,
        delay_s: float = 0.0,
        loss_rate: float = 0.0,
        rng: random.Random | None = None,
        graph: nx.Graph | None = None,
    ) -> None:
        self.delay_s = float(delay_s)
        self.loss_rate = float(loss_rate)
        self._rng = rng or random.Random()
        self._graph: nx.Graph | None = None
        self._hops: dict[str, dict[str, int]] = {}
        if graph is not None:
            self.set_graph(graph)

    def set_graph(self, graph: nx.Graph | None) -> None:
        """Use *graph* (nodes are agent ids) for all following messages."""
        self._graph = graph
        self._hops = {}

    def hops(self, sender: str | None, receiver: str) -> int:
        """Shortest-path length from *sender* to *receiver* (1 if unknown)."""
        g = self._graph
        if g is None or sender is None or sender not in g or receiver not in g:
            return 1
        lengths = self._hops.get(sender)
        if lengths is None:
            lengths = nx.single_source_shortest_path_length(g, sender)
            self._hops[sender] = lengths
        return max(1, lengths.get(receiver, 1))

    def calculate_communication(
        self, current_time: float, messages: list[MessagePackage]
    ) -> CommunicationSimulationResult:
        results: list[PackageResult] = []
        for msg in messages:
            hops = self.hops(msg.sender_id, msg.receiver_id)
            loss = self.loss_rate if hops == 1 else 1.0 - (1.0 - self.loss_rate) ** hops
            reached = self._rng.random() >= loss
            results.append(PackageResult(reached=reached, delay_s=hops * self.delay_s))
        return CommunicationSimulationResult(package_results=results)
