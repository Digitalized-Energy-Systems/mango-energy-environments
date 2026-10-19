"""Tests for the communication topologies and their communication simulation."""

from __future__ import annotations

import random

import networkx as nx
import pypsa
import pytest
from mango.simulation.communication import MessagePackage, SimpleCommunicationSimulation

from mango_energy_environments.environments.scheduling import (
    LOAD,
    STORAGE,
    THERMAL,
    ComponentRef,
    TopologyCommunicationSimulation,
    build_communication_graph,
    bus_graph,
)


def _net() -> pypsa.Network:
    """b0 - b1 - b2 (line), b2 = b3 (link), b4 isolated; b1 has only a load."""
    n = pypsa.Network()
    for b in ("b0", "b1", "b2", "b3", "b4"):
        n.add("Bus", b)
    n.add("Line", "l01", bus0="b0", bus1="b1", x=0.1, s_nom=100)
    n.add("Line", "l12", bus0="b1", bus1="b2", x=0.1, s_nom=100)
    n.add("Link", "k23", bus0="b2", bus1="b3", p_nom=100)
    n.add("Generator", "g0", bus="b0", p_nom=10)
    n.add("Generator", "g1", bus="b0", p_nom=10)
    n.add("Generator", "g2", bus="b2", p_nom=10)
    n.add("StorageUnit", "s3", bus="b3", p_nom=10)
    n.add("Generator", "g4", bus="b4", p_nom=10)
    n.add("Load", "d1", bus="b1", p_set=5)
    n.add("Load", "d3", bus="b3", p_set=5)
    return n


G = {i: ComponentRef(THERMAL, f"g{i}") for i in (0, 1, 2, 4)}
S3 = ComponentRef(STORAGE, "s3")
D1 = ComponentRef(LOAD, "d1")
D3 = ComponentRef(LOAD, "d3")
PARTICIPANTS = [G[2], G[0], S3, G[4], G[1]]
LOADS = [D1, D3]


class TestBusGraph:
    def test_lines_and_links_join_buses(self):
        graph = bus_graph(_net())
        assert set(graph.edges) == {("b0", "b1"), ("b1", "b2"), ("b2", "b3")}
        assert "b4" in graph


class TestCompleteTopology:
    def test_everyone_linked(self):
        graph = build_communication_graph("complete", _net(), PARTICIPANTS, LOADS)
        n = len(PARTICIPANTS) + len(LOADS)
        assert graph.number_of_edges() == n * (n - 1) // 2


class TestRingTopology:
    def test_cycle_ordered_by_id(self):
        graph = build_communication_graph("ring", _net(), PARTICIPANTS)
        order = [G[0], G[1], G[2], G[4], S3]
        expected = {frozenset(e) for e in zip(order, order[1:] + order[:1])}
        assert {frozenset(e) for e in graph.edges} == expected

    def test_loads_attach_via_bus(self):
        graph = build_communication_graph("ring", _net(), PARTICIPANTS, LOADS)
        assert set(graph[D3]) == {S3}
        # b1 has no participant: nearest buses b0 and b2 (1 hop) -> lowest name b0
        assert set(graph[D1]) == {G[0], G[1]}

    def test_two_participants_single_edge(self):
        graph = build_communication_graph("ring", _net(), [G[0], G[1]])
        assert graph.number_of_edges() == 1


class TestGridTopology:
    def test_bus_cliques_and_neighbouring_buses(self):
        graph = build_communication_graph("grid", _net(), PARTICIPANTS, LOADS)
        sub = graph.subgraph(PARTICIPANTS)
        assert sub.has_edge(G[0], G[1])  # same bus
        assert sub.has_edge(G[2], S3)  # link b2 - b3
        # b1 hosts no participant: b0 and b2 become neighbours through it
        assert sub.has_edge(G[0], G[2]) and sub.has_edge(G[1], G[2])
        assert not sub.has_edge(G[0], S3)

    def test_islands_bridged_so_participants_connected(self):
        graph = build_communication_graph("grid", _net(), PARTICIPANTS, LOADS)
        assert nx.is_connected(graph.subgraph(PARTICIPANTS))
        assert set(graph[G[4]]) == {G[0], G[1]}  # b4 bridged to b0 (lowest name)

    def test_custom_bus_mapping(self):
        bus_of = {G[0]: "b0", G[2]: "b2"}
        graph = build_communication_graph("grid", _net(), [G[0], G[2]], bus_of=bus_of)
        assert graph.has_edge(G[0], G[2])

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown topology"):
            build_communication_graph("star", _net(), PARTICIPANTS)


def _pkg(sender: str, receiver: str) -> MessagePackage:
    return MessagePackage(
        sender_id=sender, receiver_id=receiver, sent_time=0.0, content=None
    )


class TestTopologyCommunicationSimulation:
    def test_delay_grows_with_hops(self):
        sim = TopologyCommunicationSimulation(
            delay_s=0.5, graph=nx.path_graph(["a", "b", "c"])
        )
        res = sim.calculate_communication(
            0.0, [_pkg("a", "b"), _pkg("a", "c"), _pkg("x", "c")]
        )
        assert [r.delay_s for r in res.package_results] == [0.5, 1.0, 0.5]
        assert all(r.reached for r in res.package_results)

    def test_matches_simple_simulation_without_graph(self):
        pkgs = [_pkg(f"a{i}", f"a{i + 1}") for i in range(500)]
        ours = TopologyCommunicationSimulation(0.02, 0.1, random.Random(3))
        simple = SimpleCommunicationSimulation(
            default_delay_s=0.02, loss_percent=0.1, rng=random.Random(3)
        )
        assert (
            ours.calculate_communication(0.0, pkgs).package_results
            == simple.calculate_communication(0.0, pkgs).package_results
        )

    def test_loss_compounds_per_hop(self):
        graph = nx.path_graph(["a", "b", "c", "d"])
        sim = TopologyCommunicationSimulation(0.0, 0.2, random.Random(0), graph)
        res = sim.calculate_communication(0.0, [_pkg("a", "d")] * 20000)
        reached = sum(r.reached for r in res.package_results) / 20000
        assert reached == pytest.approx(0.8**3, abs=0.02)
