"""Monee integration utilities.

Provides thin wrappers around the monee Python API, mirroring the helper
functions from MangoEnergyEnvironments.jl/src/base/monee.jl.  Because monee is
a native Python package (not accessed via PyCall), no type conversion is
needed — all objects are plain Python values.
"""

from __future__ import annotations

import networkx as nx


# ---------------------------------------------------------------------------
# Energy flow
# ---------------------------------------------------------------------------


def energyflow(monee_net):
    """Run steady-state energy flow on *monee_net* and return the result object.

    Wraps :func:`monee.run_energy_flow`.  The returned result object exposes
    ``.network`` to access the post-flow network state.
    """
    import monee

    return monee.run_energy_flow(monee_net)


# ---------------------------------------------------------------------------
# Variable bounds
# ---------------------------------------------------------------------------


def upper(var_or_const):
    """Return the upper bound of a monee ``Var`` or the value itself for constants.

    Equivalent to ``monee.model.core.upper(value)``.
    """
    from monee.model.core import upper as _upper

    return _upper(var_or_const)


def lower(var_or_const):
    """Return the lower bound of a monee ``Var`` or the value itself for constants."""
    from monee.model.core import lower as _lower

    return _lower(var_or_const)


# ---------------------------------------------------------------------------
# Graph analysis helpers
# ---------------------------------------------------------------------------


def edge_centrality(net) -> dict:
    """Compute edge betweenness centrality on the monee network graph.

    Returns a dict mapping ``(u, v)`` edge tuples to centrality scores.
    """
    return nx.edge_betweenness_centrality(net.graph)


def connected_components(net) -> list[set]:
    """Return a list of sets of node IDs, one per connected component."""
    return list(nx.connected_components(net.graph))


# ---------------------------------------------------------------------------
# Network factories
# ---------------------------------------------------------------------------


def _create_monee_bench():
    """Create the small monee benchmark multi-energy network."""
    from monee.network import mes

    return mes.create_monee_benchmark_net()


def _create_mv_multi_cigre():
    """Create the MV CIGRE multi-energy benchmark network."""
    from monee.network import mes

    return mes.create_mv_multi_cigre()


def fetch_example_net():
    """Return the small monee benchmark multi-energy network.

    Equivalent to ``MangoEnergyEnvironments.fetch_example_net()``.
    """
    return _create_monee_bench()


def fetch_cigre_net():
    """Return the MV CIGRE multi-energy benchmark network.

    Equivalent to ``MangoEnergyEnvironments.fetch_cigre_net()``.
    """
    return _create_mv_multi_cigre()


# ---------------------------------------------------------------------------
# Load-shedding optimisation
# ---------------------------------------------------------------------------


def solve_load_shedding_optimization(
    net,
    bound_vm: tuple[float, float] = (0.9, 1.1),
    bound_t: tuple[float, float] = (0.95, 1.05),
    bound_pressure: tuple[float, float] = (0.9, 2.0),
    ext_el_grid_bound: tuple[float, float] = (0.0, 1.0),
    ext_gas_grid_bound: tuple[float, float] = (0.0, 1.0),
):
    """Solve load-shedding optimisation with tight operational bounds.

    Parameters mirror the Julia ``solve_load_shedding_optimization`` function.
    """
    import monee

    return monee.solve_load_shedding_problem(
        net,
        bound_vm,
        bound_t,
        bound_pressure,
        ext_el_grid_bound,
        ext_gas_grid_bound,
    )


def solve_load_shedding_optimization_relaxed(net):
    """Solve load-shedding optimisation with very relaxed bounds.

    Useful as a feasibility check or warm-start for tighter formulations.
    """
    import monee

    return monee.solve_load_shedding_problem(
        net,
        (0.5, 1.5),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.0, 10.0),
        (0.0, 10.0),
        use_ext_grid_bounds=False,
    )


def calc_general_resilience_performance(net) -> float:
    """Return the general resilience performance metric for *net*.

    Uses the inverse formulation (``inv=True``) so that higher values mean
    better resilience, matching the Julia convention.
    """
    import monee

    return monee.problem.calc_general_resilience_performance(net, inv=True)
