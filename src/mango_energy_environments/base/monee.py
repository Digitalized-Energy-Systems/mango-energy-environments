"""Monee integration utilities."""
from __future__ import annotations

import networkx as nx

import monee
from monee.model.core import upper as _upper
from monee.model.core import lower as _lower
from monee.network import mes


def energyflow(monee_net):
    """Run steady-state energy flow on *monee_net* and return the result object.

    The returned result object exposes ``.network`` to access the post-flow
    network state.

    ``exclude_unconnected_nodes=True`` keeps the LP feasible after a
    failure that severs a component: without it, monee assembles
    equations for the disconnected component too and the solver returns
    "infeasible" while leaving ``regulation`` at the constructor default
    of 1.0 — which the served-fraction metric then reads as "everything
    served".  This mirrors the same flag used on the oracle path
    (see ``experiment/eval/oracle.py``).
    """

    return monee.run_energy_flow(
        monee_net, solver="gurobi", exclude_unconnected_nodes=True,
    )


def create_physics_stepper(
    monee_net,
    *,
    solve_time_limit_s: float | None = None,
    max_history: int = 4,
):
    """Build the persistent :class:`monee.Stepper` that drives the environment's
    physics solves.

    The stepper works directly on *monee_net* (``copy_base=False``) so agent
    setpoint writes and failure deactivations on the live net are picked up by
    the next step without an override channel; each step still solves on a
    per-step network copy and pushes the solved state into the shared
    ``StepState``, which is what activates monee's inter-step dynamics
    (gas linepack, lumped thermal capacitance, storage SoC).

    Backend selection: a net carrying an ``islanding_config`` solves on
    monee's native gurobipy backend — the islanding extension's indicator
    constraints are bilinear (binary x continuous) and cannot pass Pyomo's LP
    writer, which is exactly the failure mode the single-shot ``energyflow``
    path hits. Plain nets keep the Pyomo+Gurobi path ``energyflow`` uses, so
    campaigns without extensions solve on the identical solver stack as
    before.

    ``on_step_error="skip"`` keeps temporal integration conservative: a failed
    step's interval is carried into the next successful solve instead of being
    silently dropped.
    """
    common: dict = {
        "copy_base": False,
        "on_step_error": "skip",
        "max_history": max_history,
        "simulation": True,
        "exclude_unconnected_nodes": True,
    }
    if getattr(monee_net, "islanding_config", None) is not None:
        from monee.solver.gurobipy import GurobipySolver

        params: dict = {}
        if solve_time_limit_s is not None:
            params["TimeLimit"] = float(solve_time_limit_s)
        return monee.Stepper(monee_net, solver=GurobipySolver(params=params), **common)
    return monee.Stepper(monee_net, solver="gurobi", **common)


def upper(var_or_const):
    """Return the upper bound of a monee ``Var``, or the value itself for constants."""

    return _upper(var_or_const)


def lower(var_or_const):
    """Return the lower bound of a monee ``Var``, or the value itself for constants."""

    return _lower(var_or_const)


def edge_centrality(net) -> dict:
    """Compute edge betweenness centrality on the monee network graph.

    Returns a dict mapping ``(u, v)`` edge tuples to centrality scores.
    """
    return nx.edge_betweenness_centrality(net.graph)


def connected_components(net) -> list[set]:
    """Return a list of sets of node IDs, one per connected component."""
    return list(nx.connected_components(net.graph))


def _create_monee_bench():

    return mes.create_monee_benchmark_net()


def _create_mv_multi_cigre():

    return mes.create_mv_multi_cigre()


def fetch_example_net():
    """Return the small monee benchmark multi-energy network."""
    return _create_monee_bench()


def fetch_cigre_net():
    """Return the MV CIGRE multi-energy benchmark network."""
    return _create_mv_multi_cigre()


def solve_load_shedding_optimization(
    net,
    bound_vm: tuple[float, float] = (0.9, 1.1),
    bound_t: tuple[float, float] = (0.95, 1.05),
    bound_pressure: tuple[float, float] = (0.9, 2.0),
    ext_el_grid_bound: tuple[float, float] = (0.0, 1.0),
    ext_gas_grid_bound: tuple[float, float] = (0.0, 1.0),
):
    """Solve load-shedding optimisation with tight operational bounds."""

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

    Uses the inverse formulation so that higher values mean better resilience.
    """

    return monee.problem.calc_general_resilience_performance(net, inv=True)
