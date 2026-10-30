# CLAUDE.md — mango-energy-environments (MEE)

Part of the energy scheduling benchmark workspace (`~/development/esb`, see its `CLAUDE.md` for
goal, rules and plan). Branch `restructuring`. Never `git push` unless Jonas asks.

## Scope (IMPORTANT — check every change against it)

- Owns: energy-network representation and simulation (PyPSA now, pandapower later), component
  classification, time series, observe/act, component agents bound to network elements, generic
  network loading + preparation steps (format adapters, cost model, slack detection, profiles).
- Must not contain: distributed-algorithm logic, economic-dispatch problem formulation,
  central-dispatch solver, benchmark datasets/runs. Never import distributed_resource_optimization.

Repo split (all three): see `README.md` § Scope of this package.
