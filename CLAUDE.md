# CLAUDE.md — mango-energy-environments (MEE)

Part of the energy scheduling benchmark workspace (`~/development/esb`, see its `CLAUDE.md` for
goal, rules and plan). Branch `restructuring`. Never `git push` unless Jonas asks.
Whenever new code is written here, first invoke the ponytail skill (`ponytail:ponytail`, level full)
— smallest change that fully solves it.

## Scope (IMPORTANT — check every change against it)

- Owns: energy-network representation and simulation in two modes — **physical** (real backend
  frameworks: pandapower, monee, with their physical calculations) and **copper-plate** (no physics;
  lines only define which agents are connected; energy moves freely between any actors; the
  benchmark's main mode). Also: component classification, time series, observe/act, component
  agents bound to network elements, network sources converted into backend networks (e.g.
  PyPSA → pandapower), generic preparation steps (cost model, slack detection, profiles).
- Keep existing functionality (monee restoration env, pandapower scheduling env, express API) working.
- Must not contain: distributed-algorithm logic, economic-dispatch problem formulation,
  central-dispatch solver, benchmark datasets/runs. Never import distributed_resource_optimization.

Repo split (all three): see `README.md` § Scope of this package.
