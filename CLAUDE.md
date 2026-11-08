# CLAUDE.md — mango-energy-environments (MEE)

Part of the energy scheduling benchmark workspace (`~/development/esb`, see its `CLAUDE.md` for
goal, rules and plan). Branch `restructuring`. Never `git push` unless Jonas asks.
Whenever new code is written here, first invoke the ponytail skill (`ponytail:ponytail`, level full)
— smallest change that fully solves it.

## Scope (IMPORTANT — check every change against it)

- Owns: energy-network representation and simulation in two modes — **physical** (real backend
  frameworks: pandapower, monee, with their physical calculations) and **copper-plate** (no physics;
  lines only define which agents are connected; energy moves freely between any actors; the
  benchmark's main mode). Also: component classification, time series, observe/act, binding
  agents to network elements (one component or a whole bus), network sources converted into backend networks (e.g.
  PyPSA → pandapower), generic preparation steps (cost model, slack detection, profiles).
- **Preparation is separate from environments** (2026-11-03): `network/` converts any source
  (`from_pypsa`, later `from_pandapower`) into the common `Network` and prepares it with
  source-independent steps (`network/prepare.py`); environments only run a prepared `Network` and
  never contain conversion or preparation code. Backends that need their own data model get a
  converter from `Network` (`network/pandapower.py`: `to_pandapower` for `PowerSystemsBehavior`).
- Keep existing functionality (monee restoration env, pandapower scheduling env, express API) working.
- Upstream code (Rico Schrage's) may be changed freely — Rico agreed (2026-11-03).
- Must not contain: distributed-algorithm logic, economic-dispatch problem formulation,
  central-dispatch solver, benchmark datasets/runs. Never import distributed_resource_optimization.

Repo split (all three): see `README.md` § Scope of this package.
