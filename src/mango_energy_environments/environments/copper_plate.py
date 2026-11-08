"""Copper-plate backend: energy moves freely between all components, no physics.

Runs a prepared :class:`~mango_energy_environments.network.Network`. Lines only decide which
components (agents) are connected; the balance is one pool per time step. Agents see their
whole horizon at once and set a whole-horizon schedule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from mango.simulation.environment import Behavior

from ..network.model import LOAD, Network

__all__ = ["CopperPlateBehavior"]


class CopperPlateBehavior(Behavior):
    """mango behaviour: agents observe horizon arrays and set schedules; pool balance."""

    def __init__(self, network: Network) -> None:
        self.network = network
        self.schedules: dict[str, np.ndarray] = {}  # component id -> MW per step
        self._components_of: dict[str, list[str]] = {}  # agent id -> its component ids

    def install(self, agent, **kwargs) -> None:
        """Bind *agent* to one component (``id=``) or all components at a bus (``bus=``).

        mango's world.register calls install for every agent; others stay unbound.
        """
        if "id" in kwargs:
            cid = kwargs["id"]
            if cid not in self.network.components:
                raise KeyError(f"unknown component {cid!r}")
            self._components_of[agent.aid] = [cid]
        elif "bus" in kwargs:
            cids = sorted(
                cid
                for cid, c in self.network.components.items()
                if c.bus == kwargs["bus"]
            )
            if not cids:
                raise KeyError(f"no components at bus {kwargs['bus']!r}")
            self._components_of[agent.aid] = cids

    def _component(self, agent_id: str, cid: str | None) -> str:
        """*cid* if it belongs to the agent; None only for a one-component agent."""
        cids = self._components_of[agent_id]
        if cid is None:
            if len(cids) != 1:
                raise ValueError(f"agent {agent_id!r} owns {cids}, pass a component id")
            return cids[0]
        if cid not in cids:
            raise KeyError(f"component {cid!r} does not belong to agent {agent_id!r}")
        return cid

    def observe(self, agent_id: str, name: str, cid: str | None = None) -> Any:
        if name == "components":
            return list(self._components_of[agent_id])
        c = self.network.components[self._component(agent_id, cid)]
        if name == "statics":
            return {"kind": c.kind, "bus": c.bus, "cost": c.cost, **c.params}
        if name == "p_min":
            return c.p_min
        if name in ("p_max", "demand"):
            return c.p_max
        raise KeyError(f"no observation {name!r}")

    def act(self, agent_id: str, action: str, schedule, cid: str | None = None) -> None:
        if action != "set_schedule":
            raise KeyError(f"no action {action!r}")
        schedule = np.asarray(schedule, dtype=float)
        if schedule.shape != (len(self.network.time_index),):
            raise ValueError(
                f"schedule shape {schedule.shape}, expected one value per step"
            )
        self.schedules[self._component(agent_id, cid)] = schedule

    def balance(self) -> np.ndarray:
        """Residual per step: injections − demand (MW). Loads always draw their demand."""
        residual = np.zeros(len(self.network.time_index))
        for cid, c in self.network.components.items():
            if c.kind == LOAD:
                residual -= c.p_max
            elif cid in self.schedules:
                residual += self.schedules[cid]
        return residual
