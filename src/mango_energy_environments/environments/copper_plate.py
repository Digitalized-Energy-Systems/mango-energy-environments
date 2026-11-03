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
        self._component_of: dict[str, str] = {}  # agent id -> component id

    def install(self, agent, **kwargs) -> None:
        # mango's world.register calls install for every agent; only bind those given an id.
        if "id" not in kwargs:
            return
        cid = kwargs["id"]
        if cid not in self.network.components:
            raise KeyError(f"unknown component {cid!r}")
        self._component_of[agent.aid] = cid

    def observe(self, agent_id: str, name: str) -> Any:
        c = self.network.components[self._component_of[agent_id]]
        if name == "statics":
            return {"kind": c.kind, "bus": c.bus, "cost": c.cost, **c.params}
        if name == "p_min":
            return c.p_min
        if name in ("p_max", "demand"):
            return c.p_max
        raise KeyError(f"no observation {name!r}")

    def act(self, agent_id: str, action: str, schedule) -> None:
        if action != "set_schedule":
            raise KeyError(f"no action {action!r}")
        schedule = np.asarray(schedule, dtype=float)
        if schedule.shape != (len(self.network.time_index),):
            raise ValueError(
                f"schedule shape {schedule.shape}, expected one value per step"
            )
        self.schedules[self._component_of[agent_id]] = schedule

    def balance(self) -> np.ndarray:
        """Residual per step: injections − demand (MW). Loads always draw their demand."""
        residual = np.zeros(len(self.network.time_index))
        for cid, c in self.network.components.items():
            if c.kind == LOAD:
                residual -= c.p_max
            elif cid in self.schedules:
                residual += self.schedules[cid]
        return residual
