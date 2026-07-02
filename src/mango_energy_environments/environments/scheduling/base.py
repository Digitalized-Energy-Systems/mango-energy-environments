"""Shared taxonomy for the scheduling-environment behaviors.

Both :class:`~mango_energy_environments.environments.scheduling.power_systems_scheduling.PowerSystemsBehavior`
(pandapower) and :class:`~mango_energy_environments.environments.scheduling.pypsa_scheduling.PyPSABehavior`
(PyPSA) speak the same component taxonomy defined here, so scenario code can
be written against ``THERMAL``/``RENEWABLE``/``LOAD``/``STORAGE`` and
:class:`ComponentRef` without caring which backend is underneath.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

__all__ = [
    "THERMAL",
    "RENEWABLE",
    "LOAD",
    "STORAGE",
    "DEFAULT_RENEWABLE_CARRIERS",
    "ComponentRef",
    "PowerUpdateInfo",
    "SchedulingBehavior",
    "calculate_initial_time",
    "get_possible_components",
    "get_components_by_type",
]

#: Dispatchable / synchronous generation.
THERMAL = "thermal"
#: Non-dispatchable / renewable generation (wind, solar, run-of-river, ...).
RENEWABLE = "renewable"
#: Demand / load.
LOAD = "load"
#: Energy storage / battery.
STORAGE = "storage"

#: Carrier names classified as renewable across both backends.
DEFAULT_RENEWABLE_CARRIERS: frozenset[str] = frozenset(
    {
        "wind",
        "onwind",
        "offwind",
        "offwind-ac",
        "offwind-dc",
        "solar",
        "pv",
        "hydro",
        "ror",
        "biomass",
        "geothermal",
    }
)


@dataclass(frozen=True)
class ComponentRef:
    """Reference to a single component managed by a scheduling behavior.

    Parameters
    ----------
    element_type:
        One of :data:`THERMAL`, :data:`RENEWABLE`, :data:`LOAD`, :data:`STORAGE`.
    component_id:
        The backend-native component identifier — a PyPSA row label (``str``)
        or a pandapower row index (``int``), depending on which behavior
        created the reference.
    """

    element_type: str
    component_id: str | int

    def __iter__(self):
        yield self.element_type
        yield self.component_id


@dataclass(frozen=True)
class PowerUpdateInfo:
    """Agent event emitted when a component's power value changes.

    Carries no payload; the agent re-reads its observer to get the new value.
    """


class SchedulingBehavior(Protocol):
    """Common contract shared by every scheduling-environment backend.

    Both ``PowerSystemsBehavior`` (pandapower) and ``PyPSABehavior`` (PyPSA)
    satisfy this protocol, which is what makes them interchangeable from a
    scenario's point of view.
    """

    def calculate_initial_time(self) -> datetime: ...

    def get_possible_components(self) -> list[ComponentRef]: ...

    def get_components_by_type(self, types: list[str]) -> list[ComponentRef]: ...


def calculate_initial_time(behavior: SchedulingBehavior) -> datetime:
    return behavior.calculate_initial_time()


def get_possible_components(behavior: SchedulingBehavior) -> list[ComponentRef]:
    return behavior.get_possible_components()


def get_components_by_type(
    behavior: SchedulingBehavior, types: list[str]
) -> list[ComponentRef]:
    return behavior.get_components_by_type(types)
