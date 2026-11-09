"""SimBench source: a SimBench grid as pandapower net + hourly absolute profiles, the input of
:func:`~mango_energy_environments.network.pandapower.from_pandapower`."""

from __future__ import annotations

import pandas as pd
import simbench

__all__ = ["load_simbench"]


def load_simbench(code: str, freq: str = "h"):
    """Return ``(pp_net, profiles)`` for SimBench grid *code* (e.g. ``"1-EHV-mixed--0-sw"``).

    *profiles*: absolute ``p_mw`` per element table (``"load"``, ``"sgen"``, ``"gen"``), time ×
    element index, averaged from SimBench's 15-min values to *freq*; the real dates are kept.
    """
    net = simbench.get_simbench_net(code)
    absolute = simbench.get_absolute_values(net, profiles_instead_of_study_cases=True)
    first = pd.to_datetime(
        net.profiles["load"]["time"].iloc[0], format="%d.%m.%Y %H:%M"
    )
    # labels are local time (DST gap and repeat); the rows are consecutive 15-min steps
    time = pd.date_range(first, periods=len(net.profiles["load"]), freq="15min")
    profiles = {
        table: df.set_axis(time).resample(freq).mean()
        for (table, column), df in absolute.items()
        if column == "p_mw" and df.shape[1]
    }
    return net, profiles
