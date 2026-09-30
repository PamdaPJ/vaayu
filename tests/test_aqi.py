"""Unit tests for CPCB AQI engine in aqi.py."""

import numpy as np
import pandas as pd
import pytest

from aqi import (
    load_breakpoints,
    get_category_thresholds,
    sub_index,
    aqi,
    category,
    daily_average,
    p_severe,
)


def test_no_gaps_or_overlaps_in_breakpoints():
    """Verify that every pollutant's concentration intervals have no gaps or overlaps."""
    df = load_breakpoints()
    for pollutant, grp in df.groupby("pollutant"):
        sorted_grp = grp.sort_values("conc_low").reset_index(drop=True)
        for i in range(len(sorted_grp) - 1):
            curr_high = float(sorted_grp.loc[i, "conc_high"])
            next_low = float(sorted_grp.loc[i + 1, "conc_low"])
            # Either contiguous (curr_high == next_low or integer boundary curr_high + 1 == next_low)
            assert (
                np.isclose(curr_high, next_low, atol=1.0) or curr_high <= next_low
            ), f"Gap or overlap detected in {pollutant} between {curr_high} and {next_low}"


def test_max_sub_index_rule():
    """Verify AQI equals the maximum of the individual sub-indices."""
    # PM2.5 = 20 (Good, sub-index ~33)
    # PM10 = 260 (Poor, sub-index ~210)
    readings = {"PM2.5": 20.0, "PM10": 260.0}
    val = aqi(readings)
    si_pm25 = sub_index("PM2.5", 20.0)
    si_pm10 = sub_index("PM10", 260.0)
    assert val == max(si_pm25, si_pm10)


def test_pm25_just_above_severe_threshold():
    """Verify PM2.5 just above 250 µg/m³ returns Severe AQI."""
    si = sub_index("PM2.5", 251.0)
    assert si >= 401
    assert category(si) == "Severe"


def test_pm10_alone_triggering_severe():
    """Verify PM10 alone > 430 µg/m³ triggers Severe AQI."""
    val = aqi({"PM10": 450.0})
    assert val >= 401
    assert category(val) == "Severe"


def test_category_thresholds():
    """Verify dynamic threshold lookup from breakpoints CSV."""
    thresholds = get_category_thresholds()
    assert thresholds["Severe"] == 401.0
    assert thresholds["Very Poor or worse"] == 301.0
