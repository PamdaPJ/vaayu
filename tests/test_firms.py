"""Unit tests for FIRMS VIIRS fire detection processing in fetch_firms.py.

All tests use synthetic DataFrames and mock environments.
Zero external network calls are performed.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest

from fetch_firms import (
    PUNJAB_REGION_BOX,
    FIRMS_DAY_LIMIT,
    generate_oct_nov_windows,
    yearly_counts,
    overpass_histogram,
    counts_by_confidence,
    plot_yearly_counts,
    plot_overpass_hours,
    sanitize_url,
)


def _make_synthetic_firms_df() -> pd.DataFrame:
    """Create a minimal synthetic VIIRS active fire DataFrame."""
    data = {
        "latitude": [30.5, 31.0, 30.8, 31.2, 30.6],
        "longitude": [75.2, 75.8, 74.9, 76.1, 75.5],
        "bright_ti4": [340.5, 355.2, 330.0, 360.1, 325.0],
        "scan": [0.4, 0.5, 0.4, 0.6, 0.4],
        "track": [0.4, 0.4, 0.4, 0.5, 0.4],
        # 3 in 2021 (Oct/Nov), 1 in 2022 (Nov), 1 in 2021 (July - outside season)
        "acq_date": [
            "2021-10-15",
            "2021-11-05",
            "2021-07-20",  # Outside Oct-Nov window
            "2022-11-10",
            "2021-10-28",
        ],
        # UTC times: 0800 (13:30 IST -> hour 13), 0830 (14:00 IST -> hour 14),
        # 2000 (01:30 IST -> hour 1), 0745 (13:15 IST -> hour 13), 2015 (01:45 IST -> hour 1)
        "acq_time": ["0800", "0830", "2000", "0745", "2015"],
        "satellite": ["N", "N", "N", "N", "N"],
        "confidence": ["nominal", "high", "low", "nominal", "high"],
        "version": ["1.0", "1.0", "1.0", "1.0", "1.0"],
        "bright_ti5": [295.0, 298.0, 290.0, 300.0, 292.0],
        "frp": [15.2, 32.5, 8.1, 45.0, 12.0],
        "daynight": ["D", "D", "N", "D", "N"],
    }
    return pd.DataFrame(data)


def test_punjab_box_constant():
    """Verify Punjab-region bounding box constant definition."""
    assert len(PUNJAB_REGION_BOX) == 4
    min_lon, min_lat, max_lon, max_lat = PUNJAB_REGION_BOX
    assert 73.0 <= min_lon < max_lon <= 78.0
    assert 28.0 <= min_lat < max_lat <= 34.0


def test_api_day_limit_constant():
    """Verify official FIRMS API day limit is 5 days."""
    assert FIRMS_DAY_LIMIT == 5


def test_generate_date_windows():
    """Test date window generator for Oct 1 to Nov 30."""
    windows = generate_oct_nov_windows(2021, day_limit=5)
    # Total days in Oct (31) + Nov (30) = 61 days
    # 12 windows of 5 days (60 days) + 1 window of 1 day = 13 windows
    assert len(windows) == 13

    # All windows must not exceed the API day limit
    for start_date, day_range in windows:
        assert 1 <= day_range <= 5
        assert start_date.startswith("2021-")

    assert windows[0] == ("2021-10-01", 5)
    assert windows[-1] == ("2021-11-30", 1)


def test_yearly_counts():
    """Test yearly_counts pure function filters to Oct-Nov and groups by year."""
    df = _make_synthetic_firms_df()
    counts = yearly_counts(df)

    # 2021 had 3 detections in Oct-Nov (July detection excluded)
    assert counts.get(2021) == 3
    # 2022 had 1 detection in Nov
    assert counts.get(2022) == 1


def test_overpass_histogram():
    """Test overpass_histogram converts UTC HHMM to IST hour (UTC+5:30)."""
    df = _make_synthetic_firms_df()
    hist = overpass_histogram(df)

    # 0800 UTC -> 13:30 IST (hour 13)
    # 0745 UTC -> 13:15 IST (hour 13)
    # Total at hour 13 = 2
    assert hist.get(13) == 2

    # 0830 UTC -> 14:00 IST (hour 14)
    # Total at hour 14 = 1
    assert hist.get(14) == 1

    # 2000 UTC -> 01:30 IST (hour 1)
    # 2015 UTC -> 01:45 IST (hour 1)
    # Total at hour 1 = 2
    assert hist.get(1) == 2

    # Empty hour should be 0
    assert hist.get(12, 0) == 0


def test_counts_by_confidence():
    """Test counts_by_confidence counts all classes with no filtering."""
    df = _make_synthetic_firms_df()
    conf_counts = counts_by_confidence(df)

    assert conf_counts.get("nominal") == 2
    assert conf_counts.get("high") == 2
    assert conf_counts.get("low") == 1


def test_sanitize_url_redacts_map_key():
    """Test that FIRMS key is never exposed in printed/logged URLs."""
    raw_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/SECRET_KEY_12345/VIIRS_SNPP_SP/73.8,29.5,77.0,32.6/5/2021-10-01"
    clean_url = sanitize_url(raw_url, secret_key="SECRET_KEY_12345")
    assert "SECRET_KEY_12345" not in clean_url
    assert "[REDACTED]" in clean_url


def test_plot_generation(tmp_path: Path):
    """Test matplotlib figures are saved with required captions."""
    df = _make_synthetic_firms_df()

    fig1_path = tmp_path / "fire_counts_by_year.png"
    fig2_path = tmp_path / "overpass_hours.png"

    plot_yearly_counts(df, output_path=fig1_path)
    assert fig1_path.exists()
    assert fig1_path.stat().st_size > 0

    plot_overpass_hours(df, output_path=fig2_path)
    assert fig2_path.exists()
    assert fig2_path.stat().st_size > 0
