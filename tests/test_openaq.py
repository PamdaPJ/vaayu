"""Tests for OpenAQ data fetcher and preprocessor.

Uses synthetic fixtures and mocks only (no network calls):
- Winter seasons window generation (2020-21, 2021-22, 2025-26) with leap-year handling
- Window skipping outside sensor datetimeFirst/datetimeLast with exact logging
- Prohibition on filling or interpolating missing data
- Timestamp convention parsing and consistent IST timezone conversion
- CPCB column layout compliance with 'source' set to 'openaq'
- Distinct station slug generation per location id
- Guardrail failing loudly on identical PM2.5 series
- Pagination traversal and cache skipping
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from fetch_openaq import (
    CPCB_COLUMNS,
    aggregate_subhourly_to_hourly,
    build_cpcb_layout_dataframe,
    convert_sensor_hours_to_series,
    discover_all_faridabad_locations,
    fetch_and_cache_sensor_window,
    fetch_measurements_for_locations,
    fetch_paginated_results,
    generate_season_windows,
    get_february_end_day,
    get_location_details_and_pm_sensors,
    is_leap_year,
    make_station_slug,
    parse_openaq_datetime_to_ist,
    should_skip_window_for_sensor,
)


def test_winter_seasons_generation_and_leap_year():
    """Verify winter seasons: Oct 1 to Feb 28/29 for 2020-21, 2021-22, 2025-26."""
    # Leap year checks
    assert is_leap_year(2024) is True
    assert get_february_end_day(2024) == 29
    assert is_leap_year(2021) is False
    assert get_february_end_day(2021) == 28
    assert is_leap_year(2022) is False
    assert get_february_end_day(2022) == 28
    assert is_leap_year(2026) is False
    assert get_february_end_day(2026) == 28

    windows = generate_season_windows()
    # 3 seasons * 5 months = 15 windows
    assert len(windows) == 15

    # Season 2020-2021 boundaries
    s2020_windows = [w for w in windows if w[2] == "2020-2021"]
    assert len(s2020_windows) == 5
    assert s2020_windows[0][0] == "2020-10-01T00:00:00Z"
    assert s2020_windows[-1][1] == "2021-03-01T00:00:00Z"  # Feb 28 23:59:59 UTC

    # Season 2021-2022 boundaries
    s2021_windows = [w for w in windows if w[2] == "2021-2022"]
    assert len(s2021_windows) == 5
    assert s2021_windows[0][0] == "2021-10-01T00:00:00Z"
    assert s2021_windows[-1][1] == "2022-03-01T00:00:00Z"  # Feb 28 23:59:59 UTC

    # Season 2025-2026 boundaries
    s2025_windows = [w for w in windows if w[2] == "2025-2026"]
    assert len(s2025_windows) == 5
    assert s2025_windows[0][0] == "2025-10-01T00:00:00Z"
    assert s2025_windows[-1][1] == "2026-03-01T00:00:00Z"  # Feb 28 23:59:59 UTC


def test_skip_window_outside_sensor_bounds():
    """Verify that windows outside a sensor's active period are skipped with an exact reason."""
    d_first = "2021-11-15T00:00:00Z"
    d_last = "2022-01-15T00:00:00Z"

    # 1. Window ending before sensor started -> MUST SKIP
    skip_early, reason_early = should_skip_window_for_sensor(
        dt_from_iso="2020-10-01T00:00:00Z",
        dt_to_iso="2020-11-01T00:00:00Z",
        datetime_first=d_first,
        datetime_last=d_last,
        sensor_id=101,
        parameter="pm25",
        location_name="Station X",
        location_id=999,
    )
    assert skip_early is True
    assert "ends before sensor's datetimeFirst" in reason_early
    assert str(101) in reason_early

    # 2. Window starting after sensor ended -> MUST SKIP
    skip_late, reason_late = should_skip_window_for_sensor(
        dt_from_iso="2025-10-01T00:00:00Z",
        dt_to_iso="2025-11-01T00:00:00Z",
        datetime_first=d_first,
        datetime_last=d_last,
        sensor_id=101,
        parameter="pm25",
        location_name="Station X",
        location_id=999,
    )
    assert skip_late is True
    assert "starts after sensor's datetimeLast" in reason_late

    # 3. Window overlapping active range -> MUST NOT SKIP
    skip_active, reason_active = should_skip_window_for_sensor(
        dt_from_iso="2021-11-01T00:00:00Z",
        dt_to_iso="2021-12-01T00:00:00Z",
        datetime_first=d_first,
        datetime_last=d_last,
        sensor_id=101,
        parameter="pm25",
        location_name="Station X",
        location_id=999,
    )
    assert skip_active is False
    assert reason_active is None


def test_no_filling_or_interpolation():
    """Verify that missing hours are never filled or interpolated."""
    raw_records = [
        {"period": {"datetimeFrom": {"local": "2021-10-01T05:00:00+05:30"}}, "value": 45.0},
        {"period": {"datetimeFrom": {"local": "2021-10-01T08:00:00+05:30"}}, "value": 75.0},  # 2 hours gap
    ]

    df = convert_sensor_hours_to_series(raw_records, parameter_name="pm25")

    # Only recorded entries exist, missing hours are NOT fabricated or interpolated
    assert len(df) == 2
    assert df["pm25"].iloc[0] == 45.0
    assert df["pm25"].iloc[1] == 75.0


def test_duplicates_removed():
    """Verify that exact duplicate (sensor/timestamp, value) rows are dropped."""
    raw_records = [
        {"period": {"datetimeFrom": {"local": "2021-10-01T05:00:00+05:30"}}, "value": 50.0},
        {"period": {"datetimeFrom": {"local": "2021-10-01T05:00:00+05:30"}}, "value": 50.0},  # Duplicate
        {"period": {"datetimeFrom": {"local": "2021-10-01T06:00:00+05:30"}}, "value": 60.0},
        {"period": {"datetimeFrom": {"local": "2021-10-01T06:00:00+05:30"}}, "value": 60.0},  # Duplicate
    ]

    df = convert_sensor_hours_to_series(raw_records, parameter_name="pm25")
    assert len(df) == 2
    assert df["pm25"].tolist() == [50.0, 60.0]


def test_half_hour_offset_windows_dropped():
    """Verify that records with period.datetimeFrom at minute 30 in IST are dropped,

    keeping only records aligned to minute 00 in IST.
    """
    raw_records = [
        # 05:00 IST -> minute 00, MUST KEEP
        {
            "period": {"datetimeFrom": {"local": "2021-10-01T05:00:00+05:30"}},
            "coverage": {"percentComplete": 100.0},
            "value": 50.0,
        },
        # 05:30 IST -> minute 30, MUST DROP
        {
            "period": {"datetimeFrom": {"local": "2021-10-01T05:30:00+05:30"}},
            "coverage": {"percentComplete": 100.0},
            "value": 80.0,
        },
        # 06:00 IST -> minute 00, MUST KEEP
        {
            "period": {"datetimeFrom": {"local": "2021-10-01T06:00:00+05:30"}},
            "coverage": {"percentComplete": 100.0},
            "value": 60.0,
        },
        # 06:30 IST -> minute 30, MUST DROP
        {
            "period": {"datetimeFrom": {"local": "2021-10-01T06:30:00+05:30"}},
            "coverage": {"percentComplete": 100.0},
            "value": 90.0,
        },
    ]

    df = convert_sensor_hours_to_series(raw_records, parameter_name="pm25")
    assert len(df) == 2
    assert df["Timestamp"].dt.minute.tolist() == [0, 0]
    assert df["pm25"].tolist() == [50.0, 60.0]


def test_low_coverage_windows_dropped():
    """Verify that records with coverage.percentComplete below min_coverage_pct (default 75%) are dropped."""
    raw_records = [
        # 100% complete -> Kept
        {
            "period": {"datetimeFrom": {"local": "2021-10-01T05:00:00+05:30"}},
            "coverage": {"percentComplete": 100.0},
            "value": 50.0,
        },
        # 75% complete -> Kept (meets threshold)
        {
            "period": {"datetimeFrom": {"local": "2021-10-01T06:00:00+05:30"}},
            "coverage": {"percentComplete": 75.0},
            "value": 60.0,
        },
        # 74.9% complete -> Dropped (< 75%)
        {
            "period": {"datetimeFrom": {"local": "2021-10-01T07:00:00+05:30"}},
            "coverage": {"percentComplete": 74.9},
            "value": 70.0,
        },
        # 50% complete -> Dropped (< 75%)
        {
            "period": {"datetimeFrom": {"local": "2021-10-01T08:00:00+05:30"}},
            "coverage": {"percentComplete": 50.0},
            "value": 80.0,
        },
    ]

    # Default threshold (75%)
    df_default = convert_sensor_hours_to_series(raw_records, parameter_name="pm25", min_coverage_pct=75.0)
    assert len(df_default) == 2
    assert df_default["pm25"].tolist() == [50.0, 60.0]

    # Custom threshold (50%)
    df_custom = convert_sensor_hours_to_series(raw_records, parameter_name="pm25", min_coverage_pct=50.0)
    assert len(df_custom) == 4
    assert df_custom["pm25"].tolist() == [50.0, 60.0, 70.0, 80.0]


def test_no_averaging_across_overlapping_windows():
    """Verify that overlapping 1-hour windows (e.g. 05:00 and 05:30) are NOT averaged together;

    the half-hour window is dropped and the 05:00 window keeps its exact original value.
    """
    raw_records = [
        # 1-hour window 05:00-06:00 IST with value 100.0
        {
            "period": {
                "datetimeFrom": {"local": "2021-10-01T05:00:00+05:30"},
                "datetimeTo": {"local": "2021-10-01T06:00:00+05:30"},
            },
            "coverage": {"percentComplete": 100.0},
            "value": 100.0,
        },
        # Overlapping 1-hour window 05:30-06:30 IST with value 200.0
        {
            "period": {
                "datetimeFrom": {"local": "2021-10-01T05:30:00+05:30"},
                "datetimeTo": {"local": "2021-10-01T06:30:00+05:30"},
            },
            "coverage": {"percentComplete": 100.0},
            "value": 200.0,
        },
    ]

    df = convert_sensor_hours_to_series(raw_records, parameter_name="pm25")
    # Only 1 record retained, value is exactly 100.0 (NOT (100 + 200)/2 = 150.0)
    assert len(df) == 1
    assert df["Timestamp"].iloc[0].hour == 5
    assert df["Timestamp"].iloc[0].minute == 0
    assert df["pm25"].iloc[0] == 100.0


def test_timestamp_convention_and_ist_conversion():
    """Verify that hourly timestamps are consistently converted to Indian Standard Time (Asia/Kolkata)."""
    # 00:00:00 UTC is 05:30:00 IST (+05:30)
    utc_str = "2021-10-01T00:00:00Z"
    ts_str = parse_openaq_datetime_to_ist(utc_str)
    assert str(ts_str.tz) == "Asia/Kolkata"
    assert ts_str.hour == 5
    assert ts_str.minute == 30
    assert ts_str.day == 1

    # OpenAQ v3 dict format in 'period': {"utc": "2021-10-01T18:30:00Z"} -> 00:00 Next Day in IST
    dict_dt = {"utc": "2021-10-01T18:30:00Z"}
    ts_dict = parse_openaq_datetime_to_ist(dict_dt)
    assert str(ts_dict.tz) == "Asia/Kolkata"
    assert ts_dict.hour == 0
    assert ts_dict.day == 2


def test_cpcb_output_layout_with_source_column():
    """Verify output matches load_cpcb.py column layout plus 'source' set to 'openaq'."""
    times = pd.date_range("2021-10-01 00:00", periods=3, freq="h", tz="Asia/Kolkata")
    df_st1 = pd.DataFrame({
        "Timestamp": times,
        "station": "anand_vihar_new_delhi_dpcc",
        "pm25": [50.0, 60.0, 70.0],
        "pm10": [100.0, 1000.0, 120.0],  # 1000 should set pm10_capped
    })

    out_df = build_cpcb_layout_dataframe([df_st1], validate_identical_pm25=True)

    # Check exact columns
    assert list(out_df.columns) == CPCB_COLUMNS
    assert "source" in out_df.columns
    assert (out_df["source"] == "openaq").all()

    # Check Timestamp IST
    assert str(out_df["Timestamp"].dt.tz) == "Asia/Kolkata"

    # Check station slug
    assert out_df["station"].iloc[0] == "anand_vihar_new_delhi_dpcc"

    # Check capping flag
    assert out_df["pm10_capped"].tolist() == [False, True, False]

    # Check unpopulated pollutants are NaN
    assert out_df["no2"].isna().all()
    assert out_df["so2"].isna().all()
    assert out_df["co"].isna().all()
    assert out_df["ws"].isna().all()


def test_guard_fails_on_identical_pm25_series():
    """Verify that build_cpcb_layout_dataframe fails loudly on identical PM2.5 series across stations."""
    times = pd.date_range("2021-10-01 00:00", periods=5, freq="h", tz="Asia/Kolkata")
    identical_vals = [45.0, 50.0, 60.0, 55.0, 70.0]

    df_st1 = pd.DataFrame({"Timestamp": times, "station": "station_alpha", "pm25": identical_vals, "pm10": 100.0})
    df_st2 = pd.DataFrame({"Timestamp": times, "station": "station_beta", "pm25": identical_vals, "pm10": 120.0})

    with pytest.raises(ValueError, match="Duplicate PM2.5 series detected"):
        build_cpcb_layout_dataframe([df_st1, df_st2], validate_identical_pm25=True)


def test_pagination_handling():
    """Verify that fetch_paginated_results fetches across multiple pages."""
    mock_pages = {
        1: {
            "meta": {"page": 1, "limit": 2, "found": 4},
            "results": [
                {"id": 1, "value": 45.0, "datetime": {"utc": "2021-10-01T00:00:00Z"}},
                {"id": 2, "value": 50.0, "datetime": {"utc": "2021-10-01T01:00:00Z"}},
            ],
        },
        2: {
            "meta": {"page": 2, "limit": 2, "found": 4},
            "results": [
                {"id": 3, "value": 55.0, "datetime": {"utc": "2021-10-01T02:00:00Z"}},
                {"id": 4, "value": 60.0, "datetime": {"utc": "2021-10-01T03:00:00Z"}},
            ],
        },
    }

    mock_request = MagicMock()
    mock_request.side_effect = lambda url, headers, params: mock_pages[params["page"]]

    results = fetch_paginated_results(
        endpoint_url="https://api.openaq.org/v3/sensors/123/hours",
        headers={"X-API-Key": "test-key"},
        params={"limit": 2},
        limit=2,
        request_fn=mock_request,
    )

    assert len(results) == 4
    assert [r["id"] for r in results] == [1, 2, 3, 4]
    assert mock_request.call_count == 2


def test_window_caching_and_skipping(tmp_path: Path):
    """Verify that cached response windows are read from disk and network call is skipped."""
    cache_dir = tmp_path / "openaq_cache"
    cache_dir.mkdir()

    station = "test_station"
    param = "pm25"
    dt_from = "2021-10-01T00:00:00Z"
    dt_to = "2021-11-01T00:00:00Z"

    mock_request = MagicMock()
    mock_request.return_value = {
        "meta": {"found": 1},
        "results": [{"value": 110.0, "datetime": {"utc": "2021-10-01T05:00:00Z"}}],
    }

    # 1. First fetch -> cache miss
    is_cached_1, results_1 = fetch_and_cache_sensor_window(
        sensor_id=999,
        station=station,
        parameter=param,
        datetime_from=dt_from,
        datetime_to=dt_to,
        headers={"X-API-Key": "mock"},
        cache_dir=cache_dir,
        request_fn=mock_request,
    )
    assert is_cached_1 is False
    assert len(results_1) == 1
    assert mock_request.call_count == 1

    # 2. Second fetch -> cache hit
    is_cached_2, results_2 = fetch_and_cache_sensor_window(
        sensor_id=999,
        station=station,
        parameter=param,
        datetime_from=dt_from,
        datetime_to=dt_to,
        headers={"X-API-Key": "mock"},
        cache_dir=cache_dir,
        request_fn=mock_request,
    )
    assert is_cached_2 is True
    assert len(results_2) == 1
    assert mock_request.call_count == 1  # No additional network call


def test_make_station_slug():
    """Verify clean, distinct station slug generation."""
    slug1 = make_station_slug("Anand Vihar, New Delhi - DPCC", 235)
    assert slug1 == "anand_vihar_new_delhi_dpcc"

    slug2 = make_station_slug("Sector- 16A, Faridabad - HSPCB", 5617)
    assert slug2 == "sector_16a_faridabad_hspcb"


def test_fetch_measurements_selective_location_ids_and_skipping(tmp_path: Path):
    """Verify selective fetching for requested IDs and skipping of out-of-bound windows."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()

    mock_request = MagicMock()

    def mock_dispatch(url, headers, params=None):
        if url.endswith("/locations/6924"):
            return {
                "results": [{
                    "id": 6924,
                    "name": "Indirapuram, Ghaziabad - UPPCB",
                    "provider": {"name": "CPCB"},
                    "owner": {"name": "UPPCB"},
                    "coordinates": {"latitude": 28.6462, "longitude": 77.3581},
                }]
            }
        elif url.endswith("/locations/6924/sensors"):
            return {
                "results": [
                    {
                        "id": 19884,
                        "parameter": {"name": "pm25"},
                        # Sensor only active in 2021-2022
                        "datetimeFirst": {"utc": "2021-10-01T00:00:00Z"},
                        "datetimeLast": {"utc": "2022-02-28T23:59:59Z"},
                    }
                ]
            }
        elif "/sensors/19884/hours" in url:
            return {
                "meta": {"found": 1},
                "results": [
                    {
                        "period": {
                            "datetimeFrom": {"local": "2021-10-01T05:00:00+05:30"},
                            "datetimeTo": {"local": "2021-10-01T06:00:00+05:30"},
                        },
                        "coverage": {"percentComplete": 100.0},
                        "value": 85.0,
                    }
                ],
            }
        return {"results": []}

    mock_request.side_effect = mock_dispatch

    # Request seasons 2020-2021 and 2021-2022
    df, failures = fetch_measurements_for_locations(
        location_ids=[6924],
        api_key="mock",
        seasons=[(2020, 2021), (2021, 2022)],
        cache_dir=cache_dir,
        request_fn=mock_request,
    )

    assert len(df) == 1
    assert df["station"].iloc[0] == "indirapuram_ghaziabad_uppcb"
    assert df["source"].iloc[0] == "openaq"
    assert df["pm25"].iloc[0] == 85.0
    assert len(failures) == 0
