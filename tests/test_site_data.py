"""Tests for site/data.json structure, schema compliance, and data validity."""

import json
from pathlib import Path
import pytest

from make_site_data import (
    OUTPUT_FILE,
    build_site_data,
    write_site_data,
)


@pytest.fixture(scope="module")
def site_data_json_path(tmp_path_factory) -> Path:
    """Ensure site/data.json exists or generate into test location."""
    if OUTPUT_FILE.exists():
        return OUTPUT_FILE
    # Generate if not yet written
    return write_site_data(OUTPUT_FILE)


def test_schema_keys_exist(site_data_json_path: Path):
    """Verify that all required top-level and nested schema keys exist."""
    with open(site_data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Top-level keys
    required_top_keys = [
        "meta",
        "label",
        "stations",
        "breakpoints",
        "selftest_vectors",
        "inversion_examples",
    ]
    for key in required_top_keys:
        assert key in data, f"Missing top-level key: {key}"

    # 2. Meta keys
    meta = data["meta"]
    assert "generated_at" in meta
    assert "sources" in meta
    assert "station_display_names" in meta
    assert "breakpoints_verified" in meta
    assert "fire_available" in meta
    assert isinstance(meta["breakpoints_verified"], bool)
    assert isinstance(meta["fire_available"], bool)

    # 3. Label specification
    assert data["label"] == "PM-based AQI (PM2.5 and PM10 sub-indices)"

    # 4. Stations specification
    stations = data["stations"]
    assert len(stations) == 3
    for slug, st_info in stations.items():
        assert "station_slug" in st_info
        assert "display_name" in st_info
        assert "daily_aqi" in st_info
        assert "category_counts" in st_info
        assert "monthly_coverage" in st_info

        # Daily AQI has 151 calendar days (Oct 1 to Feb 28)
        daily_aqi = st_info["daily_aqi"]
        assert len(daily_aqi) == 151
        first_day = daily_aqi[0]
        assert "date" in first_day
        assert "pm25" in first_day
        assert "pm10" in first_day
        assert "aqi" in first_day
        assert "category" in first_day

        # Monthly coverage has 5 months (Oct-Feb)
        cov = st_info["monthly_coverage"]
        assert len(cov) == 5
        for m in cov:
            assert "month" in m
            assert "hours_possible" in m
            assert "valid_hours_pm25" in m
            assert "valid_hours_pm10" in m
            assert "passing_days" in m


def test_no_nan_in_json(site_data_json_path: Path):
    """Verify strictly that no NaN values exist in the JSON output (missing values must be null)."""
    with open(site_data_json_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    import re
    # Raw string scan for invalid JSON NaN identifiers (without false positive on 'anand')
    assert "NaN" not in raw_text, "Found raw 'NaN' in JSON file; missing values must be null."
    assert not re.search(r":\s*nan\b", raw_text, re.IGNORECASE), "Found ': nan' in JSON file; missing values must be null."
    assert "Infinity" not in raw_text, "Found raw 'Infinity' in JSON file."

    # Strict parse rejecting NaN constants
    def fail_on_constant(val):
        raise ValueError(f"Encountered non-standard JSON constant: {val}")

    parsed = json.loads(raw_text, parse_constant=fail_on_constant)
    assert isinstance(parsed, dict)

    # Confirm nulls exist in daily AQI where values are missing
    stations = parsed["stations"]
    for slug, st_info in stations.items():
        null_aqi_days = [d for d in st_info["daily_aqi"] if d["aqi"] is None]
        # Every station has some days without valid readings (e.g. early Oct or outages)
        assert len(null_aqi_days) > 0


def test_station_names_contain_no_16a(site_data_json_path: Path):
    """Verify that station names and slugs contain no reference to Sector 16A."""
    with open(site_data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check meta display names
    for slug, name in data["meta"]["station_display_names"].items():
        assert "16a" not in slug.lower(), f"Station slug '{slug}' must not refer to 16A"
        assert "16a" not in name.lower(), f"Station name '{name}' must not refer to 16A"

    # Check stations dictionary
    stations = data["stations"]
    for slug, st in stations.items():
        assert "16a" not in slug.lower(), f"Station slug '{slug}' must not refer to 16A"
        assert "16a" not in st["display_name"].lower(), f"Station display name '{st['display_name']}' must not refer to 16A"

    # Sector 11 must be present with its proper OpenAQ display name
    assert "sector_11_faridabad_hspcb" in stations
    assert stations["sector_11_faridabad_hspcb"]["display_name"] == "Sector 11, Faridabad - HSPCB"


def test_selftest_vectors_count_and_content(site_data_json_path: Path):
    """Verify selftest vectors count is at least 30, with PM10-only and category boundary cases."""
    with open(site_data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = data["selftest_vectors"]
    assert len(vectors) >= 30, f"Expected at least 30 selftest vectors, got {len(vectors)}"

    # Check required fields
    has_pm10_only = False
    categories_covered = set()

    for vec in vectors:
        assert "input" in vec
        assert "expected_aqi" in vec
        assert "expected_category" in vec
        inp = vec["input"]
        assert isinstance(inp, dict)
        assert len(inp) > 0

        # Check for PM10-only case
        if list(inp.keys()) == ["PM10"]:
            has_pm10_only = True

        cat = vec["expected_category"]
        if cat:
            categories_covered.add(cat)

    assert has_pm10_only, "Selftest vectors must contain at least one PM10-only case."

    # Verify coverage across all standard CPCB categories
    expected_categories = {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
    assert expected_categories.issubset(categories_covered), f"Missing categories: {expected_categories - categories_covered}"


def test_inversion_examples(site_data_json_path: Path):
    """Verify 3 synthetic inversion profiles are properly formatted and labelled."""
    with open(site_data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inv_examples = data["inversion_examples"]
    assert len(inv_examples) == 3

    types_found = set()
    for ex in inv_examples:
        assert ex.get("label") == "synthetic"
        p_type = ex.get("profile_type")
        types_found.add(p_type)
        assert "height_m" in ex
        assert "temp_c" in ex
        assert "layers" in ex
        assert len(ex["height_m"]) == len(ex["temp_c"])

    assert types_found == {"surface-based", "elevated", "none"}
