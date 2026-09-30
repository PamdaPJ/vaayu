"""OpenAQ Data Fetcher and Preprocessor for VAAYU.

Implements data retrieval and discovery from the OpenAQ v3 REST API (docs.openaq.org):
- Authentication via X-API-Key header from OPENAQ_API_KEY environment variable.
- Location discovery for Faridabad (25km radius and name matching) with PM2.5/PM10
  sensor periods (datetimeFirst and datetimeLast).
- Inspection of confirmed stations Anand Vihar (235) and Indirapuram (6924).
- Measurement retrieval restricted to explicit --location-ids passed by the user.
- Target windows defined as winter seasons:
    * Oct 1 2020 to Feb 28 2021
    * Oct 1 2021 to Feb 28 2022
    * Oct 1 2025 to Feb 28 2026
- Skipping of windows outside a sensor's [datetimeFirst, datetimeLast] with clear logging.
- Timestamp convention analysis and consistent conversion to Indian Standard Time (IST).
- Strict adherence to no-filling / no-interpolation rule.
- Output formatted in the identical column layout as load_cpcb.py with `source` column
  set to 'openaq' and duplicate series validation.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv
import numpy as np
import pandas as pd

from load_cpcb import check_no_identical_pm25_series

# Ensure console supports utf-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# OpenAQ API v3 configuration
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
DEFAULT_RATE_LIMIT_RPS = 60  # 60 requests per minute on free tier
CACHE_DIR = Path("data/raw/openaq")

# Standard CPCB column layout matching load_cpcb.py, with 'source' column
CPCB_COLUMNS = [
    "Timestamp",
    "station",
    "source",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "nh3",
    "at",
    "rh",
    "ws",
    "wd",
    "sr",
    "pm10_capped",
]

# Confirmed baseline locations
CONFIRMED_LOCATIONS = {
    235: "Anand Vihar (Delhi)",
    6924: "Indirapuram (Ghaziabad)",
}

# Faridabad city center coordinates for 25km radius search
FARIDABAD_CENTER_COORDS = (28.4089, 77.3178)


def get_openaq_api_key() -> str:
    """Load OPENAQ_API_KEY from environment or .env file.

    CRITICAL: Never print, log, or persist this key.
    """
    load_dotenv()
    key = os.getenv("OPENAQ_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "OPENAQ_API_KEY environment variable is not set. "
            "Please configure OPENAQ_API_KEY in your .env file."
        )
    return key.strip()


def build_auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Construct HTTP headers required for OpenAQ v3 API requests."""
    key = api_key if api_key is not None else get_openaq_api_key()
    return {
        "X-API-Key": key,
        "Accept": "application/json",
        "User-Agent": "VAAYU-AirQualityResearch/1.0",
    }


def make_http_request(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Perform an authenticated HTTP GET request with urllib and return JSON response."""
    if params:
        encoded_params = urllib.parse.urlencode(params)
        full_url = f"{url}?{encoded_params}" if "?" not in url else f"{url}&{encoded_params}"
    else:
        full_url = url

    req = urllib.request.Request(full_url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read().decode("utf-8")
        return json.loads(content)


def request_with_retry(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    request_fn: Callable = make_http_request,
) -> Dict[str, Any]:
    """Execute HTTP GET request with exponential backoff on 429 and 5xx errors."""
    backoff = initial_backoff
    last_exception = None

    for attempt in range(max_retries):
        try:
            return request_fn(url, headers=headers, params=params)
        except urllib.error.HTTPError as exc:
            last_exception = exc
            # 429 Too Many Requests or 5xx Server Errors
            if exc.code == 429 or 500 <= exc.code < 600:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                sleep_sec = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                print(
                    f"Warning: HTTP {exc.code} for request. Retrying in {sleep_sec:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})...",
                    file=sys.stderr,
                )
                time.sleep(sleep_sec)
                backoff *= 2.0
            else:
                # Client error (400, 401, 403, 404, etc.) -> re-raise immediately
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exception = exc
            print(
                f"Warning: Network error ({exc}). Retrying in {backoff:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})...",
                file=sys.stderr,
            )
            time.sleep(backoff)
            backoff *= 2.0

    raise RuntimeError(
        f"Failed to fetch {url} after {max_retries} attempts: {last_exception}"
    )


def fetch_paginated_results(
    endpoint_url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    limit: int = 1000,
    request_fn: Callable = request_with_retry,
) -> List[Dict[str, Any]]:
    """Retrieve all pages of results from an OpenAQ v3 paginated endpoint.

    Uses `page` and `limit` query parameters, terminating when `found` count
    is reached or a page returns fewer items than `limit`.
    """
    params_dict = dict(params) if params else {}
    params_dict["limit"] = limit
    page = 1
    accumulated_results: List[Dict[str, Any]] = []

    while True:
        params_dict["page"] = page
        response_json = request_fn(endpoint_url, headers=headers, params=params_dict)

        results = response_json.get("results", [])
        if not results:
            break

        accumulated_results.extend(results)

        meta = response_json.get("meta", {})
        found = meta.get("found")
        if found is not None and isinstance(found, int) and len(accumulated_results) >= found:
            break
        if len(results) < limit:
            break

        page += 1

    return accumulated_results


def get_location_details_and_pm_sensors(
    location_id: int,
    headers: Dict[str, str],
    request_fn: Callable = request_with_retry,
) -> Dict[str, Any]:
    """Retrieve detailed location metadata and its PM2.5/PM10 sensors with date bounds."""
    # 1. Location metadata
    loc_url = f"{OPENAQ_BASE_URL}/locations/{location_id}"
    try:
        loc_resp = request_fn(loc_url, headers=headers)
        loc_data = loc_resp.get("results", [{}])[0]
    except Exception as exc:
        print(f"Error fetching location {location_id}: {exc}", file=sys.stderr)
        loc_data = {"id": location_id, "name": f"Location {location_id}"}

    # 2. Sensors for this location
    sensors_url = f"{OPENAQ_BASE_URL}/locations/{location_id}/sensors"
    try:
        sensors_resp = request_fn(sensors_url, headers=headers)
        sensors = sensors_resp.get("results", [])
    except Exception as exc:
        print(f"Error fetching sensors for location {location_id}: {exc}", file=sys.stderr)
        sensors = loc_data.get("sensors", [])

    pm_sensors = []
    has_pm25 = False
    has_pm10 = False

    for s in sensors:
        param = s.get("parameter", {})
        pname = param.get("name", "").lower()
        if pname in ["pm25", "pm10"]:
            if pname == "pm25":
                has_pm25 = True
            elif pname == "pm10":
                has_pm10 = True

            dt_first = s.get("datetimeFirst")
            dt_last = s.get("datetimeLast")
            pm_sensors.append({
                "sensor_id": s.get("id"),
                "parameter": pname,
                "datetime_first": dt_first.get("utc") if isinstance(dt_first, dict) else dt_first,
                "datetime_last": dt_last.get("utc") if isinstance(dt_last, dict) else dt_last,
            })

    # Provider / owner extraction
    provider = loc_data.get("provider")
    pname = provider.get("name") if isinstance(provider, dict) else provider
    owner = loc_data.get("owner")
    oname = owner.get("name") if isinstance(owner, dict) else owner

    if pname and oname and pname != oname:
        provider_owner = f"{pname} / {oname}"
    else:
        provider_owner = pname or oname or "Unknown"

    coords = loc_data.get("coordinates", {})
    return {
        "id": loc_data.get("id", location_id),
        "name": loc_data.get("name", "Unknown"),
        "provider_owner": provider_owner,
        "latitude": coords.get("latitude") if isinstance(coords, dict) else None,
        "longitude": coords.get("longitude") if isinstance(coords, dict) else None,
        "has_pm25": has_pm25,
        "has_pm10": has_pm10,
        "pm_sensors": pm_sensors,
    }


def discover_all_faridabad_locations(
    api_key: Optional[str] = None,
    request_fn: Callable = request_with_retry,
) -> List[Dict[str, Any]]:
    """List ALL OpenAQ locations within 25 km of Faridabad and by name containing 'Faridabad'."""
    headers = build_auth_headers(api_key)
    locations_by_id: Dict[int, Dict[str, Any]] = {}

    # Query 1: 25 km radius around Faridabad center
    lat, lon = FARIDABAD_CENTER_COORDS
    radius_params = {
        "coordinates": f"{lat:.4f},{lon:.4f}",
        "radius": 25000,
        "limit": 100,
    }
    radius_locs = fetch_paginated_results(
        f"{OPENAQ_BASE_URL}/locations",
        headers=headers,
        params=radius_params,
        limit=100,
        request_fn=request_fn,
    )
    for loc in radius_locs:
        lid = loc.get("id")
        if lid:
            locations_by_id[lid] = loc

    # Query 2: Search across country locations for "Faridabad" in name
    country_params = {"iso": "IN", "limit": 1000}
    try:
        in_locs = fetch_paginated_results(
            f"{OPENAQ_BASE_URL}/locations",
            headers=headers,
            params=country_params,
            limit=1000,
            request_fn=request_fn,
        )
        for loc in in_locs:
            if "faridabad" in loc.get("name", "").lower():
                lid = loc.get("id")
                if lid:
                    locations_by_id[lid] = loc
    except Exception as exc:
        print(f"Notice: Country-level location search encountered: {exc}", file=sys.stderr)

    # For each discovered location, fetch sensor details
    discovered_details = []
    sorted_ids = sorted(locations_by_id.keys())
    for lid in sorted_ids:
        details = get_location_details_and_pm_sensors(lid, headers=headers, request_fn=request_fn)
        discovered_details.append(details)

    return discovered_details


def print_location_pm_summary(loc: Dict[str, Any], label: Optional[str] = None) -> None:
    """Print formatted summary of a location and its PM sensors."""
    header = f"=== {label} [ID: {loc['id']}] ===" if label else f"Location ID: {loc['id']}"
    print(header)
    print(f"  • Name:           {loc['name']}")
    print(f"  • Provider/Owner: {loc['provider_owner']}")
    print(f"  • Coordinates:    Lat {loc['latitude']}, Lon {loc['longitude']}")
    print(f"  • Has PM2.5:      {'Yes' if loc['has_pm25'] else 'No'} | Has PM10: {'Yes' if loc['has_pm10'] else 'No'}")

    if loc["pm_sensors"]:
        print("  • PM Sensors:")
        for s in loc["pm_sensors"]:
            p = s["parameter"].upper()
            sid = s["sensor_id"]
            d_first = s["datetime_first"] or "N/A"
            d_last = s["datetime_last"] or "N/A"
            print(f"      - Sensor {sid} ({p}): {d_first}  -->  {d_last}")
    else:
        print("  • PM Sensors:     None reported")
    print("-" * 78)


def print_timestamp_convention_report() -> None:
    """Print analysis of OpenAQ v3 hourly timestamp convention and IST conversion."""
    print("=" * 78)
    print("OpenAQ v3 Hourly Timestamp Convention Analysis")
    print("=" * 78)
    print("1. Documentation Specification (docs.openaq.org):")
    print("   • In OpenAQ API v3, aggregated hourly data (/v3/sensors/{id}/hours) provides")
    print("     explicit period bounds inside the 'period' object:")
    print("       - 'datetimeFrom': ISO-8601 UTC start of the 1-hour interval")
    print("       - 'datetimeTo':   ISO-8601 UTC end of the 1-hour interval")
    print("   • When a single 'datetime' attribute is present (e.g. in /measurements or")
    print("     summary endpoints), OpenAQ documents that the timestamp label marks the")
    print("     END of the observation period (when the hourly sample completed).")
    print("2. CPCB Alignment & IST Conversion:")
    print("   • Indian Standard Time (IST) is UTC+05:30 (Asia/Kolkata).")
    print("   • CPCB station data conventionally timestamps each reading at the hour mark.")
    print("   • VAAYU standardizes all timestamps by parsing the interval UTC timestamp")
    print("     and converting strictly to 'Asia/Kolkata' timezone without altering the interval.")
    print("=" * 78)


def run_discovery_mode(api_key: Optional[str] = None) -> None:
    """Execute complete discovery report for Anand Vihar, Indirapuram, and Faridabad."""
    headers = build_auth_headers(api_key)

    print("=" * 78)
    print("OpenAQ v3 Station Discovery & Sensor Period Report")
    print("=" * 78)

    # Print timestamp convention report
    print_timestamp_convention_report()

    # 1. Print details for confirmed Anand Vihar (235) and Indirapuram (6924)
    print("\n[PART 1] Confirmed Anchor Stations:")
    print("=" * 78)
    for lid, station_label in CONFIRMED_LOCATIONS.items():
        details = get_location_details_and_pm_sensors(lid, headers=headers)
        print_location_pm_summary(details, label=f"Confirmed Station: {station_label}")

    # 2. Discover ALL locations within 25km of Faridabad or containing 'Faridabad'
    print("\n[PART 2] OpenAQ Locations within 25km of Faridabad or Named 'Faridabad':")
    print("=" * 78)
    faridabad_locs = discover_all_faridabad_locations(api_key=api_key)

    # Highlight locations with 'Faridabad' in the name first
    name_matches = [l for l in faridabad_locs if "faridabad" in l["name"].lower()]
    other_matches = [l for l in faridabad_locs if "faridabad" not in l["name"].lower()]

    print(f"\n--- Stations with 'Faridabad' in name ({len(name_matches)} found) ---")
    for loc in name_matches:
        print_location_pm_summary(loc)

    print(f"\n--- Other stations within 25 km radius ({len(other_matches)} found) ---")
    for loc in other_matches:
        print_location_pm_summary(loc)

    print("=" * 78)
    print("ACTION REQUIRED:")
    print("Never auto-selecting a Faridabad station.")
    print("Please inspect the PM sensor periods above and select your desired station location ID.")
    print("Then fetch data by running:")
    print("  python fetch_openaq.py --location-ids 235,6924,<FARIDABAD_LOCATION_ID>")
    print("=" * 78)


def is_leap_year(year: int) -> bool:
    """Determine whether a calendar year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def get_february_end_day(year: int) -> int:
    """Return the final day of February (29 for leap years, 28 otherwise)."""
    return 29 if is_leap_year(year) else 28


def generate_season_windows(
    seasons: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[str, str, str, str]]:
    """Generate monthly request windows for the Oct 1 to Feb 28/29 winter smog seasons.

    Winter Seasons specified:
      - 2020-2021: Oct 1 2020 to Feb 28 2021
      - 2021-2022: Oct 1 2021 to Feb 28 2022
      - 2025-2026: Oct 1 2025 to Feb 28 2026

    Returns list of tuples: (datetime_from_iso, datetime_to_iso, season_label, month_name)
    """
    if seasons is None:
        seasons = [(2020, 2021), (2021, 2022), (2025, 2026)]

    windows = []
    for start_yr, end_yr in seasons:
        season_label = f"{start_yr}-{end_yr}"
        feb_days = get_february_end_day(end_yr)

        season_months = [
            (f"{start_yr}-10-01T00:00:00Z", f"{start_yr}-11-01T00:00:00Z", "Oct"),
            (f"{start_yr}-11-01T00:00:00Z", f"{start_yr}-12-01T00:00:00Z", "Nov"),
            (f"{start_yr}-12-01T00:00:00Z", f"{end_yr}-01-01T00:00:00Z", "Dec"),
            (f"{end_yr}-01-01T00:00:00Z", f"{end_yr}-02-01T00:00:00Z", "Jan"),
            (f"{end_yr}-02-01T00:00:00Z", f"{end_yr}-03-01T00:00:00Z", "Feb"),
        ]

        for dt_from, dt_to, m_name in season_months:
            windows.append((dt_from, dt_to, season_label, m_name))

    return windows


def should_skip_window_for_sensor(
    dt_from_iso: str,
    dt_to_iso: str,
    datetime_first: Optional[str],
    datetime_last: Optional[str],
    sensor_id: int,
    parameter: str,
    location_name: str,
    location_id: int,
) -> Tuple[bool, Optional[str]]:
    """Determine if a window falls completely outside a sensor's active date range.

    Returns: (should_skip, reason)
    """
    w_start = pd.to_datetime(dt_from_iso, utc=True)
    w_end = pd.to_datetime(dt_to_iso, utc=True)

    if datetime_first:
        s_first = pd.to_datetime(datetime_first, utc=True)
        if w_end <= s_first:
            reason = (
                f"Skipping Loc {location_id} ({location_name}) sensor {sensor_id} ({parameter.upper()}) "
                f"window {dt_from_iso} to {dt_to_iso}: window ends before sensor's datetimeFirst ({datetime_first})"
            )
            return True, reason

    if datetime_last:
        s_last = pd.to_datetime(datetime_last, utc=True)
        if w_start >= s_last:
            reason = (
                f"Skipping Loc {location_id} ({location_name}) sensor {sensor_id} ({parameter.upper()}) "
                f"window {dt_from_iso} to {dt_to_iso}: window starts after sensor's datetimeLast ({datetime_last})"
            )
            return True, reason

    return False, None


def parse_openaq_datetime_to_ist(raw_dt: Any) -> pd.Timestamp:
    """Parse an OpenAQ timestamp field to Indian Standard Time (Asia/Kolkata)."""
    if isinstance(raw_dt, dict):
        dt_str = raw_dt.get("utc") or raw_dt.get("local")
    else:
        dt_str = str(raw_dt)

    ts = pd.to_datetime(dt_str)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("Asia/Kolkata")


def aggregate_subhourly_to_hourly(
    df: pd.DataFrame,
    time_col: str = "Timestamp",
    value_col: str = "value",
    min_samples_per_hour: int = 3,
    expected_samples_per_hour: int = 4,
    resolution_adaptive: bool = True,
) -> pd.DataFrame:
    """Aggregate sub-hourly readings to hourly averages.

    - Drops exact duplicate (Timestamp, value) rows.
    - If data is 15-minute sub-hourly (4 expected samples), keeps the hour only if
      at least min_samples_per_hour samples exist (default 3 of 4).
    - If data is 30-minute sub-hourly (2 expected samples), keeps the hour if at least
      1 sample exists and computes the mean.
    - If data is already hourly (1 expected sample), keeps valid readings directly.
    - Never interpolates or fills missing hours.
    """
    if df.empty:
        return pd.DataFrame(columns=[time_col, value_col])

    # 1. Drop exact duplicates
    df_clean = df.drop_duplicates(subset=[time_col, value_col]).copy()
    if df_clean.empty:
        return pd.DataFrame(columns=[time_col, value_col])

    # 2. Floor timestamp to hour
    df_clean["hour_bin"] = df_clean[time_col].dt.floor("h")

    # 3. Assess resolution
    sample_counts_per_hour = df_clean.groupby("hour_bin")[value_col].count()
    max_samples = sample_counts_per_hour.max()

    # Determine required sample threshold
    if resolution_adaptive:
        if max_samples >= 4:
            # 15-minute data: enforce 3 of 4 rule
            req_threshold = min_samples_per_hour
        elif max_samples == 2:
            # 30-minute data: at least 1 sample required
            req_threshold = 1
        else:
            # Hourly data: 1 sample
            req_threshold = 1
    else:
        req_threshold = min_samples_per_hour

    # 4. Group and compute hourly average
    grouped = df_clean.groupby("hour_bin")[value_col].agg(["count", "mean"]).reset_index()
    valid_hours = grouped[grouped["count"] >= req_threshold]

    result = valid_hours[["hour_bin", "mean"]].rename(columns={"hour_bin": time_col, "mean": value_col})
    return result.sort_values(time_col).reset_index(drop=True)


def convert_sensor_hours_to_series(
    records: List[Dict[str, Any]],
    parameter_name: str,
    min_coverage_pct: float = 75.0,
) -> pd.DataFrame:
    """Convert raw OpenAQ hourly sensor records to a DataFrame with IST Timestamps.

    Rules:
    - The API records are already hourly aggregates.
    - Keep only records whose period.datetimeFrom is at minute 00 in IST (drops half-hour offset windows).
    - Drop records with coverage.percentComplete below min_coverage_pct (default: 75.0%).
    - Do not average overlapping windows.
    - Use datetimeFrom as the timestamp (hour start).
    - Drop exact duplicate (Timestamp, value) rows.
    - Never fill or interpolate missing hours.
    """
    if not records:
        return pd.DataFrame(columns=["Timestamp", parameter_name])

    rows = []
    for rec in records:
        val = rec.get("value")
        if val is None or pd.isna(val):
            continue

        p_obj = rec.get("period", {})
        if isinstance(p_obj, dict):
            raw_dt = p_obj.get("datetimeFrom")
        else:
            raw_dt = None

        if raw_dt is None:
            raw_dt = rec.get("datetime")

        if raw_dt is None:
            continue

        ts_ist = parse_openaq_datetime_to_ist(raw_dt)

        # 1. Keep only records whose period.datetimeFrom is at minute 00 in IST
        if ts_ist.minute != 0:
            continue

        # 2. Drop records with coverage.percentComplete below min_coverage_pct
        cov_obj = rec.get("coverage", {})
        if isinstance(cov_obj, dict):
            cov_pct = cov_obj.get("percentComplete")
            if cov_pct is not None and cov_pct < min_coverage_pct:
                continue

        rows.append({
            "Timestamp": ts_ist,
            parameter_name: float(val),
        })

    if not rows:
        return pd.DataFrame(columns=["Timestamp", parameter_name])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Timestamp", parameter_name])
    return df.sort_values("Timestamp").reset_index(drop=True)


def fetch_and_cache_sensor_window(
    sensor_id: int,
    station: str,
    parameter: str,
    datetime_from: str,
    datetime_to: str,
    headers: Dict[str, str],
    cache_dir: Path = CACHE_DIR,
    request_fn: Callable = request_with_retry,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Fetch hourly data for one sensor window, caching raw JSON to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    from_tag = datetime_from[:10].replace("-", "")
    to_tag = datetime_to[:10].replace("-", "")
    cache_filename = f"{station}_{parameter}_{from_tag}_{to_tag}.json"
    cache_path = cache_dir / cache_filename

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return True, data.get("results", [])
        except Exception:
            pass

    endpoint = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours"
    params = {
        "datetime_from": datetime_from,
        "datetime_to": datetime_to,
        "limit": 1000,
    }

    results = fetch_paginated_results(
        endpoint, headers=headers, params=params, limit=1000, request_fn=request_fn
    )

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"found": len(results)}, "results": results}, f, indent=2)

    return False, results


def make_station_slug(location_name: str, location_id: Optional[int] = None) -> str:
    """Generate a clean, distinct station slug for a location ID."""
    clean_name = re.sub(r"[^a-zA-Z0-9]+", "_", location_name).strip("_").lower()
    return clean_name


def build_cpcb_layout_dataframe(
    station_dfs: List[pd.DataFrame],
    validate_identical_pm25: bool = True,
) -> pd.DataFrame:
    """Format and combine station dataframes into the exact CPCB hourly schema.

    Schema: Timestamp in IST, station, source ('openaq'), pm25, pm10, other pollutants
    left empty (NaN), and pm10_capped flag.
    Never fills or interpolates missing records.
    """
    if not station_dfs:
        return pd.DataFrame(columns=CPCB_COLUMNS)

    combined = pd.concat(station_dfs, ignore_index=True)

    # Ensure source column is set to openaq
    combined["source"] = "openaq"

    for col in CPCB_COLUMNS:
        if col not in combined.columns:
            if col == "pm10_capped":
                combined[col] = combined["pm10"] == 1000
            elif col == "source":
                combined[col] = "openaq"
            else:
                combined[col] = np.nan

    combined["pm10_capped"] = combined["pm10"] == 1000
    combined = combined.sort_values(by=["station", "Timestamp"]).reset_index(drop=True)
    result_df = combined[CPCB_COLUMNS]

    if validate_identical_pm25:
        check_no_identical_pm25_series(result_df)

    return result_df


def fetch_measurements_for_locations(
    location_ids: List[int],
    api_key: Optional[str] = None,
    seasons: Optional[List[Tuple[int, int]]] = None,
    cache_dir: Path = CACHE_DIR,
    request_fn: Callable = request_with_retry,
) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch measurements for explicitly specified location IDs only.

    Winter seasons: 2020-2021, 2021-2022, 2025-2026.
    Skips any window outside a sensor's active datetime bounds with clear reasons logged.
    Never fills or interpolates missing values.
    """
    headers = build_auth_headers(api_key)
    windows = generate_season_windows(seasons=seasons)
    all_station_dfs = []
    failures = []

    print_timestamp_convention_report()
    print(f"\nFetching measurements for Location IDs: {location_ids} across winter seasons...")

    for loc_id in location_ids:
        # Get sensor details for location
        details = get_location_details_and_pm_sensors(loc_id, headers=headers, request_fn=request_fn)
        loc_name = details["name"]
        station_slug = make_station_slug(loc_name, loc_id)

        pm25_sensors = [s for s in details["pm_sensors"] if s["parameter"] == "pm25"]
        pm10_sensors = [s for s in details["pm_sensors"] if s["parameter"] == "pm10"]

        all_pm25_records = []
        all_pm10_records = []

        # 1. Fetch PM2.5 sensors
        for s_info in pm25_sensors:
            s_id = s_info["sensor_id"]
            d_first = s_info["datetime_first"]
            d_last = s_info["datetime_last"]

            for dt_from, dt_to, season_lbl, m_name in windows:
                skip, reason = should_skip_window_for_sensor(
                    dt_from_iso=dt_from,
                    dt_to_iso=dt_to,
                    datetime_first=d_first,
                    datetime_last=d_last,
                    sensor_id=s_id,
                    parameter="pm25",
                    location_name=loc_name,
                    location_id=loc_id,
                )
                if skip:
                    print(f"  • {reason}")
                    continue

                try:
                    is_cached, results = fetch_and_cache_sensor_window(
                        sensor_id=s_id,
                        station=station_slug,
                        parameter="pm25",
                        datetime_from=dt_from,
                        datetime_to=dt_to,
                        headers=headers,
                        cache_dir=cache_dir,
                        request_fn=request_fn,
                    )
                    cache_status = "CACHED" if is_cached else "FETCHED"
                    print(f"  [{cache_status}] Loc {loc_id} ({loc_name}) PM2.5 sensor {s_id} ({season_lbl} {m_name}): {len(results)} records")
                    all_pm25_records.extend(results)
                except Exception as exc:
                    err_msg = f"Loc {loc_id} ({loc_name}) PM2.5 sensor {s_id} window {dt_from} to {dt_to}: {exc}"
                    print(f"  [ERROR] {err_msg}", file=sys.stderr)
                    failures.append(err_msg)

        # 2. Fetch PM10 sensors
        for s_info in pm10_sensors:
            s_id = s_info["sensor_id"]
            d_first = s_info["datetime_first"]
            d_last = s_info["datetime_last"]

            for dt_from, dt_to, season_lbl, m_name in windows:
                skip, reason = should_skip_window_for_sensor(
                    dt_from_iso=dt_from,
                    dt_to_iso=dt_to,
                    datetime_first=d_first,
                    datetime_last=d_last,
                    sensor_id=s_id,
                    parameter="pm10",
                    location_name=loc_name,
                    location_id=loc_id,
                )
                if skip:
                    print(f"  • {reason}")
                    continue

                try:
                    is_cached, results = fetch_and_cache_sensor_window(
                        sensor_id=s_id,
                        station=station_slug,
                        parameter="pm10",
                        datetime_from=dt_from,
                        datetime_to=dt_to,
                        headers=headers,
                        cache_dir=cache_dir,
                        request_fn=request_fn,
                    )
                    cache_status = "CACHED" if is_cached else "FETCHED"
                    print(f"  [{cache_status}] Loc {loc_id} ({loc_name}) PM10 sensor {s_id} ({season_lbl} {m_name}): {len(results)} records")
                    all_pm10_records.extend(results)
                except Exception as exc:
                    err_msg = f"Loc {loc_id} ({loc_name}) PM10 sensor {s_id} window {dt_from} to {dt_to}: {exc}"
                    print(f"  [ERROR] {err_msg}", file=sys.stderr)
                    failures.append(err_msg)

        df_pm25 = convert_sensor_hours_to_series(all_pm25_records, "pm25")
        df_pm10 = convert_sensor_hours_to_series(all_pm10_records, "pm10")

        if df_pm25.empty and df_pm10.empty:
            print(f"Notice: No PM data retrieved for location {loc_id} ({loc_name}) in requested seasons.")
            continue

        merged = pd.merge(df_pm25, df_pm10, on="Timestamp", how="outer")
        merged["station"] = station_slug
        all_station_dfs.append(merged)

    final_df = build_cpcb_layout_dataframe(all_station_dfs)

    if failures:
        print("\nFailures encountered during fetch:")
        for fail in failures:
            print(f"  • {fail}")

    return final_df, failures


def main():
    parser = argparse.ArgumentParser(
        description="OpenAQ Station Discovery and PM2.5/PM10 Data Fetcher for VAAYU"
    )
    parser.add_argument(
        "--location-ids",
        type=str,
        default=None,
        help="Comma-separated OpenAQ location IDs to fetch (e.g. --location-ids 235,6924,5617)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        default=False,
        help="Run discovery mode to list Faridabad stations and sensor dates",
    )

    args = parser.parse_args()

    try:
        api_key = get_openaq_api_key()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.location_ids:
        raw_ids = [int(i.strip()) for i in args.location_ids.split(",") if i.strip().isdigit()]
        if not raw_ids:
            print("Error: No valid integer IDs provided to --location-ids.", file=sys.stderr)
            sys.exit(1)
        fetch_measurements_for_locations(raw_ids, api_key=api_key)
    else:
        run_discovery_mode(api_key=api_key)


if __name__ == "__main__":
    main()
