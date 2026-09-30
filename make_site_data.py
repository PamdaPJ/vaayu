"""Generate site/data.json for VAAYU web interface from real project data.

Inputs:
- data/processed/openaq_hourly.csv (2025-26 rows only; 2020-22 ignored due to low coverage)
- data/cpcb_breakpoints.csv (read dynamically via aqi.py; never hard-coded)
- inversion.py (for synthetic inversion examples)
- figures/fire_counts_by_year.png & overpass_hours.png (checks existence; sets fire_available accordingly)

Outputs:
- site/data.json adhering to strict schema rules with null for missing values (never NaN).
"""

from calendar import monthrange
from datetime import date, datetime, timezone
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import aqi
import inversion

DATA_DIR = Path("data")
PROCESSED_HOURLY_CSV = DATA_DIR / "processed" / "openaq_hourly.csv"
UNVERIFIED_MARKER = DATA_DIR / "UNVERIFIED"
OUTPUT_DIR = Path("site")
OUTPUT_FILE = OUTPUT_DIR / "data.json"

FIGURES_DIR = Path("figures")
FIRE_COUNTS_PNG = FIGURES_DIR / "fire_counts_by_year.png"
OVERPASS_HOURS_PNG = FIGURES_DIR / "overpass_hours.png"

# Display names exactly as listed on OpenAQ (Sector 11 must NOT be called Sector 16A)
STATION_METADATA = {
    "anand_vihar_new_delhi_dpcc": {
        "display_name": "Anand Vihar, New Delhi - DPCC",
        "city": "Delhi",
        "location_id": 235,
        "coordinates": {"latitude": 28.646835, "longitude": 77.316032},
    },
    "indirapuram_ghaziabad_uppcb": {
        "display_name": "Indirapuram, Ghaziabad - UPPCB",
        "city": "Ghaziabad",
        "location_id": 6924,
        "coordinates": {"latitude": 28.646233, "longitude": 77.358075},
    },
    "sector_11_faridabad_hspcb": {
        "display_name": "Sector 11, Faridabad - HSPCB",
        "city": "Faridabad",
        "location_id": 10908,
        "coordinates": {"latitude": 28.376058, "longitude": 77.315741},
    },
}

WINTER_2025_MONTHS = [
    (2025, 10),
    (2025, 11),
    (2025, 12),
    (2026, 1),
    (2026, 2),
]


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert float NaNs, infinities, and numpy scalars to JSON-safe types.

    Ensures that missing values serialize cleanly as null.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif pd.isna(obj):
        return None
    return obj


def load_2025_2026_openaq_data(csv_path: Path = PROCESSED_HOURLY_CSV) -> pd.DataFrame:
    """Load hourly OpenAQ data filtering exclusively to the 2025-2026 winter season."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed hourly file not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")

    # Filter to 2025-26 winter season only (2025-10-01 00:00 to 2026-03-01 00:00 IST)
    # Ignore 2020-22 rows because their coverage is too low
    start_ts = pd.Timestamp("2025-10-01 00:00:00", tz="Asia/Kolkata")
    end_ts = pd.Timestamp("2026-03-01 00:00:00", tz="Asia/Kolkata")

    df_2025 = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)].copy()
    return df_2025


def build_inversion_examples() -> List[Dict[str, Any]]:
    """Build 3 synthetic sounding profiles with outputs of inversion.find_inversions."""
    examples = []

    # 1. Surface-based inversion
    p1_h = [0.0, 100.0, 200.0, 300.0, 400.0]
    p1_t = [8.0, 10.0, 12.0, 11.0, 9.0]
    inv1 = inversion.find_inversions(p1_h, p1_t)
    examples.append({
        "label": "synthetic",
        "profile_type": "surface-based",
        "description": "Surface-based radiation inversion layer from ground level up to 200 m",
        "height_m": p1_h,
        "temp_c": p1_t,
        "layers": [
            {
                "base_height": layer.base_height,
                "top_height": layer.top_height,
                "depth_m": layer.depth_m,
                "strength_c": layer.strength_c,
                "type": layer.kind,
            }
            for layer in inv1
        ],
    })

    # 2. Elevated inversion
    p2_h = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    p2_t = [15.0, 14.0, 14.5, 17.0, 15.0, 13.0]
    inv2 = inversion.find_inversions(p2_h, p2_t)
    examples.append({
        "label": "synthetic",
        "profile_type": "elevated",
        "description": "Elevated subsidence inversion capping boundary layer between 100 m and 300 m",
        "height_m": p2_h,
        "temp_c": p2_t,
        "layers": [
            {
                "base_height": layer.base_height,
                "top_height": layer.top_height,
                "depth_m": layer.depth_m,
                "strength_c": layer.strength_c,
                "type": layer.kind,
            }
            for layer in inv2
        ],
    })

    # 3. No inversion (standard lapse-rate)
    p3_h = [0.0, 100.0, 200.0, 300.0, 400.0]
    p3_t = [20.0, 18.5, 17.0, 15.5, 14.0]
    inv3 = inversion.find_inversions(p3_h, p3_t)
    examples.append({
        "label": "synthetic",
        "profile_type": "none",
        "description": "Standard lapse-rate profile with continuous cooling with altitude (no inversion)",
        "height_m": p3_h,
        "temp_c": p3_t,
        "layers": [],
    })

    return examples


def generate_selftest_vectors(breakpoints_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate at least 30 selftest vectors by executing aqi.py across boundary conditions."""
    vectors: List[Dict[str, Any]] = []
    seen_inputs = set()

    def add_case(inp: Dict[str, float], desc: str):
        key = tuple(sorted((k, round(v, 4)) for k, v in inp.items()))
        if key in seen_inputs:
            return
        seen_inputs.add(key)
        val = aqi.aqi(inp, breakpoints_df=breakpoints_df)
        cat = aqi.category(val, breakpoints_df=breakpoints_df)
        vectors.append({
            "input": inp,
            "expected_aqi": val,
            "expected_category": cat,
            "description": desc,
        })

    # 1. PM2.5 boundary and midpoint values for each official category
    pm25_bp = breakpoints_df[
        breakpoints_df["pollutant"].str.upper().str.replace(".", "", regex=False) == "PM25"
    ]
    for _, row in pm25_bp.iterrows():
        c_low = float(row["conc_low"])
        c_high = float(row["conc_high"])
        c_mid = round((c_low + c_high) / 2.0, 1)
        cat = str(row["category"])
        add_case({"PM2.5": c_low}, f"PM2.5 lower boundary for {cat}")
        add_case({"PM2.5": c_mid}, f"PM2.5 midpoint for {cat}")
        add_case({"PM2.5": c_high}, f"PM2.5 upper boundary for {cat}")

    # 2. PM10 boundary and midpoint values for each official category (PM10-only cases)
    pm10_bp = breakpoints_df[
        breakpoints_df["pollutant"].str.upper().str.replace(".", "", regex=False) == "PM10"
    ]
    for _, row in pm10_bp.iterrows():
        c_low = float(row["conc_low"])
        c_high = float(row["conc_high"])
        c_mid = round((c_low + c_high) / 2.0, 1)
        cat = str(row["category"])
        add_case({"PM10": c_low}, f"PM10-only lower boundary for {cat}")
        add_case({"PM10": c_mid}, f"PM10-only midpoint for {cat}")
        add_case({"PM10": c_high}, f"PM10-only upper boundary for {cat}")

    # 3. Joint cases where PM10 dominates over PM2.5
    add_case({"PM2.5": 20.0, "PM10": 150.0}, "Joint: PM10 Moderate dominates PM2.5 Good")
    add_case({"PM2.5": 50.0, "PM10": 300.0}, "Joint: PM10 Poor dominates PM2.5 Satisfactory")
    add_case({"PM2.5": 80.0, "PM10": 450.0}, "Joint: PM10 Severe dominates PM2.5 Moderate")

    # 4. Joint cases where PM2.5 dominates over PM10
    add_case({"PM2.5": 110.0, "PM10": 80.0}, "Joint: PM2.5 Poor dominates PM10 Satisfactory")
    add_case({"PM2.5": 180.0, "PM10": 150.0}, "Joint: PM2.5 Very Poor dominates PM10 Moderate")
    add_case({"PM2.5": 320.0, "PM10": 200.0}, "Joint: PM2.5 Severe dominates PM10 Moderate")

    # 5. Boundary and edge conditions
    add_case({"PM2.5": 0.0, "PM10": 0.0}, "Joint: Zero concentrations (Good)")
    add_case({"PM2.5": 600.0}, "Beyond max concentration extrapolation (Severe)")
    add_case({"PM10": 600.0}, "PM10 beyond max concentration extrapolation (Severe)")

    return vectors


def build_site_data() -> Dict[str, Any]:
    """Compile the entire structured dataset for site/data.json."""
    # 1. Breakpoints verification and loading via aqi.py
    breakpoints_df = aqi.load_breakpoints()
    breakpoints_verified = not UNVERIFIED_MARKER.exists()

    # 2. Dynamic read of minimum valid hours rule
    min_valid_hours = int(inspect.signature(aqi.daily_average).parameters["min_valid_hours"].default)

    # 3. Load processed 2025-26 data
    df_2025 = load_2025_2026_openaq_data(PROCESSED_HOURLY_CSV)
    df_ts = df_2025.set_index("Timestamp").sort_index()

    # Date calendar for Oct 1, 2025 to Feb 28, 2026
    start_date = date(2025, 10, 1)
    end_date = date(2026, 2, 28)
    calendar_dates = [
        start_date + pd.Timedelta(days=i).to_pytimedelta()
        for i in range((end_date - start_date).days + 1)
    ]

    stations_output: Dict[str, Any] = {}

    for slug, meta in STATION_METADATA.items():
        st_slice = df_ts[df_ts["station"] == slug]

        # Compute 4 pm - 4 pm IST daily averages via aqi.py
        if not st_slice.empty and "pm25" in st_slice.columns:
            da_pm25 = aqi.daily_average(st_slice["pm25"], min_valid_hours=min_valid_hours)
        else:
            da_pm25 = pd.Series(dtype=float)

        if not st_slice.empty and "pm10" in st_slice.columns:
            da_pm10 = aqi.daily_average(st_slice["pm10"], min_valid_hours=min_valid_hours)
        else:
            da_pm10 = pd.Series(dtype=float)

        daily_aqi_records = []
        category_counts: Dict[str, int] = {
            "Good": 0,
            "Satisfactory": 0,
            "Moderate": 0,
            "Poor": 0,
            "Very Poor": 0,
            "Severe": 0,
        }

        for cal_date in calendar_dates:
            v_pm25 = da_pm25.get(cal_date)
            v_pm10 = da_pm10.get(cal_date)

            v_pm25 = None if (v_pm25 is None or pd.isna(v_pm25)) else float(v_pm25)
            v_pm10 = None if (v_pm10 is None or pd.isna(v_pm10)) else float(v_pm10)

            readings: Dict[str, float] = {}
            if v_pm25 is not None:
                readings["PM2.5"] = v_pm25
            if v_pm10 is not None:
                readings["PM10"] = v_pm10

            if readings:
                aqi_val = aqi.aqi(readings, breakpoints_df=breakpoints_df)
                cat_val = aqi.category(aqi_val, breakpoints_df=breakpoints_df)
                if cat_val in category_counts:
                    category_counts[cat_val] += 1
            else:
                aqi_val = None
                cat_val = None

            daily_aqi_records.append({
                "date": cal_date.isoformat(),
                "pm25": v_pm25,
                "pm10": v_pm10,
                "aqi": aqi_val,
                "category": cat_val,
            })

        # Monthly coverage
        monthly_coverage = []
        for yr, mo in WINTER_2025_MONTHS:
            days_in_month = monthrange(yr, mo)[1]
            hp = days_in_month * 24

            m_str = f"{yr:04d}-{mo:02d}"
            # Extract month slice
            m_slice = st_slice[(st_slice.index.year == yr) & (st_slice.index.month == mo)]
            v25_hrs = int(m_slice["pm25"].notna().sum()) if not m_slice.empty and "pm25" in m_slice.columns else 0
            v10_hrs = int(m_slice["pm10"].notna().sum()) if not m_slice.empty and "pm10" in m_slice.columns else 0

            # Daily passing count for this month
            if not m_slice.empty and "pm25" in m_slice.columns and m_slice["pm25"].notna().any():
                da_m = aqi.daily_average(m_slice["pm25"], min_valid_hours=min_valid_hours)
                passing_days = int(da_m.notna().sum())
            else:
                passing_days = 0

            monthly_coverage.append({
                "month": m_str,
                "hours_possible": hp,
                "valid_hours_pm25": v25_hrs,
                "valid_hours_pm10": v10_hrs,
                "coverage_pm25": round(v25_hrs / hp, 4),
                "coverage_pm10": round(v10_hrs / hp, 4),
                "passing_days": passing_days,
                "days_in_month": days_in_month,
            })

        stations_output[slug] = {
            "station_slug": slug,
            "display_name": meta["display_name"],
            "city": meta["city"],
            "location_id": meta["location_id"],
            "coordinates": meta["coordinates"],
            "daily_aqi": daily_aqi_records,
            "category_counts": category_counts,
            "monthly_coverage": monthly_coverage,
        }

    # 4. Check figures existence
    fire_available = FIRE_COUNTS_PNG.exists() and OVERPASS_HOURS_PNG.exists()

    # 5. Compile full data object
    site_data = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": ["OpenAQ", "CPCB"],
            "station_display_names": {slug: meta["display_name"] for slug, meta in STATION_METADATA.items()},
            "breakpoints_verified": breakpoints_verified,
            "fire_available": fire_available,
            "min_valid_hours": min_valid_hours,
        },
        "label": "PM-based AQI (PM2.5 and PM10 sub-indices)",
        "stations": stations_output,
        "breakpoints": breakpoints_df.to_dict(orient="records"),
        "selftest_vectors": generate_selftest_vectors(breakpoints_df),
        "inversion_examples": build_inversion_examples(),
    }

    return sanitize_for_json(site_data)


def write_site_data(output_path: Path = OUTPUT_FILE) -> Path:
    """Generate and write site/data.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_site_data()

    # Ensure valid JSON without raw NaNs
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, allow_nan=False)

    return output_path


def main():
    path = write_site_data()
    file_size_kb = path.stat().st_size / 1024
    print(f"Successfully generated {path} ({file_size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
