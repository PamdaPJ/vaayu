"""CPCB Raw Data Loader and Preprocessor for VAAYU.

Loads hourly CPCB air quality station CSV files, extracts station names from
filenames, cleans missing values, parses timestamps to Indian Standard Time (IST),
filters and standardizes pollutant and meteorological columns, flags PM10
saturation capping, checks for timestamp duplicates and continuity gaps, writes
the combined dataset to data/processed/cpcb_hourly.csv, and prints data quality summaries.
"""

from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple
import pandas as pd

# Ensure console supports utf-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Target variables to retain and their regex matching patterns in raw CSV headers
POLLUTANT_METEO_PATTERNS = {
    "pm25": r"^PM2\.?5",
    "pm10": r"^PM10",
    "no2": r"^NO2\b",
    "so2": r"^SO2\b",
    "co": r"^CO\b",
    "o3": r"^Ozone\b",
    "nh3": r"^NH3\b",
    "at": r"^AT\b",
    "rh": r"^RH\b",
    "ws": r"^WS\b",
    "wd": r"^WD\b",
    "sr": r"^SR\b",
}


def extract_station_name(filepath: Path) -> str:
    """Extract station name from CPCB raw CSV filename.

    Extracts the portion after 'site_<id>_' and before the agency suffix
    (e.g., '_hspcb', '_dpcc', '_uppcb').

    Examples:
        Raw_Data_2021_site_263_sector_16a_faridabad_hspcb_1Hr.csv -> sector_16a_faridabad
        Raw_Data_2021_site_301_anand_vihar_delhi_dpcc_1Hr.csv       -> anand_vihar_delhi
        Raw_Data_2021_site_5082_indirapuram_ghaziabad_uppcb_1Hr.csv -> indirapuram_ghaziabad
    """
    fname = filepath.name

    # Check for site_<id>_<station>_(hspcb|dpcc|uppcb) or general agency pattern
    match = re.search(
        r"site_\d+_(.*?)(?:_(?:hspcb|dpcc|uppcb|[a-zA-Z]+))?_1Hr\.csv$",
        fname,
        re.IGNORECASE,
    )
    if match and match.group(1):
        return match.group(1)

    # Direct fallback for pattern explicitly before _hspcb
    match_hspcb = re.search(r"site_\d+_(.*?)_hspcb", fname, re.IGNORECASE)
    if match_hspcb and match_hspcb.group(1):
        return match_hspcb.group(1)

    # General fallback: strip prefixes and suffixes
    stem = filepath.stem
    stem = re.sub(r"^Raw_Data_\d{4}_site_\d+_", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(?:hspcb|dpcc|uppcb)?[_.]?1Hr$", "", stem, flags=re.IGNORECASE)
    return stem


def identify_columns_to_keep(df: pd.DataFrame) -> Dict[str, str]:
    """Map raw dataframe column names to canonical lowercase target names."""
    rename_map = {}
    for col in df.columns:
        for target_name, pattern in POLLUTANT_METEO_PATTERNS.items():
            if re.search(pattern, col.strip(), re.IGNORECASE):
                rename_map[col] = target_name
                break
    return rename_map


def load_single_csv(filepath: Path) -> pd.DataFrame:
    """Load a single raw CPCB CSV file, clean, parse, and filter columns."""
    na_values = ["NA", "N/A", "null", "NULL", "None", "-", "", "nan", "NaN"]
    df = pd.read_csv(filepath, na_values=na_values, encoding="utf-8")

    # Add station column parsed from filename
    station = extract_station_name(filepath)
    df["station"] = station

    # Map pollutant and meteorological columns
    col_mapping = identify_columns_to_keep(df)
    df = df.rename(columns=col_mapping)

    # Parse timestamps to IST (Asia/Kolkata / UTC+05:30)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        if df["Timestamp"].dt.tz is None:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("Asia/Kolkata")

    # Coerce target variables to numeric float
    target_cols = list(POLLUTANT_METEO_PATTERNS.keys())
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    # Select only required columns: Timestamp, station, plus target variables
    ordered_cols = ["Timestamp", "station"] + target_cols
    return df[ordered_cols]


def check_station_continuity(
    station_df: pd.DataFrame,
) -> Tuple[int, int, List[pd.Timestamp]]:
    """Check for duplicate timestamps and continuity gaps (> 1 hour)."""
    sorted_df = station_df.sort_values("Timestamp")
    duplicates = int(sorted_df.duplicated(subset=["Timestamp"]).sum())

    diffs = sorted_df["Timestamp"].diff()
    gap_rows = sorted_df[diffs > pd.Timedelta(hours=1)]
    gaps = len(gap_rows)
    gap_timestamps = gap_rows["Timestamp"].tolist()
    return duplicates, gaps, gap_timestamps


def check_no_identical_pm25_series(df: pd.DataFrame) -> None:
    """Validate that no two stations share an identical PM2.5 series for any year.

    Compares the value arrays of PM2.5 for each pair of stations within the same
    calendar year. Fails loudly by raising ValueError if any pair is identical.
    """
    if "station" not in df.columns or "pm25" not in df.columns or "Timestamp" not in df.columns:
        return

    import itertools
    import numpy as np

    for year, grp in df.groupby(df["Timestamp"].dt.year):
        stations = sorted(grp["station"].unique())
        for st1, st2 in itertools.combinations(stations, 2):
            s1 = grp[grp["station"] == st1].sort_values("Timestamp")
            s2 = grp[grp["station"] == st2].sort_values("Timestamp")

            if len(s1) > 0 and len(s1) == len(s2):
                v1 = s1["pm25"].to_numpy()
                v2 = s2["pm25"].to_numpy()
                if np.array_equal(v1, v2, equal_nan=True):
                    raise ValueError(
                        f"Duplicate PM2.5 series detected: stations '{st1}' and '{st2}' "
                        f"have identical PM2.5 series for year {year}."
                    )


def load_cpcb_data(
    raw_dir: Path = Path("data/raw/cpcb"),
    validate_identical_pm25: bool = True,
) -> pd.DataFrame:
    """Load all CPCB CSVs from raw_dir, combine them, and apply PM10 capping flag."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir.resolve()}")

    dfs = [load_single_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort deterministically by station and timestamp
    combined_df = combined_df.sort_values(
        by=["station", "Timestamp"]
    ).reset_index(drop=True)

    # Flag PM10 == 1000 in pm10_capped without altering original values
    combined_df["pm10_capped"] = combined_df["pm10"] == 1000

    # Validate that no two stations have identical PM2.5 series for the same year
    if validate_identical_pm25:
        check_no_identical_pm25_series(combined_df)

    return combined_df


def print_station_report(df: pd.DataFrame) -> None:
    """Print per station: date range, row count, % missing for pm25 & pm10, capped PM10 hours, and gaps."""
    print("=" * 86)
    print("CPCB Per-Station Processing & Quality Report")
    print("=" * 86)

    for station, grp in df.groupby("station"):
        d_min = grp["Timestamp"].min().strftime("%Y-%m-%d %H:%M %Z")
        d_max = grp["Timestamp"].max().strftime("%Y-%m-%d %H:%M %Z")
        row_count = len(grp)

        pm25_missing = grp["pm25"].isna().sum()
        pm25_pct = (pm25_missing / row_count) * 100 if row_count > 0 else 0.0

        pm10_missing = grp["pm10"].isna().sum()
        pm10_pct = (pm10_missing / row_count) * 100 if row_count > 0 else 0.0

        capped_pm10_count = int(grp["pm10_capped"].sum())
        duplicates, gaps, _ = check_station_continuity(grp)

        print(f"Station: {station}")
        print(f"  • Date Range:             {d_min} to {d_max}")
        print(f"  • Total Rows:             {row_count:,}")
        print(f"  • PM2.5 Missing:          {pm25_missing:,} ({pm25_pct:.2f}%)")
        print(f"  • PM10 Missing:           {pm10_missing:,} ({pm10_pct:.2f}%)")
        print(f"  • PM10 Capped Hours:      {capped_pm10_count} (PM10 == 1000 µg/m³)")
        print(f"  • Timestamp Integrity:    {duplicates} duplicates, {gaps} gaps (>1 hr)")
        print("-" * 86)

    total_rows = len(df)
    total_capped = int(df["pm10_capped"].sum())
    print(f"Dataset Total: {total_rows:,} rows across {df['station'].nunique()} stations | Total PM10 capped hours: {total_capped}")
    print("=" * 86)


def process_cpcb(
    raw_dir: Path = Path("data/raw/cpcb"),
    output_path: Path = Path("data/processed/cpcb_hourly.csv"),
) -> pd.DataFrame:
    """Execute complete CPCB raw-to-processed pipeline and print report."""
    print(f"Loading raw CPCB files from {raw_dir}...")
    df = load_cpcb_data(raw_dir=raw_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing processed hourly data to {output_path}...")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df):,} rows to {output_path}.\n")

    print_station_report(df)
    return df


if __name__ == "__main__":
    process_cpcb()
