"""CPCB Air Quality Index (AQI) Engine for VAAYU.

Implements the official Central Pollution Control Board (CPCB) National Air Quality
Index (NAQI) sub-index interpolation, AQI aggregation (max sub-index), categorization,
4 pm-4 pm IST 24-hour daily averaging, and probabilistic P(Severe AQI) estimation.
All breakpoints are loaded dynamically from data/cpcb_breakpoints.csv.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd

DEFAULT_BREAKPOINTS_PATH = Path("data/cpcb_breakpoints.csv")


def load_breakpoints(filepath: Union[str, Path] = DEFAULT_BREAKPOINTS_PATH) -> pd.DataFrame:
    """Load official CPCB pollutant breakpoints from CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Breakpoints file not found: {path.resolve()}")
    df = pd.read_csv(path)
    required_cols = {"pollutant", "conc_low", "conc_high", "index_low", "index_high", "category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Breakpoints CSV missing required columns: {required_cols - set(df.columns)}")
    return df


def get_category_thresholds(filepath: Union[str, Path] = DEFAULT_BREAKPOINTS_PATH) -> Dict[str, float]:
    """Retrieve AQI category boundary thresholds dynamically from the breakpoints CSV."""
    df = load_breakpoints(filepath)
    # Severe index low: typically 401 (AQI > 400 or >= 401)
    severe_low = df[df["category"] == "Severe"]["index_low"].min()
    # Very Poor index low: typically 301 (AQI > 300 or >= 301)
    very_poor_low = df[df["category"] == "Very Poor"]["index_low"].min()

    return {
        "Severe": float(severe_low),
        "Very Poor or worse": float(very_poor_low),
    }


def sub_index(
    pollutant: str,
    conc: float,
    breakpoints_df: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    """Compute CPCB sub-index for a given pollutant concentration via linear interpolation.

    Formula:
        I = I_low + ((I_high - I_low) / (C_high - C_low)) * (C - C_low)
    """
    if pd.isna(conc) or conc < 0:
        return None

    if breakpoints_df is None:
        breakpoints_df = load_breakpoints()

    # Normalize pollutant name matching (case-insensitive, strip whitespace/symbols)
    p_norm = pollutant.strip().upper().replace(".", "").replace(" ", "")
    df_p = breakpoints_df[
        breakpoints_df["pollutant"].str.strip().str.upper().str.replace(".", "", regex=False).str.replace(" ", "", regex=False) == p_norm
    ]

    if df_p.empty:
        return None

    for _, row in df_p.iterrows():
        c_low = float(row["conc_low"])
        c_high = float(row["conc_high"])
        i_low = float(row["index_low"])
        i_high = float(row["index_high"])

        if c_low <= conc <= c_high:
            sub = i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
            return float(round(sub))

    # If concentration exceeds max breakpoint, extrapolate using highest slope
    max_row = df_p.loc[df_p["conc_high"].idxmax()]
    c_low = float(max_row["conc_low"])
    c_high = float(max_row["conc_high"])
    i_low = float(max_row["index_low"])
    i_high = float(max_row["index_high"])
    sub = i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
    return float(round(sub))


def aqi(
    readings: Dict[str, float],
    breakpoints_df: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    """Compute overall AQI from a dict of pollutant concentrations (AQI = max sub-index)."""
    if breakpoints_df is None:
        breakpoints_df = load_breakpoints()

    sub_indices = []
    for pollutant, conc in readings.items():
        if conc is not None and not pd.isna(conc):
            si = sub_index(pollutant, conc, breakpoints_df=breakpoints_df)
            if si is not None:
                sub_indices.append(si)

    if not sub_indices:
        return None

    return float(max(sub_indices))


def category(
    aqi_value: Optional[float],
    breakpoints_df: Optional[pd.DataFrame] = None,
) -> str:
    """Return CPCB AQI category string for a given numeric AQI value."""
    if aqi_value is None or pd.isna(aqi_value):
        return "Unknown"

    if breakpoints_df is None:
        breakpoints_df = load_breakpoints()

    # Find matching category by index_low <= aqi_value <= index_high
    cats = breakpoints_df[["index_low", "index_high", "category"]].drop_duplicates()
    for _, row in cats.iterrows():
        if float(row["index_low"]) <= aqi_value <= float(row["index_high"]):
            return str(row["category"])

    if aqi_value > cats["index_high"].max():
        return "Severe"
    return "Good"


def daily_average(
    hourly_series: pd.Series,
    min_valid_hours: int = 16,
) -> pd.Series:
    """Compute daily averages for 4 pm - 4 pm IST window.

    Each daily reporting window runs from 17:00 on Day T-1 to 16:00 on Day T.
    If valid observations in a window are fewer than min_valid_hours, the result is NaN.

    Parameters:
        hourly_series: pd.Series with DateTimeIndex in IST.
        min_valid_hours: Minimum number of valid hours required (default: 16 out of 24).

    Returns:
        pd.Series indexed by reporting date (datetime.date).
    """
    if hourly_series.empty:
        return pd.Series(dtype=float)

    # Shift timestamps so (Day T-1 17:00 to Day T 16:00) maps to Day T
    dt_index = hourly_series.index
    if not isinstance(dt_index, pd.DatetimeIndex):
        dt_index = pd.to_datetime(dt_index)
        hourly_series = hourly_series.copy()
        hourly_series.index = dt_index

    # Reporting date: subtract 16 hours and 1 second, then add 1 day
    shifted = dt_index - pd.Timedelta(hours=16, seconds=1)
    reporting_dates = shifted.date + pd.Timedelta(days=1)

    df_tmp = pd.DataFrame({"val": hourly_series.values, "rep_date": reporting_dates})
    
    def _agg_window(s: pd.Series) -> float:
        valid_count = s.count()
        if valid_count < min_valid_hours:
            return np.nan
        return float(s.mean())

    daily = df_tmp.groupby("rep_date")["val"].agg(_agg_window)
    return daily


def p_severe(
    pm25_samples: Sequence[float],
    pm10_samples: Sequence[float],
    breakpoints_df: Optional[pd.DataFrame] = None,
) -> float:
    """Compute fraction of joint PM2.5 and PM10 samples resulting in Severe AQI (AQI > 400)."""
    p25 = np.asarray(pm25_samples, dtype=float)
    p10 = np.asarray(pm10_samples, dtype=float)

    if len(p25) == 0 or len(p10) == 0 or len(p25) != len(p10):
        return 0.0

    if breakpoints_df is None:
        breakpoints_df = load_breakpoints()

    thresholds = get_category_thresholds()
    severe_threshold = thresholds["Severe"]

    severe_count = 0
    valid_count = 0
    for v25, v10 in zip(p25, p10):
        if pd.isna(v25) and pd.isna(v10):
            continue
        readings = {}
        if not pd.isna(v25):
            readings["PM2.5"] = v25
        if not pd.isna(v10):
            readings["PM10"] = v10

        val = aqi(readings, breakpoints_df=breakpoints_df)
        if val is not None:
            valid_count += 1
            # Severe is AQI >= severe_threshold (401) or AQI > 400
            if val >= severe_threshold or val > 400:
                severe_count += 1

    return float(severe_count / valid_count) if valid_count > 0 else 0.0
