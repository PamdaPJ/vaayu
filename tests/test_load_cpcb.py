from pathlib import Path
import pandas as pd
import pytest
from load_cpcb import (
    extract_station_name,
    load_single_csv,
    load_cpcb_data,
    check_no_identical_pm25_series,
    check_station_continuity,
)


def _create_synthetic_cpcb_csv(
    filepath: Path,
    timestamps: list,
    pm25_values: list,
    pm10_values: list = None,
) -> None:
    """Helper to generate minimal valid CPCB-formatted CSV files."""
    if pm10_values is None:
        pm10_values = [50.0] * len(timestamps)

    header = (
        "Timestamp,PM2.5 (µg/m³),PM10 (µg/m³),NO2 (µg/m³),SO2 (µg/m³),"
        "CO (mg/m³),Ozone (µg/m³),NH3 (µg/m³),AT (°C),RH (%),WS (m/s),WD (deg),SR (W/mt2)"
    )
    rows = [header]
    for t, p25, p10 in zip(timestamps, pm25_values, pm10_values):
        p25_str = "NA" if pd.isna(p25) else str(p25)
        p10_str = "NA" if pd.isna(p10) else str(p10)
        rows.append(f"{t},{p25_str},{p10_str},25.0,12.0,1.2,35.0,20.0,18.0,75.0,1.5,210.0,150.0")

    filepath.write_text("\n".join(rows), encoding="utf-8")


def test_extract_station_name():
    p1 = Path("data/raw/cpcb/Raw_Data_2021_site_263_sector_16a_faridabad_hspcb_1Hr.csv")
    assert extract_station_name(p1) == "sector_16a_faridabad"

    p2 = Path("data/raw/cpcb/Raw_Data_2025_site_301_anand_vihar_delhi_dpcc_1Hr.csv")
    assert extract_station_name(p2) == "anand_vihar_delhi"

    p3 = Path("data/raw/cpcb/Raw_Data_2023_site_5082_indirapuram_ghaziabad_uppcb_1Hr.csv")
    assert extract_station_name(p3) == "indirapuram_ghaziabad"


def test_load_single_csv():
    p = Path("data/raw/cpcb/Raw_Data_2021_site_263_sector_16a_faridabad_hspcb_1Hr.csv")
    if not p.exists():
        p = Path("data/raw/cpcb_unverified/Raw_Data_2021_site_263_sector_16a_faridabad_hspcb_1Hr.csv")
    df = load_single_csv(p)

    assert "station" in df.columns
    assert df["station"].iloc[0] == "sector_16a_faridabad"
    assert "Timestamp" in df.columns
    assert str(df["Timestamp"].dt.tz) == "Asia/Kolkata"

    expected_cols = [
        "Timestamp", "station", "pm25", "pm10", "no2", "so2", "co", "o3",
        "nh3", "at", "rh", "ws", "wd", "sr"
    ]
    for col in expected_cols:
        assert col in df.columns

    assert pd.isna(df.loc[0, "pm25"])


def test_identical_pm25_series_synthetic(tmp_path: Path):
    """Test that two identical synthetic files fail loudly, while two different ones pass."""
    timestamps = [
        "2021-01-01 00:00:00",
        "2021-01-01 01:00:00",
        "2021-01-01 02:00:00",
        "2021-01-01 03:00:00",
    ]
    series_base = [45.0, 52.5, float("nan"), 60.0]
    series_diff = [45.0, 58.0, float("nan"), 70.0]

    # --- 1. Two IDENTICAL synthetic files -> must fail loudly with ValueError ---
    dir_identical = tmp_path / "identical_test"
    dir_identical.mkdir()

    f1_ident = dir_identical / "Raw_Data_2021_site_101_station_a_hspcb_1Hr.csv"
    f2_ident = dir_identical / "Raw_Data_2021_site_102_station_b_hspcb_1Hr.csv"
    _create_synthetic_cpcb_csv(f1_ident, timestamps, series_base)
    _create_synthetic_cpcb_csv(f2_ident, timestamps, series_base)

    with pytest.raises(ValueError, match="Duplicate PM2.5 series detected"):
        load_cpcb_data(dir_identical)

    # --- 2. Two DIFFERENT synthetic files -> must load successfully ---
    dir_different = tmp_path / "different_test"
    dir_different.mkdir()

    f1_diff = dir_different / "Raw_Data_2021_site_101_station_a_hspcb_1Hr.csv"
    f2_diff = dir_different / "Raw_Data_2021_site_102_station_b_hspcb_1Hr.csv"
    _create_synthetic_cpcb_csv(f1_diff, timestamps, series_base)
    _create_synthetic_cpcb_csv(f2_diff, timestamps, series_diff)

    df_diff = load_cpcb_data(dir_different)
    assert len(df_diff) == 8
    assert set(df_diff["station"].unique()) == {"station_a", "station_b"}


def test_raw_cpcb_fails_loudly_on_duplicates():
    """Verify that calling load_cpcb_data on data/raw/cpcb catches the duplicate files."""
    raw_dir = Path("data/raw/cpcb")
    if not any(raw_dir.glob("*.csv")):
        raw_dir = Path("data/raw/cpcb_unverified")
    with pytest.raises(ValueError, match="Duplicate PM2.5 series detected"):
        load_cpcb_data(raw_dir)


def test_pm10_capped_synthetic(tmp_path: Path):
    """Test that PM10 == 1000 is flagged in pm10_capped without altering numeric values."""
    dir_capping = tmp_path / "capping_test"
    dir_capping.mkdir()

    timestamps = ["2021-01-01 00:00:00", "2021-01-01 01:00:00"]
    p25 = [30.0, 40.0]
    p10 = [250.0, 1000.0]

    f = dir_capping / "Raw_Data_2021_site_101_station_a_hspcb_1Hr.csv"
    _create_synthetic_cpcb_csv(f, timestamps, p25, p10)

    df = load_cpcb_data(dir_capping)
    assert "pm10_capped" in df.columns
    assert df["pm10_capped"].tolist() == [False, True]
    assert df["pm10"].iloc[1] == 1000.0
