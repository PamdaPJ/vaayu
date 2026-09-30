"""Unit tests for baseline probabilistic forecasting model in baseline.py.

All tests use synthetic data only. Zero real data files are accessed.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

from baseline import (
    compute_brier_score,
    compute_brier_skill_score,
    bootstrap_brier_ci,
    build_daily_features_and_labels,
    split_train_test,
    compute_persistence_forecast,
    check_no_identical_stations,
    run_pipeline,
)


def test_labels_not_created_across_gaps():
    """Verify that labels are not created across a missing day or year gap."""
    # Dates: 2021-12-30, 2021-12-31, then a gap to 2023-01-01, 2023-01-02
    dates = [
        date(2021, 12, 30),
        date(2021, 12, 31),
        date(2023, 1, 1),
        date(2023, 1, 2),
    ]
    aqi_values = [420.0, 410.0, 350.0, 450.0]  # All Severe except 2023-01-01

    df_daily = pd.DataFrame({
        "date": dates,
        "station": ["station_1"] * 4,
        "aqi": aqi_values,
    })

    thresholds = {"Severe": 401.0, "Very Poor or worse": 301.0}
    df_features = build_daily_features_and_labels(
        df_daily,
        horizons=[1, 2],
        thresholds=thresholds,
    )

    # For 2021-12-30:
    # h=1 target is 2021-12-31 (present) -> should be valid (1.0)
    # h=2 target is 2022-01-01 (missing!) -> MUST BE NaN (not taken from 2023-01-01)
    row_dec30 = df_features[df_features["date"] == date(2021, 12, 30)].iloc[0]
    assert row_dec30["y_Severe_h1"] == 1.0
    assert pd.isna(row_dec30["y_Severe_h2"]), "Label leaked across missing day/year gap!"

    # For 2021-12-31:
    # h=1 target is 2022-01-01 (missing!) -> MUST BE NaN (never shifted across year gap to 2023-01-01)
    row_dec31 = df_features[df_features["date"] == date(2021, 12, 31)].iloc[0]
    assert pd.isna(row_dec31["y_Severe_h1"]), "Label leaked across multi-year gap!"

    # For 2023-01-01:
    # aqi_t_minus_1 target is 2022-12-31 (missing!) -> MUST BE NaN
    row_jan1 = df_features[df_features["date"] == date(2023, 1, 1)].iloc[0]
    assert pd.isna(row_jan1["aqi_t_minus_1"]), "Feature leaked across multi-year gap!"


def test_no_test_year_rows_in_training():
    """Verify train/test split strictly enforces no test-year rows in training."""
    dates = [
        date(2021, 10, 15),
        date(2021, 11, 20),
        date(2023, 10, 10),
        date(2025, 1, 5),
        date(2025, 2, 10),
    ]
    df = pd.DataFrame({
        "date": dates,
        "year": [d.year for d in dates],
        "station": ["st_1"] * 5,
        "aqi_t": [200, 350, 420, 410, 390],
        "y_Severe_h1": [0, 1, 1, 1, 0],
    })

    train_df, test_df = split_train_test(
        df,
        train_years=[2021, 2023],
        test_year=2025,
    )

    assert len(train_df) == 3
    assert len(test_df) == 2
    assert not any(train_df["year"] == 2025), "Data leakage: test-year rows found in training set!"
    assert all(train_df["year"].isin([2021, 2023]))
    assert all(test_df["year"] == 2025)


def test_brier_score_hand_computed():
    """Verify Brier score matches hand calculation on a tiny synthetic sample."""
    # True binary outcomes: y = [1, 0, 1, 0]
    # Forecast probabilities: p = [0.8, 0.2, 0.6, 0.1]
    # Squared errors:
    # (0.8 - 1)^2 = 0.04
    # (0.2 - 0)^2 = 0.04
    # (0.6 - 1)^2 = 0.16
    # (0.1 - 0)^2 = 0.01
    # Sum = 0.25, Mean = 0.25 / 4 = 0.0625
    y_true = [1, 0, 1, 0]
    y_prob = [0.8, 0.2, 0.6, 0.1]

    bs = compute_brier_score(y_true, y_prob)
    assert bs == pytest.approx(0.0625, rel=1e-5)


def test_brier_skill_self_comparison():
    """Verify Brier Skill Score is exactly 0.0 when comparing a forecast with itself."""
    bs_val = 0.142857
    bss_self = compute_brier_skill_score(bs_model=bs_val, bs_ref=bs_val)
    assert bss_self == pytest.approx(0.0, abs=1e-7)


def test_bootstrap_reproducible_with_fixed_seed():
    """Verify 7-day block bootstrap confidence interval is 100% reproducible with fixed seed."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1] * 3)
    y_prob = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.2, 0.6, 0.8, 0.2, 0.7, 0.3, 0.1, 0.75] * 3)

    ci1 = bootstrap_brier_ci(y_true, y_prob, block_size=7, n_boot=200, seed=42)
    ci2 = bootstrap_brier_ci(y_true, y_prob, block_size=7, n_boot=200, seed=42)

    assert ci1 == ci2
    assert ci1[0] <= ci1[1]


def test_duplicate_station_guard_triggers_on_identical_stations():
    """Verify that pooling fails loudly with ValueError if two stations have identical PM2.5 series."""
    ts = pd.date_range("2021-01-01", periods=10, freq="h")
    pm25_identical = [40.0, 50.0, 60.0, np.nan, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]

    df_dup = pd.DataFrame({
        "Timestamp": list(ts) * 2,
        "station": ["station_A"] * 10 + ["station_B"] * 10,
        "pm25": pm25_identical * 2,
    })

    with pytest.raises(ValueError, match="Duplicate station data detected"):
        check_no_identical_stations(df_dup)


def test_persistence_forecast_on_short_handmade_series():
    """Verify persistence forecast: event on day t predicts event on day t+h."""
    # Day t event: 1 if day t is Severe, 0 otherwise
    # Horizon h=1: forecast for day t+1 is day t's event status
    # Horizon h=2: forecast for day t+2 is day t's event status
    events = [0, 1, 1, 0, 1]
    p_h1 = compute_persistence_forecast(events)
    assert p_h1 == [0, 1, 1, 0, 1]


def test_run_pipeline_raises_on_identical_stations_temporary(tmp_path):
    """Verify run_pipeline raises on a temporary dataset with identical stations without touching real data."""
    temp_hourly_csv = tmp_path / "cpcb_hourly.csv"
    temp_breakpoints_csv = tmp_path / "cpcb_breakpoints.csv"

    # Minimal valid breakpoints
    temp_breakpoints_csv.write_text(
        "pollutant,averaging_hours,conc_low,conc_high,index_low,index_high,category\n"
        "PM2.5,24,0,30,0,50,Good\n"
        "PM2.5,24,31,60,51,100,Satisfactory\n"
        "PM2.5,24,61,90,101,200,Moderate\n"
        "PM2.5,24,91,120,201,300,Poor\n"
        "PM2.5,24,121,250,301,400,Very Poor\n"
        "PM2.5,24,251,500,401,500,Severe\n",
        encoding="utf-8",
    )

    # 2 stations with identical PM2.5 series
    ts = pd.date_range("2021-10-01 00:00:00", periods=24, freq="h")
    pm25_vals = [50.0 + i for i in range(24)]

    df_temp = pd.DataFrame({
        "Timestamp": list(ts) * 2,
        "station": ["st_alpha"] * 24 + ["st_beta"] * 24,
        "pm25": pm25_vals * 2,
    })
    df_temp.to_csv(temp_hourly_csv, index=False)

    with pytest.raises(ValueError, match="Duplicate station data detected"):
        run_pipeline(
            hourly_csv=temp_hourly_csv,
            breakpoints_csv=temp_breakpoints_csv,
            results_path=tmp_path / "results.json",
            figures_dir=tmp_path / "figures",
        )


def test_run_pipeline_aborts_on_unverified_marker(tmp_path):
    """Verify run_pipeline aborts if an UNVERIFIED marker file exists next to breakpoints."""
    temp_hourly_csv = tmp_path / "cpcb_hourly.csv"
    temp_breakpoints_csv = tmp_path / "cpcb_breakpoints.csv"
    marker = tmp_path / "UNVERIFIED"
    marker.write_text("Unverified marker", encoding="utf-8")

    temp_breakpoints_csv.write_text(
        "pollutant,averaging_hours,conc_low,conc_high,index_low,index_high,category\n"
        "PM2.5,24,0,30,0,50,Good\n"
        "PM2.5,24,251,500,401,500,Severe\n",
        encoding="utf-8",
    )

    ts = pd.date_range("2021-10-01 00:00:00", periods=24, freq="h")
    df_temp = pd.DataFrame({
        "Timestamp": ts,
        "station": ["st_single"] * 24,
        "pm25": [50.0 + i for i in range(24)],
    })
    df_temp.to_csv(temp_hourly_csv, index=False)

    with pytest.raises(RuntimeError, match="UNVERIFIED"):
        run_pipeline(
            hourly_csv=temp_hourly_csv,
            breakpoints_csv=temp_breakpoints_csv,
            results_path=tmp_path / "results.json",
            figures_dir=tmp_path / "figures",
        )
