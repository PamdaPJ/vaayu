"""Baseline Probabilistic Forecasting Model for VAAYU.

Predicts next-day and multi-horizon (h in {1, 2, 3}) probabilities of Severe
and Very Poor or worse AQI events for Delhi NCR monitoring stations using:
- Logistic regression on lagged daily AQI and seasonal harmonic features
- Time-based train/test splits (train on 2021 & 2023, test on 2025; Oct-Feb window)
- Reference forecasts: Climatology and Persistence
- Evaluation via Brier score, Brier skill score, 7-day block bootstrap 95% CI,
  and sample-annotated reliability diagrams.
- Duplicate station guardrail before pooling.
"""

from datetime import date, datetime, timedelta
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

import aqi


# --- Duplicate Station Guard ---

def check_no_identical_stations(hourly_df: pd.DataFrame) -> None:
    """Fail loudly if any two stations share an identical PM2.5 series in the same year."""
    if "station" not in hourly_df.columns or "pm25" not in hourly_df.columns:
        return

    ts = pd.to_datetime(hourly_df["Timestamp"])
    years = ts.dt.year.unique()

    for year in years:
        yr_mask = ts.dt.year == year
        yr_df = hourly_df[yr_mask]
        stations = sorted(yr_df["station"].unique())

        for st1, st2 in itertools.combinations(stations, 2):
            s1 = yr_df[yr_df["station"] == st1].sort_values("Timestamp")
            s2 = yr_df[yr_df["station"] == st2].sort_values("Timestamp")

            if len(s1) > 0 and len(s1) == len(s2):
                v1 = s1["pm25"].to_numpy()
                v2 = s2["pm25"].to_numpy()
                if np.array_equal(v1, v2, equal_nan=True):
                    raise ValueError(
                        f"Duplicate station data detected: stations '{st1}' and '{st2}' "
                        f"have identical PM2.5 series for year {year}."
                    )


# --- Daily Table and AQI Aggregation ---

def build_daily_table(
    hourly_df: pd.DataFrame,
    min_valid_hours: int = 16,
) -> pd.DataFrame:
    """Construct daily average pollutant and AQI table per station (4 pm - 4 pm IST window).

    Days failing min_valid_hours are missing (NaN), not filled.
    """
    df = hourly_df.copy()
    if not isinstance(df["Timestamp"].dtype, pd.DatetimeTZDtype):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        if df["Timestamp"].dt.tz is None:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("Asia/Kolkata")

    breakpoints = aqi.load_breakpoints()
    pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]

    daily_records = []
    for station, grp in df.groupby("station"):
        grp = grp.sort_values("Timestamp").set_index("Timestamp")

        station_daily_pollutants = {}
        for p in pollutant_cols:
            if p in grp.columns:
                series = grp[p]
                station_daily_pollutants[p] = aqi.daily_average(
                    series, min_valid_hours=min_valid_hours
                )

        if not station_daily_pollutants:
            continue

        df_p_daily = pd.DataFrame(station_daily_pollutants)

        for rep_date, row in df_p_daily.iterrows():
            readings = {}
            # Map canonical column names to AQI pollutant names
            name_map = {
                "pm25": "PM2.5",
                "pm10": "PM10",
                "no2": "NO2",
                "so2": "SO2",
                "co": "CO",
                "o3": "O3",
                "nh3": "NH3",
            }
            for p, val in row.items():
                if not pd.isna(val):
                    readings[name_map.get(p, p)] = float(val)

            day_aqi = aqi.aqi(readings, breakpoints_df=breakpoints)
            daily_records.append({
                "date": rep_date,
                "station": station,
                "aqi": day_aqi,
                **{f"conc_{k}": v for k, v in row.items()},
            })

    daily_df = pd.DataFrame(daily_records)
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
        daily_df = daily_df.sort_values(["station", "date"]).reset_index(drop=True)
    return daily_df


# --- Features and Strict Calendar Shift Labels ---

def build_daily_features_and_labels(
    daily_df: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3),
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute lag features and horizon event labels respecting strict calendar continuity.

    Never shifts across missing day or multi-year gaps.
    """
    if thresholds is None:
        thresholds = aqi.get_category_thresholds()

    rows = []
    for station, grp in daily_df.groupby("station"):
        # Create map from calendar date to daily AQI
        date_map = {row["date"]: row["aqi"] for _, row in grp.iterrows()}

        for _, row in grp.iterrows():
            d = row["date"]
            aqi_t = row["aqi"]

            # Feature: aqi on day t-1 (strictly d - 1 day)
            prev_d = d - timedelta(days=1)
            aqi_t_minus_1 = date_map.get(prev_d, np.nan)

            # Seasonal harmonics
            m = d.month
            m_sin = math.sin(2.0 * math.pi * m / 12.0)
            m_cos = math.cos(2.0 * math.pi * m / 12.0)

            rec = {
                "date": d,
                "year": d.year,
                "month": d.month,
                "station": station,
                "aqi_t": aqi_t,
                "aqi_t_minus_1": aqi_t_minus_1,
                "month_sin": m_sin,
                "month_cos": m_cos,
            }

            # Labels for each horizon and threshold
            for h in horizons:
                target_d = d + timedelta(days=h)
                target_aqi = date_map.get(target_d, np.nan)

                for thresh_name, thresh_val in thresholds.items():
                    col_name = f"y_{thresh_name.replace(' ', '_')}_h{h}"
                    if pd.isna(target_aqi):
                        rec[col_name] = np.nan
                    else:
                        # Event is positive if AQI >= threshold value (e.g. >= 401 or >= 301)
                        rec[col_name] = 1.0 if target_aqi >= thresh_val else 0.0

            rows.append(rec)

    df_out = pd.DataFrame(rows)
    return df_out


def split_train_test(
    features_df: pd.DataFrame,
    train_years: Sequence[int] = (2021, 2023),
    test_year: int = 2025,
    target_months: Sequence[int] = (10, 11, 12, 1, 2),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically by winter years restricted to Oct-Feb days.

    Asserts that no test-year rows are present in the training set.
    """
    df = features_df.copy()
    if "date" in df.columns:
        dt_series = pd.to_datetime(df["date"])
        if "month" not in df.columns:
            df["month"] = dt_series.dt.month
        if "year" not in df.columns:
            df["year"] = dt_series.dt.year

    # Restrict to Oct-Feb days
    df_winter = df[df["month"].isin(target_months)].copy()

    train_df = df_winter[df_winter["year"].isin(train_years)].copy()
    test_df = df_winter[df_winter["year"] == test_year].copy()

    # Strict assertion: no test year data leakage into training
    assert not any(train_df["year"] == test_year), (
        f"Data leakage detected! Training set contains observations from test year {test_year}."
    )
    return train_df, test_df


# --- Reference Forecasts ---

def compute_climatology_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> np.ndarray:
    """Predict event probability based on training set monthly event rate."""
    valid_train = train_df.dropna(subset=[label_col])
    monthly_rates = valid_train.groupby("month")[label_col].mean().to_dict()
    global_rate = float(valid_train[label_col].mean()) if len(valid_train) > 0 else 0.0

    test_probs = [monthly_rates.get(m, global_rate) for m in test_df["month"]]
    return np.array(test_probs, dtype=float)


def compute_persistence_forecast(
    events_t: Sequence[float],
) -> List[float]:
    """Predict event on day t+h using event state on day t."""
    return [float(e) if not pd.isna(e) else np.nan for e in events_t]


# --- Verification Metrics ---

def compute_brier_score(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """Compute Brier Score: mean squared error between binary outcomes and forecast probabilities."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_p - y_t) ** 2))


def compute_brier_skill_score(bs_model: float, bs_ref: float) -> float:
    """Compute Brier Skill Score relative to reference forecast."""
    if bs_ref == 0.0:
        return 0.0 if bs_model == 0.0 else -np.inf
    return float(1.0 - (bs_model / bs_ref))


def bootstrap_brier_ci(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    block_size: int = 7,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval using 7-day block resampling."""
    rng = np.random.RandomState(seed)
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    n = len(y_t)

    if n < block_size:
        base_score = float(np.mean((y_p - y_t) ** 2))
        return base_score, base_score

    # Chunk chronologically into 7-day blocks
    blocks = [list(range(i, min(i + block_size, n))) for i in range(0, n, block_size)]
    n_blocks = len(blocks)

    boot_scores = []
    for _ in range(n_boot):
        chosen_indices = []
        for _ in range(n_blocks):
            b_idx = rng.randint(0, n_blocks)
            chosen_indices.extend(blocks[b_idx])
        chosen_indices = chosen_indices[:n]

        bs = float(np.mean((y_p[chosen_indices] - y_t[chosen_indices]) ** 2))
        boot_scores.append(bs)

    ci_low = float(np.percentile(boot_scores, 2.5))
    ci_high = float(np.percentile(boot_scores, 97.5))
    return ci_low, ci_high


# --- Reliability Diagram ---

def plot_reliability_diagram(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    horizon: int,
    threshold_name: str,
    output_dir: Path = Path("figures"),
) -> Path:
    """Generate and save reliability diagram with sample counts per probability bin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = threshold_name.replace(" ", "_")
    output_path = output_dir / f"reliability_h{horizon}_{slug}.png"

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    n_events = int(np.sum(y_t == 1.0))

    # Use fewer bins when events are few
    n_bins = 5 if n_events < 10 else 10

    prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=n_bins, strategy="uniform")

    # Count sample distribution across uniform bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.histogram(y_p, bins=bin_edges)[0]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    ax.plot(prob_pred, prob_true, "s-", color="#1b9e77", label=f"Model (h={horizon})")

    # Annotate sample counts
    for x, y, count in zip(prob_pred, prob_true, bin_counts[bin_counts > 0]):
        ax.annotate(
            f"n={count}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    ax.set_title(
        f"Reliability Diagram: Horizon h={horizon}\nThreshold: {threshold_name}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Observed Event Fraction", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Main Experiment Runner ---

def run_pipeline(
    hourly_csv: Path = Path("data/processed/cpcb_hourly.csv"),
    breakpoints_csv: Path = Path("data/cpcb_breakpoints.csv"),
    results_path: Path = Path("results.json"),
    figures_dir: Path = Path("figures"),
) -> Dict[str, Any]:
    """Execute complete baseline probabilistic forecasting pipeline.

    Execution order:
    1. Load data/processed/cpcb_hourly.csv.
    2. Run duplicate-station guard FIRST and abort with a clear error message
       before computing anything if any two stations have identical pm25 arrays
       for the same year. Also abort if data/cpcb_breakpoints.csv has an UNVERIFIED
       marker file next to it.
    3. Print thresholds used for 'Severe' and 'Very Poor or worse'.
    4. Only then build daily tables, models, metrics, figures, and results.json.
    """
    hourly_csv = Path(hourly_csv)
    breakpoints_csv = Path(breakpoints_csv)
    results_path = Path(results_path)
    figures_dir = Path(figures_dir)

    # 1. Load hourly data
    if not hourly_csv.exists():
        raise FileNotFoundError(f"Hourly dataset not found: {hourly_csv.resolve()}")
    hourly_df = pd.read_csv(hourly_csv)

    # 2. Run duplicate-station guard FIRST before computing anything
    check_no_identical_stations(hourly_df)

    # 2b. Check if an UNVERIFIED marker file exists next to breakpoints
    bp_parent = breakpoints_csv.parent
    unverified_candidates = [
        bp_parent / "UNVERIFIED",
        bp_parent / f"{breakpoints_csv.name}.UNVERIFIED",
        bp_parent / f"{breakpoints_csv.stem}.UNVERIFIED",
    ]
    if any(p.exists() for p in unverified_candidates):
        raise RuntimeError(
            f"Aborting: Found UNVERIFIED marker file next to {breakpoints_csv}. "
            "Breakpoints table must be verified before running baseline pipeline."
        )

    # 3. Print thresholds used
    thresholds = aqi.get_category_thresholds(breakpoints_csv)
    print("=" * 76)
    print("VAAYU Baseline Probabilistic Model: CPCB Thresholds")
    print("=" * 76)
    print(f"  • Severe Threshold:             AQI >= {thresholds['Severe']}")
    print(f"  • Very Poor or worse Threshold: AQI >= {thresholds['Very Poor or worse']}")
    print("=" * 76)

    # 4. Build daily table, models, metrics, figures, and results.json
    daily_df = build_daily_table(hourly_df, min_valid_hours=16)

    horizons = [1, 2, 3]
    feat_df = build_daily_features_and_labels(daily_df, horizons=horizons, thresholds=thresholds)

    # Group experiments: individual stations and pooled
    experiments = {}
    stations = sorted(feat_df["station"].unique())
    for st in stations:
        experiments[st] = feat_df[feat_df["station"] == st].copy()
    experiments["pooled"] = feat_df.copy()

    all_results: Dict[str, Any] = {
        "train_years": [2021, 2023],
        "test_years": [2025],
        "thresholds_used": thresholds,
        "experiments": {},
    }

    feature_cols = ["aqi_t", "aqi_t_minus_1", "month_sin", "month_cos"]

    print("\nTraining models and evaluating on test year 2025 (Oct–Feb window)...")
    for exp_name, exp_df in experiments.items():
        train_df, test_df = split_train_test(
            exp_df, train_years=[2021, 2023], test_year=2025, target_months=[10, 11, 12, 1, 2]
        )
        all_results["experiments"][exp_name] = {}

        for thresh_name, thresh_val in thresholds.items():
            thresh_slug = thresh_name.replace(" ", "_")
            all_results["experiments"][exp_name][thresh_name] = {}

            for h in horizons:
                target_col = f"y_{thresh_slug}_h{h}"

                tr = train_df.dropna(subset=feature_cols + [target_col])
                te = test_df.dropna(subset=feature_cols + [target_col])

                n_test_days = len(te)
                n_events = int(te[target_col].sum()) if n_test_days > 0 else 0

                status = "valid"
                if n_events < 10:
                    status = "insufficient events"
                    print(
                        f"WARNING [{exp_name} | {thresh_name} | h={h}]: "
                        f"Test set has {n_events} events (< 10). Wide confidence intervals expected."
                    )

                if n_test_days == 0 or len(tr) == 0:
                    all_results["experiments"][exp_name][thresh_name][f"h{h}"] = {
                        "status": "no data",
                        "test_days": n_test_days,
                        "events": n_events,
                    }
                    continue

                X_train = tr[feature_cols].to_numpy()
                y_train = tr[target_col].to_numpy()
                X_test = te[feature_cols].to_numpy()
                y_test = te[target_col].to_numpy()

                clf = LogisticRegression(solver="lbfgs", random_state=42)
                if len(np.unique(y_train)) > 1:
                    clf.fit(X_train, y_train)
                    p_model = clf.predict_proba(X_test)[:, 1]
                else:
                    p_model = np.full(len(X_test), fill_value=y_train[0])

                p_clim = compute_climatology_forecast(tr, te, target_col)
                day_t_event = (te["aqi_t"] >= thresh_val).astype(float).to_numpy()
                p_pers = compute_persistence_forecast(day_t_event)

                bs_model = compute_brier_score(y_test, p_model)
                bs_clim = compute_brier_score(y_test, p_clim)
                bs_pers = compute_brier_score(y_test, p_pers)

                bss_clim = compute_brier_skill_score(bs_model, bs_clim)
                bss_pers = compute_brier_skill_score(bs_model, bs_pers)
                ci_low, ci_high = bootstrap_brier_ci(y_test, p_model, block_size=7, seed=42)

                plot_reliability_diagram(
                    y_test,
                    p_model,
                    horizon=h,
                    threshold_name=f"{exp_name} {thresh_name}",
                    output_dir=figures_dir,
                )

                all_results["experiments"][exp_name][thresh_name][f"h{h}"] = {
                    "status": status,
                    "test_days": n_test_days,
                    "events": n_events,
                    "brier_score": round(bs_model, 5),
                    "brier_ci_95": [round(ci_low, 5), round(ci_high, 5)],
                    "brier_clim": round(bs_clim, 5),
                    "brier_pers": round(bs_pers, 5),
                    "bss_climatology": round(bss_clim, 4),
                    "bss_persistence": round(bss_pers, 4),
                }

    # Save results.json
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation metrics to {results_path}")

    # Print summary table
    print("\n" + "=" * 92)
    print(f"{'Experiment':<22} | {'Threshold':<18} | {'H':<3} | {'Events/Days':<12} | {'Brier (95% CI)':<20} | {'BSS (Clim)':<10}")
    print("=" * 92)
    for exp_k, exp_v in all_results["experiments"].items():
        for th_k, th_v in exp_v.items():
            for h_k, h_res in th_v.items():
                if "brier_score" in h_res:
                    ci_str = f"[{h_res['brier_ci_95'][0]:.3f}, {h_res['brier_ci_95'][1]:.3f}]"
                    ev_str = f"{h_res['events']}/{h_res['test_days']}"
                    print(
                        f"{exp_k:<22} | {th_k:<18} | {h_k:<3} | {ev_str:<12} | "
                        f"{h_res['brier_score']:.4f} {ci_str:<13} | {h_res['bss_climatology']:<10.4f}"
                    )
    print("=" * 92)

    return all_results


# Backward-compatible alias
run_baseline_pipeline = run_pipeline


if __name__ == "__main__":
    run_pipeline()
