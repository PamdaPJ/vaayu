"""Generate expanding-window Ridge regression probabilistic forecasts for VAAYU.

Baseline statistical forecasting model (no meteorology) using:
- Daily PM-based AQI per station (4 pm - 4 pm IST window, min_valid_hours from aqi.py)
- Expanding window Ridge regression on log(AQI) for horizons h in {1, 2, 3}
- Pooled across stations using only days with target date <= D (strictly no future leakage)
- Features: AQI on D, AQI on D-1, mean AQI on D-6..D, PM10/PM2.5 ratio on D, month sin/cos
- Skip feature values across missing days (no filling); skip as-of date if < 30 training rows
- Predictive distribution via training empirical residuals (80% interval and category probabilities)
- Linear feature contributions with plain-language template names (no weather text)
- Verification against observed AQI for verifiable target days
- Evaluation over verifiable forecasts: Brier scores, Brier skill vs persistence and climatology
  (monthly rates from days <= D), 7-day block bootstrap 95% CIs with fixed seed, 80% interval coverage
- Caveats list: pooled stations, single winter, short history, no meteorology, insufficient events flag
"""

from datetime import date, datetime, timedelta, timezone
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import aqi

DATA_DIR = Path("data")
PROCESSED_HOURLY_CSV = DATA_DIR / "processed" / "openaq_hourly.csv"
BREAKPOINTS_CSV = DATA_DIR / "cpcb_breakpoints.csv"
OUTPUT_JSON = Path("site") / "data.json"

FEATURE_NAMES = [
    "aqi_D",
    "aqi_D_minus_1",
    "mean_aqi_7d",
    "pm_ratio",
    "month_sin",
    "month_cos",
]

TEMPLATE_NAMES = {
    "aqi_D": "current AQI level",
    "aqi_D_minus_1": "previous day AQI level",
    "mean_aqi_7d": "recent 7-day average AQI",
    "pm_ratio": "PM10 to PM2.5 ratio",
    "month_sin": "seasonal cycle (sin)",
    "month_cos": "seasonal cycle (cos)",
}


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert float NaNs, infinities, and numpy scalars to JSON-safe types."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


# --- Daily Table Builder ---

def build_daily_aqi_table(
    hourly_df: pd.DataFrame,
    breakpoints_df: Optional[pd.DataFrame] = None,
    min_valid_hours: Optional[int] = None,
) -> pd.DataFrame:
    """Compute daily PM2.5, PM10, and PM-based AQI per station for 4 pm - 4 pm IST windows."""
    if breakpoints_df is None:
        breakpoints_df = aqi.load_breakpoints(BREAKPOINTS_CSV)
    if min_valid_hours is None:
        min_valid_hours = inspect.signature(aqi.daily_average).parameters["min_valid_hours"].default

    df = hourly_df.copy()
    if not isinstance(df["Timestamp"].dtype, pd.DatetimeTZDtype):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        if df["Timestamp"].dt.tz is None:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("Asia/Kolkata")

    daily_records = []
    for station, grp in df.groupby("station"):
        grp = grp.sort_values("Timestamp").set_index("Timestamp")
        s_pm25 = aqi.daily_average(grp["pm25"], min_valid_hours=min_valid_hours)
        s_pm10 = aqi.daily_average(grp["pm10"], min_valid_hours=min_valid_hours)

        df_daily = pd.DataFrame({"pm25": s_pm25, "pm10": s_pm10})
        for rep_date, row in df_daily.iterrows():
            p25 = None if pd.isna(row["pm25"]) else float(row["pm25"])
            p10 = None if pd.isna(row["pm10"]) else float(row["pm10"])
            readings = {}
            if p25 is not None:
                readings["PM2.5"] = p25
            if p10 is not None:
                readings["PM10"] = p10

            val = aqi.aqi(readings, breakpoints_df=breakpoints_df) if readings else None
            daily_records.append({
                "date": rep_date,
                "station": station,
                "pm25": p25,
                "pm10": p10,
                "aqi": val,
            })

    daily_df = pd.DataFrame(daily_records)
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
        daily_df = daily_df.sort_values(["station", "date"]).reset_index(drop=True)
    return daily_df


# --- Feature Extraction with Strict Calendar Continuity ---

def compute_features_for_date(
    data_map: Dict[Tuple[str, date], Dict[str, Any]],
    station: str,
    as_of_date: date,
) -> Optional[List[float]]:
    """Compute lag and seasonal features for a given station and as-of date.

    Features:
    0. aqi_D: AQI on as_of_date
    1. aqi_D_minus_1: AQI on (as_of_date - 1 day)
    2. mean_aqi_7d: Mean AQI over contiguous 7 days [as_of_date - 6 days .. as_of_date]
    3. pm_ratio: PM10 / PM2.5 on as_of_date
    4. month_sin: sin(2 * pi * month / 12)
    5. month_cos: cos(2 * pi * month / 12)

    Returns None if any required day is missing (no interpolation or filling).
    """
    key_d = (station, as_of_date)
    if key_d not in data_map:
        return None
    r0 = data_map[key_d]
    aqi_0 = r0.get("aqi")
    if aqi_0 is None or pd.isna(aqi_0) or math.isnan(aqi_0) or aqi_0 <= 0:
        return None

    # aqi on D-1
    prev_d = as_of_date - timedelta(days=1)
    key_prev = (station, prev_d)
    if key_prev not in data_map:
        return None
    r_prev = data_map[key_prev]
    aqi_prev = r_prev.get("aqi")
    if aqi_prev is None or pd.isna(aqi_prev) or math.isnan(aqi_prev) or aqi_prev <= 0:
        return None

    # mean aqi on D-6..D (strict 7 calendar days)
    vals_7d = []
    for k in range(7):
        dk = as_of_date - timedelta(days=k)
        key_k = (station, dk)
        if key_k not in data_map:
            return None
        rk = data_map[key_k]
        aqi_k = rk.get("aqi")
        if aqi_k is None or pd.isna(aqi_k) or math.isnan(aqi_k) or aqi_k <= 0:
            return None
        vals_7d.append(float(aqi_k))
    mean_7d = float(np.mean(vals_7d))

    # PM10 / PM2.5 ratio on D
    pm25 = r0.get("pm25")
    pm10 = r0.get("pm10")
    if (
        pm25 is None
        or pd.isna(pm25)
        or math.isnan(pm25)
        or pm25 <= 0
        or pm10 is None
        or pd.isna(pm10)
        or math.isnan(pm10)
        or pm10 < 0
    ):
        return None
    pm_ratio = float(pm10 / pm25)

    # Month harmonics
    m = as_of_date.month
    m_sin = float(np.sin(2.0 * np.pi * m / 12.0))
    m_cos = float(np.cos(2.0 * np.pi * m / 12.0))

    res = [float(aqi_0), float(aqi_prev), mean_7d, pm_ratio, m_sin, m_cos]
    if any(math.isnan(x) or math.isinf(x) for x in res):
        return None
    return res


# --- Training Set Assembly (Expanding Window, No Leakage) ---

def build_training_set(
    data_map: Dict[Tuple[str, date], Dict[str, Any]],
    stations: Sequence[str],
    all_dates: Sequence[date],
    as_of_date: date,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble pooled training matrix (X, y) with target date strictly <= as_of_date."""
    X_tr = []
    y_tr = []

    for st in stations:
        for t_date in all_dates:
            target_date = t_date + timedelta(days=horizon)
            if target_date <= as_of_date:
                feats = compute_features_for_date(data_map, st, t_date)
                if feats is not None:
                    target_key = (st, target_date)
                    if target_key in data_map:
                        t_row = data_map[target_key]
                        if t_row["aqi"] is not None and not pd.isna(t_row["aqi"]) and t_row["aqi"] > 0:
                            X_tr.append(feats)
                            y_tr.append(np.log(float(t_row["aqi"])))

    if len(X_tr) == 0:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,))
    return np.array(X_tr, dtype=float), np.array(y_tr, dtype=float)


# --- Ridge Regression & Predictive Distribution ---

def fit_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit Ridge regression on standardized features.

    Returns:
        beta_0 (intercept), beta (coefficients), mu (means), sigma (stds), residuals.
    """
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0, ddof=1)
    sigma[sigma < 1e-9] = 1.0

    Z_train = (X_train - mu) / sigma
    y_bar = float(np.mean(y_train))
    y_c = y_train - y_bar

    # Analytical Ridge: beta = (Z'Z + alpha * I)^(-1) Z' y_c
    k = Z_train.shape[1]
    beta = np.linalg.solve(Z_train.T @ Z_train + alpha * np.eye(k), Z_train.T @ y_c)
    beta_0 = y_bar

    y_hat_train = beta_0 + Z_train @ beta
    residuals = y_train - y_hat_train
    return beta_0, beta, mu, sigma, residuals


def predict_distribution(
    beta_0: float,
    beta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    residuals: np.ndarray,
    features: Sequence[float],
    breakpoints_df: pd.DataFrame,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Compute point forecast, contributions, empirical samples, and category probabilities."""
    x = np.array(features, dtype=float)
    z = (x - mu) / sigma
    contribs = beta * z
    point_forecast = beta_0 + float(np.sum(contribs))
    point_aqi = float(np.exp(point_forecast))

    # Empirical predictive distribution on AQI scale
    samples = np.exp(point_forecast + residuals)
    q10 = float(np.percentile(samples, 10))
    q90 = float(np.percentile(samples, 90))

    # Category probabilities
    cat_names = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    cats = [aqi.category(float(val), breakpoints_df=breakpoints_df) for val in samples]
    cat_probs = {c: float(np.mean([1.0 if x == c else 0.0 for x in cats])) for c in cat_names}

    # Ensure probabilities sum strictly to 1.0
    tot_prob = sum(cat_probs.values())
    if tot_prob > 0:
        cat_probs = {k: v / tot_prob for k, v in cat_probs.items()}

    p_severe = float(np.mean([1.0 if val >= thresholds["Severe"] else 0.0 for val in samples]))
    p_very_poor_plus = float(np.mean([1.0 if val >= thresholds["Very Poor or worse"] else 0.0 for val in samples]))

    # Feature contributions
    all_contributions = []
    for j, fname in enumerate(FEATURE_NAMES):
        all_contributions.append({
            "feature": fname,
            "template_name": TEMPLATE_NAMES[fname],
            "contribution": float(contribs[j]),
            "abs_contribution": float(abs(contribs[j])),
            "feature_value": float(x[j]),
        })

    # Top 3 contributions by absolute magnitude
    top_reasons = sorted(all_contributions, key=lambda c: c["abs_contribution"], reverse=True)[:3]

    return {
        "intercept": float(beta_0),
        "point_forecast": float(point_forecast),
        "point_aqi": float(round(point_aqi, 1)),
        "interval_80": [round(q10, 1), round(q90, 1)],
        "category_probabilities": {k: round(v, 4) for k, v in cat_probs.items()},
        "p_severe": round(p_severe, 4),
        "p_very_poor_plus": round(p_very_poor_plus, 4),
        "all_contributions": all_contributions,
        "top_reasons": top_reasons,
    }


# --- Verification and Evaluation ---

def compute_brier_score(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """Mean squared error between binary outcomes and forecast probabilities."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean((y_p - y_t) ** 2))


def compute_brier_skill_score(bs_model: float, bs_ref: float) -> float:
    """Compute Brier Skill Score relative to reference forecast.

    Returns 0.0 if forecast is evaluated against itself (bs_model == bs_ref).
    """
    if bs_ref == bs_model:
        return 0.0
    if bs_ref == 0.0:
        return 0.0 if bs_model == 0.0 else -float("inf")
    return float(1.0 - (bs_model / bs_ref))


def bootstrap_metric_ci(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    y_ref: Sequence[float],
    block_size: int = 7,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """Compute 95% bootstrap confidence intervals for Brier Score and Brier Skill Score."""
    rng = np.random.RandomState(seed)
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    y_r = np.asarray(y_ref, dtype=float)
    n = len(y_t)

    if n < block_size or n == 0:
        bs_m = float(np.mean((y_p - y_t) ** 2)) if n > 0 else 0.0
        bs_r = float(np.mean((y_r - y_t) ** 2)) if n > 0 else 0.0
        bss = compute_brier_skill_score(bs_m, bs_r)
        return [bs_m, bs_m], [bss, bss]

    blocks = [list(range(i, min(i + block_size, n))) for i in range(0, n, block_size)]
    n_blocks = len(blocks)

    boot_bs = []
    boot_bss = []
    for _ in range(n_boot):
        chosen = []
        for _ in range(n_blocks):
            b_idx = rng.randint(0, n_blocks)
            chosen.extend(blocks[b_idx])
        idx = chosen[:n]

        bs_m = float(np.mean((y_p[idx] - y_t[idx]) ** 2))
        bs_r = float(np.mean((y_r[idx] - y_t[idx]) ** 2))
        bss = compute_brier_skill_score(bs_m, bs_r)
        boot_bs.append(bs_m)
        boot_bss.append(bss)

    ci_bs = [round(float(np.percentile(boot_bs, 2.5)), 4), round(float(np.percentile(boot_bs, 97.5)), 4)]
    ci_bss = [round(float(np.percentile(boot_bss, 2.5)), 4), round(float(np.percentile(boot_bss, 97.5)), 4)]
    return ci_bs, ci_bss


def evaluate_forecast_set(
    forecast_list: List[Dict[str, Any]],
    thresholds: Dict[str, float],
    data_map: Dict[Tuple[str, date], Dict[str, Any]],
    all_dates: Sequence[date],
    stations: Sequence[str],
    block_size: int = 7,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute verification statistics, Brier scores, BSS, bootstrap CIs, and interval coverage."""
    verifiable = [f for f in forecast_list if f.get("observed_aqi") is not None]
    n_total = len(verifiable)

    if n_total == 0:
        return {
            "n_forecasts": 0,
            "n_events_severe": 0,
            "n_events_very_poor_plus": 0,
            "coverage_80": 0.0,
            "severe": {},
            "very_poor_plus": {},
        }

    # Observed outcomes
    obs_severe = [1.0 if f["observed_aqi"] >= thresholds["Severe"] else 0.0 for f in verifiable]
    obs_vp = [1.0 if f["observed_aqi"] >= thresholds["Very Poor or worse"] else 0.0 for f in verifiable]
    n_events_severe = int(sum(obs_severe))
    n_events_vp = int(sum(obs_vp))

    # Coverage of 80% interval
    covered_count = 0
    for f in verifiable:
        q10, q90 = f["interval_80"]
        if q10 <= f["observed_aqi"] <= q90:
            covered_count += 1
    coverage_80 = round(float(covered_count / n_total), 4)

    # Model probabilities
    prob_severe = [f["p_severe"] for f in verifiable]
    prob_vp = [f["p_very_poor_plus"] for f in verifiable]

    # Reference 1: Persistence (based on as-of date D AQI)
    pers_severe = []
    pers_vp = []
    for f in verifiable:
        d = date.fromisoformat(f["as_of_date"])
        st = f["station"]
        row_d = data_map.get((st, d))
        aqi_d = row_d["aqi"] if row_d else 0.0
        pers_severe.append(1.0 if aqi_d and aqi_d >= thresholds["Severe"] else 0.0)
        pers_vp.append(1.0 if aqi_d and aqi_d >= thresholds["Very Poor or worse"] else 0.0)

    # Reference 2: Climatology (monthly event rates from days <= D)
    clim_severe = []
    clim_vp = []
    for f in verifiable:
        d = date.fromisoformat(f["as_of_date"])
        t_date = date.fromisoformat(f["target_date"])
        target_month = t_date.month

        # Historical target pool <= d
        hist_aqis_all = []
        hist_aqis_month = []
        for s in stations:
            for dt in all_dates:
                if dt <= d:
                    r = data_map.get((s, dt))
                    if r and r["aqi"] is not None and not pd.isna(r["aqi"]) and r["aqi"] > 0:
                        hist_aqis_all.append(float(r["aqi"]))
                        if dt.month == target_month:
                            hist_aqis_month.append(float(r["aqi"]))

        pool_sev = hist_aqis_month if hist_aqis_month else hist_aqis_all
        pool_vp = hist_aqis_month if hist_aqis_month else hist_aqis_all

        rate_sev = float(np.mean([1.0 if v >= thresholds["Severe"] else 0.0 for v in pool_sev])) if pool_sev else 0.0
        rate_vp = float(np.mean([1.0 if v >= thresholds["Very Poor or worse"] else 0.0 for v in pool_vp])) if pool_vp else 0.0

        clim_severe.append(rate_sev)
        clim_vp.append(rate_vp)

    # Severe Brier Scores & Skill
    bs_sev_model = compute_brier_score(obs_severe, prob_severe)
    bs_sev_pers = compute_brier_score(obs_severe, pers_severe)
    bs_sev_clim = compute_brier_score(obs_severe, clim_severe)

    bss_sev_pers = compute_brier_skill_score(bs_sev_model, bs_sev_pers)
    bss_sev_clim = compute_brier_skill_score(bs_sev_model, bs_sev_clim)

    ci_bs_sev, ci_bss_sev_pers = bootstrap_metric_ci(obs_severe, prob_severe, pers_severe, block_size, n_boot, seed)
    _, ci_bss_sev_clim = bootstrap_metric_ci(obs_severe, prob_severe, clim_severe, block_size, n_boot, seed)

    # Very Poor+ Brier Scores & Skill
    bs_vp_model = compute_brier_score(obs_vp, prob_vp)
    bs_vp_pers = compute_brier_score(obs_vp, pers_vp)
    bs_vp_clim = compute_brier_score(obs_vp, clim_vp)

    bss_vp_pers = compute_brier_skill_score(bs_vp_model, bs_vp_pers)
    bss_vp_clim = compute_brier_skill_score(bs_vp_model, bs_vp_clim)

    ci_bs_vp, ci_bss_vp_pers = bootstrap_metric_ci(obs_vp, prob_vp, pers_vp, block_size, n_boot, seed)
    _, ci_bss_vp_clim = bootstrap_metric_ci(obs_vp, prob_vp, clim_vp, block_size, n_boot, seed)

    return {
        "n_forecasts": n_total,
        "n_events_severe": n_events_severe,
        "n_events_very_poor_plus": n_events_vp,
        "coverage_80": coverage_80,
        "severe": {
            "brier_score": round(bs_sev_model, 4),
            "brier_score_ci": ci_bs_sev,
            "brier_score_persistence": round(bs_sev_pers, 4),
            "brier_skill_vs_persistence": round(bss_sev_pers, 4),
            "brier_skill_vs_persistence_ci": ci_bss_sev_pers,
            "brier_score_climatology": round(bs_sev_clim, 4),
            "brier_skill_vs_climatology": round(bss_sev_clim, 4),
            "brier_skill_vs_climatology_ci": ci_bss_sev_clim,
        },
        "very_poor_plus": {
            "brier_score": round(bs_vp_model, 4),
            "brier_score_ci": ci_bs_vp,
            "brier_score_persistence": round(bs_vp_pers, 4),
            "brier_skill_vs_persistence": round(bss_vp_pers, 4),
            "brier_skill_vs_persistence_ci": ci_bss_vp_pers,
            "brier_score_climatology": round(bs_vp_clim, 4),
            "brier_skill_vs_climatology": round(bss_vp_clim, 4),
            "brier_skill_vs_climatology_ci": ci_bss_vp_clim,
        },
    }


# --- Pipeline Execution ---

def run_forecast_pipeline(
    hourly_csv_path: Union[str, Path] = PROCESSED_HOURLY_CSV,
    breakpoints_path: Union[str, Path] = BREAKPOINTS_CSV,
    output_json_path: Optional[Union[str, Path]] = OUTPUT_JSON,
    min_train_rows: int = 30,
    ridge_alpha: float = 1.0,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete forecast generation, verification, and evaluation pipeline."""
    hourly_csv = Path(hourly_csv_path)
    if not hourly_csv.exists():
        raise FileNotFoundError(f"Processed hourly CSV not found: {hourly_csv.resolve()}")

    breakpoints_df = aqi.load_breakpoints(breakpoints_path)
    min_valid_hours = inspect.signature(aqi.daily_average).parameters["min_valid_hours"].default
    thresholds = aqi.get_category_thresholds(breakpoints_path)

    # 1. Load hourly data and filter for 2025-26 winter season
    df_hourly = pd.read_csv(hourly_csv)
    df_hourly["Timestamp"] = pd.to_datetime(df_hourly["Timestamp"])
    df_2526 = df_hourly[(df_hourly["Timestamp"] >= "2025-10-01") & (df_hourly["Timestamp"] <= "2026-03-01")].copy()

    # 2. Build daily AQI table
    daily_df = build_daily_aqi_table(df_2526, breakpoints_df=breakpoints_df, min_valid_hours=min_valid_hours)

    data_map: Dict[Tuple[str, date], Dict[str, Any]] = {}
    for _, row in daily_df.iterrows():
        data_map[(row["station"], row["date"])] = row.to_dict()

    all_dates = sorted(daily_df["date"].unique())
    stations = sorted(daily_df["station"].unique())
    horizons = [1, 2, 3]

    forecasts: List[Dict[str, Any]] = []

    # 3. For each as-of date D and horizon h, fit expanding-window Ridge regression
    for d in all_dates:
        for h in horizons:
            X_tr, y_tr = build_training_set(data_map, stations, all_dates, d, h)
            if len(X_tr) < min_train_rows:
                continue

            beta_0, beta, mu, sigma, residuals = fit_ridge_regression(X_tr, y_tr, alpha=ridge_alpha)

            for st in stations:
                feats = compute_features_for_date(data_map, st, d)
                if feats is None:
                    continue

                dist = predict_distribution(beta_0, beta, mu, sigma, residuals, feats, breakpoints_df, thresholds)

                target_date = d + timedelta(days=h)
                obs_key = (st, target_date)
                obs_aqi = None
                obs_cat = None
                if obs_key in data_map:
                    obs_row = data_map[obs_key]
                    if obs_row["aqi"] is not None and not pd.isna(obs_row["aqi"]):
                        obs_aqi = float(obs_row["aqi"])
                        obs_cat = aqi.category(obs_aqi, breakpoints_df=breakpoints_df)

                rec = {
                    "as_of_date": d.isoformat(),
                    "station": st,
                    "horizon": int(h),
                    "target_date": target_date.isoformat(),
                    "n_train": int(len(X_tr)),
                    "intercept": dist["intercept"],
                    "point_forecast": dist["point_forecast"],
                    "point_aqi": dist["point_aqi"],
                    "interval_80": dist["interval_80"],
                    "category_probabilities": dist["category_probabilities"],
                    "p_severe": dist["p_severe"],
                    "p_very_poor_plus": dist["p_very_poor_plus"],
                    "top_reasons": dist["top_reasons"],
                    "all_contributions": dist["all_contributions"],
                    "observed_aqi": obs_aqi,
                    "observed_category": obs_cat,
                }
                forecasts.append(rec)

    # 4. Evaluation over all verifiable forecasts (overall and by horizon)
    eval_overall = evaluate_forecast_set(
        forecasts, thresholds, data_map, all_dates, stations, block_size=7, n_boot=1000, seed=bootstrap_seed
    )

    eval_by_horizon = {}
    for h in horizons:
        f_h = [f for f in forecasts if f["horizon"] == h]
        eval_by_horizon[str(h)] = evaluate_forecast_set(
            f_h, thresholds, data_map, all_dates, stations, block_size=7, n_boot=1000, seed=bootstrap_seed
        )

    # 5. Caveats list
    caveats = [
        "Models are pooled across the three monitoring stations (Anand Vihar, Indirapuram, Sector 11 Faridabad).",
        "Trained on a single winter season (2025-26) using expanding historical training windows.",
        "Short operational history with limited training samples during early winter.",
        "Statistical baseline only: incorporates no meteorological, boundary layer, or chemical transport physics.",
    ]
    if eval_overall["n_events_severe"] < 10:
        caveats.append(f"insufficient events: fewer than 10 Severe events ({eval_overall['n_events_severe']} observed) in evaluation set.")
    if eval_overall["n_events_very_poor_plus"] < 10:
        caveats.append(f"insufficient events: fewer than 10 Very Poor or worse events ({eval_overall['n_events_very_poor_plus']} observed) in evaluation set.")

    forecast_output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": "Expanding window Ridge regression on log(AQI) with empirical residuals",
            "features": [
                {"name": fname, "template_name": TEMPLATE_NAMES[fname]} for fname in FEATURE_NAMES
            ],
            "horizons": horizons,
            "min_train_rows": min_train_rows,
            "ridge_alpha": ridge_alpha,
            "interval_level": 0.80,
            "bootstrap_seed": bootstrap_seed,
        },
        "caveats": caveats,
        "evaluation": {
            "overall": eval_overall,
            "by_horizon": eval_by_horizon,
        },
        "forecasts": forecasts,
    }

    forecast_output = sanitize_for_json(forecast_output)

    # 6. Save into site/data.json under "forecast"
    if output_json_path is not None:
        out_p = Path(output_json_path)
        if out_p.exists():
            with open(out_p, "r", encoding="utf-8") as fp:
                existing_data = json.load(fp)
        else:
            existing_data = {}

        existing_data["forecast"] = forecast_output
        with open(out_p, "w", encoding="utf-8") as fp:
            json.dump(existing_data, fp, indent=2)

    return forecast_output


if __name__ == "__main__":
    res = run_forecast_pipeline()
    eval_ov = res["evaluation"]["overall"]
    print("=" * 76)
    print("VAAYU Baseline Forecast Pipeline Complete")
    print("=" * 76)
    print(f"Total Forecasts Generated:    {len(res['forecasts'])}")
    print(f"Verifiable Forecasts:         {eval_ov['n_forecasts']}")
    print(f"Severe Events Observed:       {eval_ov['n_events_severe']}")
    print(f"Very Poor+ Events Observed:   {eval_ov['n_events_very_poor_plus']}")
    print(f"80% Interval Coverage:        {eval_ov['coverage_80'] * 100:.1f}%")
    print(f"Brier Score (Severe):         {eval_ov['severe']['brier_score']}")
    print(f"BSS vs Persistence (Severe):  {eval_ov['severe']['brier_skill_vs_persistence']}")
    print(f"BSS vs Climatology (Severe):  {eval_ov['severe']['brier_skill_vs_climatology']}")
    print(f"Brier Score (Very Poor+):     {eval_ov['very_poor_plus']['brier_score']}")
    print(f"BSS vs Persistence (VP+):     {eval_ov['very_poor_plus']['brier_skill_vs_persistence']}")
    print(f"BSS vs Climatology (VP+):     {eval_ov['very_poor_plus']['brier_skill_vs_climatology']}")
    print("\nCaveats:")
    for c in res["caveats"]:
        print(f"  • {c}")
    print("=" * 76)
