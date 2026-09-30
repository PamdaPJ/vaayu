# VAAYU 72-Hour Air Quality Forecast: Verification & Audit Report

**Evaluation Season:** Held-out Winter 2025-10-01 to 2026-02-28  
**Training Period:** Earlier Winters (2020-10-01 to 2021-02-28, 2021-10-01 to 2022-02-28, 2023-10-01 to 2024-02-29)  
**Stations Evaluated:** 3 monitoring stations  
**Verifiable Observations:** 23559 hours | **Severe Hours:** 5408  

> [!IMPORTANT]
> **Methodology & Provenance Audit:**
> 1. **Zero Imputed Targets Guarantee:** Persistence-imputed values (from the `last_obs_*` features at issue time) are strictly used as input features and are NEVER used as evaluation ground-truth targets. Only genuine measured ground station observations are evaluated.
> 2. **Driver Provenance Disclosure:** Air Quality features (`pm2_5`, `pm10`, `ozone`, `nitrogen_dioxide`) are retrieved from the Open-Meteo Air Quality API (`air-quality-api.open-meteo.com/v1/air-quality`) which serves CAMS regional atmospheric composition reanalysis/analysis across historical periods. Open-Meteo does not archive previous model runs for CAMS air quality. Weather features are retrieved from `archive-api.open-meteo.com/v1/archive` (ERA5 reanalysis). Consequently, historical values at lead $h$ are **reanalysis/analysis values**, meaning the held-out evaluation is technically a **hindcast-with-analysis-drivers** and overstates true operational forecast skill where CAMS forecast errors would degrade over lead time.

---

## 1. Data Exclusions and Imputation Audit

Total calendar hours in the held-out winter season: **3,624 hours per station** (151 days × 24h). All hours with missing ground-truth observations were strictly excluded from verification targets:

| Station | Pollutant | Calendar Hours | Valid Ground Hours | Excluded / Missing Hours | Excluded Pct |
|---|---|---|---|---|---|
| `anand_vihar_new_delhi_dpcc` | **PM25** | 3624 | 2828 | 796 | 22.0% |
| `anand_vihar_new_delhi_dpcc` | **PM10** | 3624 | 2818 | 806 | 22.2% |
| `anand_vihar_new_delhi_dpcc` | **O3** | 3624 | 2144 | 1480 | 40.8% |
| `anand_vihar_new_delhi_dpcc` | **NO2** | 3624 | 2197 | 1427 | 39.4% |
| `indirapuram_ghaziabad_uppcb` | **PM25** | 3624 | 2717 | 907 | 25.0% |
| `indirapuram_ghaziabad_uppcb` | **PM10** | 3624 | 2798 | 826 | 22.8% |
| `indirapuram_ghaziabad_uppcb` | **O3** | 3624 | 0 | 3624 | 100.0% |
| `indirapuram_ghaziabad_uppcb` | **NO2** | 3624 | 0 | 3624 | 100.0% |
| `sector_11_faridabad_hspcb` | **PM25** | 3624 | 2364 | 1260 | 34.8% |
| `sector_11_faridabad_hspcb` | **PM10** | 3624 | 2378 | 1246 | 34.4% |
| `sector_11_faridabad_hspcb` | **O3** | 3624 | 0 | 3624 | 100.0% |
| `sector_11_faridabad_hspcb` | **NO2** | 3624 | 0 | 3624 | 100.0% |

> **Note on O3 and NO2 Ground Sensors:** Indirapuram and Sector 11 Faridabad ground stations in the open dataset do not report continuous O3 and NO2 channels (100% missing). NO2 and O3 evaluation is exclusively performed on Anand Vihar.

---

## 2. Multi-Pollutant Continuous Metrics across Lead Buckets

Evaluated against three independent reference baselines:
1. **Raw CAMS:** Copernicus Atmosphere Monitoring Service numerical driver forecast without statistical correction.
2. **Persistence:** Persisting the last valid station observation at or before issue time out to +72h.
3. **Climatology:** Station-specific historical median by month and hour of day from training winters.

### PM25 Verification

| Lead Bucket | Valid Samples | Model MAE | Raw CAMS MAE | Persistence MAE | Climatology MAE | Skill vs CAMS | Skill vs Pers | Skill vs Clim | 80% Int Coverage |
|---|---|---|---|---|---|---|---|---|---|
| **1-24h** | 7708 | **84.08** | 69.34 | 74.93 | 93.20 | **-21.3%** | -12.2% | +9.8% | 53.1% |
| **25-48h** | 7745 | **84.31** | 69.39 | 85.87 | 93.00 | **-21.5%** | +1.8% | +9.3% | 51.9% |
| **49-72h** | 7768 | **84.61** | 69.82 | 90.75 | 92.92 | **-21.2%** | +6.8% | +8.9% | 50.4% |
| **1-72h** | 23221 | **84.33** | 69.52 | 83.78 | 93.04 | **-21.3%** | -0.7% | +9.4% | 51.8% |

### PM10 Verification

| Lead Bucket | Valid Samples | Model MAE | Raw CAMS MAE | Persistence MAE | Climatology MAE | Skill vs CAMS | Skill vs Pers | Skill vs Clim | 80% Int Coverage |
|---|---|---|---|---|---|---|---|---|---|
| **1-24h** | 7796 | **170.48** | 191.44 | 154.48 | 185.70 | **+10.9%** | -10.4% | +8.2% | 62.0% |
| **25-48h** | 7833 | **172.63** | 191.76 | 172.46 | 185.35 | **+10.0%** | -0.1% | +6.9% | 60.6% |
| **49-72h** | 7856 | **173.73** | 193.07 | 181.34 | 185.34 | **+10.0%** | +4.2% | +6.3% | 59.8% |
| **1-72h** | 23485 | **172.28** | 192.09 | 169.32 | 185.46 | **+10.3%** | -1.7% | +7.1% | 60.8% |

### O3 Verification

| Lead Bucket | Valid Samples | Model MAE | Raw CAMS MAE | Persistence MAE | Climatology MAE | Skill vs CAMS | Skill vs Pers | Skill vs Clim | 80% Int Coverage |
|---|---|---|---|---|---|---|---|---|---|
| **1-24h** | 2143 | **11.02** | 71.23 | 5.00 | 9.41 | **+84.5%** | -120.4% | -17.1% | 67.3% |
| **25-48h** | 2120 | **11.12** | 70.97 | 5.84 | 9.46 | **+84.3%** | -90.4% | -17.5% | 66.1% |
| **49-72h** | 2099 | **11.07** | 70.45 | 6.40 | 9.48 | **+84.3%** | -73.0% | -16.8% | 65.2% |
| **1-72h** | 6362 | **11.07** | 70.89 | 5.74 | 9.45 | **+84.4%** | -92.9% | -17.1% | 66.2% |

### NO2 Verification

| Lead Bucket | Valid Samples | Model MAE | Raw CAMS MAE | Persistence MAE | Climatology MAE | Skill vs CAMS | Skill vs Pers | Skill vs Clim | 80% Int Coverage |
|---|---|---|---|---|---|---|---|---|---|
| **1-24h** | 2196 | **31.44** | 46.92 | 22.97 | 31.24 | **+33.0%** | -36.9% | -0.6% | 49.2% |
| **25-48h** | 2172 | **32.06** | 47.26 | 25.23 | 31.06 | **+32.2%** | -27.1% | -3.2% | 49.1% |
| **49-72h** | 2148 | **32.34** | 47.63 | 26.70 | 30.83 | **+32.1%** | -21.1% | -4.9% | 49.9% |
| **1-72h** | 6516 | **31.94** | 47.27 | 24.95 | 31.04 | **+32.4%** | -28.0% | -2.9% | 49.4% |

---

## 3. PM2.5 In-Depth Diagnosis

### 3.1 Error Breakdown by Month and Hour of Day
- **Peak Smoke Season (November):** The statistical corrector outperforms Raw CAMS (**Model MAE 80.28 vs Raw CAMS 91.26 µg/m³**), reducing extreme underprediction spikes.
- **Winter Inversion Peak (December):** Model MAE (84.94 µg/m³) is competitive with Raw CAMS (83.49 µg/m³), with Model Mean Bias near zero (+0.14 µg/m³ vs CAMS -63.57 µg/m³).
- **Late Season (January–February):** Ambient PM2.5 levels drop (Jan obs mean: 135.1, Feb obs mean: 125.6 µg/m³). Raw CAMS exhibits lower variance (MAE ~54 µg/m³), while the corrector overpredicts (Model MAE 95–107 µg/m³) due to high winter training bias.
- **Diurnal Profile:** Highest diurnal errors occur during late evening inversion onset (20:00–04:00 IST), where ground observations spike to 170–190 µg/m³ while Raw CAMS plateaus at 110–120 µg/m³.

### 3.2 Station Heterogeneity (Anand Vihar vs Faridabad)
- **Anand Vihar (New Delhi):** Observed mean is 199.7 µg/m³. Raw CAMS severely underpredicts (mean 97.0 µg/m³, bias -102.75). The **corrector beats Raw CAMS by 22.1%** (**Model MAE 82.21 vs Raw CAMS 105.56 µg/m³**).
- **Sector 11 Faridabad:** Observed mean is 102.2 µg/m³. Raw CAMS happens to be nearly unbiased (+0.53 µg/m³, MAE 43.24). The corrector (trained primarily on Anand Vihar data) predicts ~188 µg/m³, inflating MAE to 99.02 µg/m³.

### 3.3 Training Distribution Shift & Formulation Experiments
- **Distribution Shift:** Training winters (2020-21, 2021-22) contained only **240 distinct valid PM2.5 hours** (mean 183.0 µg/m³), whereas the test winter had **7,909 hours** (mean 150.3 µg/m³).
- **Residual Target Experiment (`obs - CAMS`):** Training LightGBM to predict the residual yields an overall MAE of **69.52 µg/m³** (bias -51.55 µg/m³), which exactly matches Raw CAMS because the residual model predicts $\approx 0$ everywhere.
- **Station Categorical Bias Experiment:** Adding station identity into training yields MAE of **85.33 µg/m³**, as station bias terms overfit the sparse training samples.

---

## 4. AQI Category Accuracy (Exact & Within-One-Tier)

Evaluated on 6 official CPCB NAQI tiers (Good, Satisfactory, Moderate, Poor, Very Poor, Severe):

| Lead Bucket | Valid Hours | Model Exact | Model Within ±1 Tier | Raw CAMS Exact | Raw CAMS Within ±1 Tier | Persistence Exact | Climatology Exact |
|---|---|---|---|---|---|---|---|
| **1-24h** | 7820 | **31.6%** | **70.4%** | 23.9% | 66.6% | 41.8% | 30.8% |
| **25-48h** | 7858 | **31.6%** | **70.6%** | 23.9% | 66.4% | 36.6% | 30.9% |
| **49-72h** | 7881 | **31.9%** | **70.6%** | 23.8% | 66.1% | 35.1% | 31.0% |
| **1-72h** | 23559 | **31.7%** | **70.5%** | 23.9% | 66.4% | 37.9% | 30.9% |

---

## 5. P(Severe) Probabilistic Calibration & Brier Skill Scores

| Lead Bucket | Evaluated Hours | Observed Severe Hours | Model Brier Score | Climatology BS | Persistence BS | BSS vs Climatology | BSS vs Persistence |
|---|---|---|---|---|---|---|---|
| **1-24h** | 7820 | 1788 | **0.2510** | 0.1994 | 0.2840 | **-0.2592** | **+0.1161** |
| **25-48h** | 7858 | 1800 | **0.2551** | 0.1995 | 0.3095 | **-0.2788** | **+0.1758** |
| **49-72h** | 7881 | 1820 | **0.2565** | 0.1999 | 0.3322 | **-0.2828** | **+0.2279** |
| **1-72h** | 23559 | 5408 | **0.2542** | 0.1996 | 0.3086 | **-0.2736** | **+0.1763** |

### Reliability Diagram Bins (Overall 1-72h)

| Predicted Probability Bin | Sample Count | Mean Forecast Probability | Observed Event Fraction |
|---|---|---|---|
| `[0.0, 0.2]` | 6845 | 0.0657 | 0.0554 |
| `[0.2, 0.4]` | 3788 | 0.3015 | 0.2284 |
| `[0.4, 0.6]` | 3972 | 0.4990 | 0.2621 |
| `[0.6, 0.8]` | 4181 | 0.7054 | 0.3332 |
| `[0.8, 1.0]` | 4773 | 0.8955 | 0.3625 |

---

## 6. Season Coverage Explanation

- **Winters 2022-23 and 2024-25 Exclusion:** The repository's ground truth dataset (`data/processed/openaq_hourly.csv`) contains continuous observations strictly for Winters 2020-21, 2021-22, and 2025-26. Historical ground data for 2022-23 and 2024-25 was neither cached in `data/raw/openaq` nor included in the repository, and no OpenAQ v3 API key is configured in the environment. Unverified raw CPCB downloads for 2023 were excluded due to duplicate station series (`test_raw_cpcb_fails_loudly_on_duplicates`).
