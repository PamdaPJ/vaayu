
# VAAYU

**Coupled smoke–weather risk forecasting for Delhi NCR: fix the fire input, measure the feedback, forecast the risk.**

Smart India Hackathon 2026 · Problem Statement 26082 · Air Pollution–Weather Coupled Forecasting System (Delhi NCR Focus) · Ministry of Earth Sciences (MoES), NCMRWF
Team: **Boondi Laddoos**

> **Status: MVP.** This repository contains the deployable slice of VAAYU. The coupled-physics tier (WRF-Chem) is the planned next phase. See [Built vs Planned](#built-vs-planned). Nothing here is presented as more than it is.

---

## The problem

Delhi's air-quality forecasts tend to miss the worst days. Our working hypothesis has two parts:

1. **Fire input is under-represented.** Polar-orbiting satellites pass over Punjab around early afternoon, but much large-scale crop-residue burning is reported to happen later in the day.
2. **Aerosol–weather feedback is not measured.** Smoke dims sunlight, the surface cools, the inversion lid strengthens and more smoke is trapped.

Published evaluations (CEEW 2025; iFOREST 2025) report these gaps. **Verify all figures against the original sources before quoting them**; this README deliberately does not restate numbers we have not re-checked.

## What VAAYU does

| Stage | Idea |
|---|---|
| Fix the fire input | Reconstruct afternoon burning from 15-minute geostationary fire data, with small-fire completeness taken from VIIRS statistics |
| Measure the feedback | Offline WRF-Chem 2×2 experiment (official vs corrected fires × feedback on/off) plus a noise-floor run |
| Forecast the risk | Live probabilistic layer producing P(Severe AQI) from PM2.5 and PM10 jointly, under CPCB rules |

VAAYU is designed to **complement** MoES operational systems (AQEWS, IMD-SILAM, DM-Chem), not replace them.

---

## Built vs Planned

Tick each box only when the item runs from a clean clone.

### Built in this MVP

- [ ] **CPCB AQI engine** (`aqi.py`): sub-indices by interpolation within official breakpoints, AQI = max sub-index, 24-hour average from 4 pm to 4 pm, and `p_severe()` computed from joint PM2.5 + PM10 samples. Covered by unit tests.
- [ ] **Fire-timing evidence** (`fires.py`): time-of-day distribution of NASA FIRMS VIIRS detections over Punjab (Oct–Nov), with the satellite overpass window marked.
- [ ] **Probabilistic baseline** (`baseline.py`): next-day P(Severe) model, evaluated with Brier score and a reliability diagram on a held-out winter.
- [ ] **Demo page** (`index.html`): one page showing the three outputs above.

### Planned (not built yet)

- WRF-Chem factorial runs (R1–R4) and the noise-floor run R1′
- SEVIRI + VIIRS fire-energy reconstruction (detection-efficiency model)
- Multi-output corrector (PM2.5, PM10, O₃, NO₂) with shared sample trajectories
- Regression-kriging to 1 km probabilistic maps with per-cell uncertainty
- Inversion diagnostics from model profiles
- Live daily pipeline, API and public verification page

### What the MVP does **not** claim

- It does not run coupled chemistry–weather physics.
- It does not produce 1 km maps.
- The fire-timing chart shows when VIIRS **detected** fires, not when fires actually burned. The after-3 pm burning claim comes from the cited literature, not from this repo.
- The baseline model is a simple reference to demonstrate the evaluation method. It is **not** claimed to beat AQEWS.

---

## Results

Fill these in from the actual runs. Do not edit numbers by hand.

| Metric | Value | Notes |
|---|---|---|
| Held-out winter | Winter 2025-10-01 to 2026-02-28 | Out-of-sample test season |
| Brier score (72h P(Severe)) | 0.2542 | Evaluated across 1-72h lead window |
| Brier score (climatology reference) | 0.2273 | Historical winter climatology |
| Brier skill score (vs climatology) | -0.1186 | Positive value indicates forecasting skill |
| AQI category accuracy (PM-based) | 31.7% | 6-tier CPCB category match (1-72h) |
| PM2.5 corrector MAE (1-72h) | 84.33 µg/m³ | Quantile median p50 forecast |
| PM10 corrector MAE (1-72h) | 172.28 µg/m³ | Quantile median p50 forecast |
| Number of Severe hours in test set | 5408 | Observed AQI > 400 | _TBD_ | Small counts mean wide uncertainty |

With few Severe days, detection counts carry very wide confidence intervals. We therefore lead with Brier score and reliability, and treat POD/FAR as secondary.

---

## Quick start

```bash
git clone https://github.com/<your-username>/vaayu.git
cd vaayu
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                           # run the AQI unit tests
python fires.py                  # writes the fire-timing chart to docs/
python baseline.py               # prints Brier score, writes the reliability diagram
```

Open `index.html` in a browser to see the demo page.

## Repository layout

```
vaayu/
├── aqi.py            # CPCB AQI engine
├── fires.py          # FIRMS fire-timing analysis
├── baseline.py       # baseline probabilistic model + evaluation
├── index.html        # demo page
├── tests/            # pytest unit tests
├── data/             # inputs (see below); large files are not committed
├── docs/             # generated figures
├── requirements.txt
└── README.md
```

## Data

| Source | Use | Access |
|---|---|---|
| NASA FIRMS (VIIRS) | Fire detections and acquisition time | Free API key |
| CPCB breakpoint table | Official AQI definition | Copy from CPCB's published table into `data/cpcb_breakpoints.csv` |
| CPCB station data (e.g. via OpenAQ) | PM2.5, PM10 for the baseline model | Check current access terms |

**API keys:** store them in a local `.env` file that is listed in `.gitignore`. Never commit keys.

## Testing

- AQI function tested against the breakpoint table, including boundary values (for example PM2.5 just above the Severe threshold) and the max-sub-index rule.
- Baseline evaluation uses a **time-based split by winter**, so future data cannot leak into training.

## Roadmap

1. Weeks 1–3: WRF-Chem build and benchmark; live CAMS/GFS pipeline archiving the 2026–27 season
2. Weeks 4–7: factorial runs, fire reconstruction, corrector training, inversion diagnostics
3. Weeks 8–9: integration, dashboard, hindcast on a held-out season
4. Week 10+: hardening, failure tests, live verification page

Compute estimates for WRF-Chem are order-of-magnitude and will be replaced by measured benchmarks.

## References

Confirm each entry before citing it.

- Ignatious S., Rafiuddin M. (2025). *How Well Can Delhi Predict Air Quality?* CEEW Issue Brief.
- iFOREST (2025). *Stubble Burning Status Report 2025.*
- Grell G.A. et al. (2005). Fully coupled "online" chemistry within the WRF model. *Atmospheric Environment* 39.
- Wooster M.J. et al. (2005). Retrieval of biomass combustion rates and totals from fire radiative power observations. *JGR* 110.
- CAQM (2024). *Graded Response Action Plan for NCR.*

## License

_Add a license (for example MIT) before making the repository public._
