# VAAYU Lite: Runnable Prototype — Stitch Design Specification

This directory contains the exported responsive HTML and CSS screens for **VAAYU Lite: runnable prototype**, generated via the Google Stitch MCP toolchain.

## Stitch Project Telemetry
- **Project ID**: `891825844227298585`
- **Design System Asset**: `assets/3121512170262063514`
- **Palette**:
  - Dark Slate background: `#1e2a35` (`--bg-slate`)
  - Terracotta accent: `#c9884f` (`--accent-terracotta`)
  - Deep red for Severe: `#8b1c2c` (`--severe-red`)
  - Teal observatory secondary: `#2f6f73` (`--teal-observatory`)
  - Off-white typography: `#f5f5f0` (`--text-parchment`)
- **Typography**:
  - Headlines & Section Titles: Serif (`EB Garamond`, `Merriweather`)
  - Body & Form Elements: Clean Sans (`Inter`)
  - Telemetry & Sounding Metrics: Tabular Monospace (`JetBrains Mono`)

---

## Exported Screens

### 1. Overview ([overview.html](file:///c:/Users/PRATYAKSHA/vaayu/design/stitch/overview.html))
- **Stitch Screen ID**: `733827018da24b3ab1e8c085a67890b5`
- **Components**:
  - Top status message notification banner slot (`[STATUS: {{system_status}}] Notice: CPCB Breakpoint verification status: {{verification_status}} | Last pipeline execution: {{pipeline_timestamp}}`).
  - Serif headline and institutional subtitle for the NCR Atmospheric Observatory.
  - Verbatim one-paragraph diagnostic problem statement callout with terracotta accent border.
  - Two-column "Built vs Planned" matrix comparing operational prototype capabilities against roadmap features.
  - Diagnostic footer with pipeline version and verification metadata.

### 2. Station Explorer ([station_explorer.html](file:///c:/Users/PRATYAKSHA/vaayu/design/stitch/station_explorer.html))
- **Stitch Screen ID**: `313295e1808549318c93752e49d0839b`
- **Components**:
  - Station selector tabs for the three verified CAAQMS stations:
    - *Anand Vihar, New Delhi - DPCC* (`{{anand_vihar_id}}`)
    - *Indirapuram, Ghaziabad - UPPCB* (`{{indirapuram_id}}`)
    - *Sector 11, Faridabad - HSPCB* (`{{sector_11_id}}`) *(explicitly Sector 11, never Sector 16A)*
  - Category count chips row for Good, Satisfactory, Moderate, Poor, Very Poor, and Severe (`{{count_severe}}` highlighted in `#8b1c2c`).
  - Daily AQI timeline chart area with category bands (`{{good_band}}` through `{{severe_band}}`), daily markers (`{{daily_aqi}}`), and missing-data indicator (`{{missing_days_count}} days missing, not filled/interpolated`).
  - Monthly sensor QA/QC completeness audit table covering Oct through Feb (`{{hours_possible}}`, `{{valid_pm25}}`, `{{valid_pm10}}`, `{{passing_days}}`).

### 3. AQI Calculator ([aqi_calculator.html](file:///c:/Users/PRATYAKSHA/vaayu/design/stitch/aqi_calculator.html))
- **Stitch Screen ID**: `f106186c33e345a7a59e664d26cd1c22`
- **Components**:
  - Prominent self-test badge: `self-test: {{passing_tests}}/{{total_tests}} passing` with breakpoint verification indicator.
  - Dual pollutant concentration input vector for PM2.5 (`{{pm25_input}}` µg/m³) and PM10 (`{{pm10_input}}` µg/m³).
  - Piecewise linear interpolation formulation card ($I = I_{low} + \frac{I_{high}-I_{low}}{C_{high}-C_{low}}(C - C_{low})$).
  - Dynamic result card displaying calculated AQI (`{{aqi_result}}`), critical pollutant, dynamic category badge (`{{category_name}}`), health advisory statement, and sub-index trace.
  - Automated CPCB verification benchmark test matrix.

### 4. Inversion Viewer ([inversion_viewer.html](file:///c:/Users/PRATYAKSHA/vaayu/design/stitch/inversion_viewer.html))
- **Stitch Screen ID**: `184fe7ab183f41e8a6e3edef7f52e443`
- **Components**:
  - Synthetic profile selector (Surface-Based Radiation Inversion, Elevated Subsidence Inversion, Standard Uncapped / Normal Lapse Rate).
  - Temperature-vs-altitude sounding diagram with height ($z$ m AGL) on Y-axis and temperature ($T$ °C) on X-axis, featuring dry adiabatic lapse rate reference ($\Gamma_d = {{gamma_d}}$ °C/km) and shaded thermal trapping zone.
  - Inversion diagnostic readout card detailing Inversion Base (`{{base_m}} m`), Top (`{{top_m}} m`), Depth (`{{depth_m}} m`), Strength (`{{strength_c}} °C`), Type (`{{inversion_type}}`), and Trapping Severity (`{{trapping_severity}}`).
  - Discrete level radiosonde inspection table ($z_0$ through $z_4$).

---

## Strict Placeholder Enforcement
In compliance with project specifications, **no realistic or invented statistics appear in any screen**. All variable values and quantities are bound to double-curly-brace placeholders (e.g., `{{value}}`, `{{aqi}}`, `{{count}}`, `{{depth_m}}`, `{{strength_c}}`).
