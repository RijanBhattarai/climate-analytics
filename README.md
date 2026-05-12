# 🌦️ Nepal Climate & Weather Analytics (2020–2024)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightblue)
![SciPy](https://img.shields.io/badge/SciPy-1.10-green?logo=scipy)
![PowerBI](https://img.shields.io/badge/Power%20BI-Ready-yellow?logo=powerbi)
![Level](https://img.shields.io/badge/Level-Intermediate-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An **intermediate-level** climate data analytics project analysing 5 years of monthly weather data across 5 Nepali cities (2020–2024). Features statistical anomaly detection, linear regression warming trends, correlation analysis, seasonal decomposition, AQI monitoring, and a full Power BI export pipeline.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [What Makes This Intermediate](#what-makes-this-intermediate)
- [Charts Generated](#charts-generated)
- [Key Findings](#key-findings)
- [Statistical Methods Used](#statistical-methods-used)
- [Power BI Guide](#power-bi-guide)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Skills Demonstrated](#skills-demonstrated)

---

## Project Overview

This project analyses monthly climate data for 5 cities in Nepal — Kathmandu, Pokhara, Biratnagar, Chitwan, and Butwal — covering 270 observations across 5 years. The analysis moves beyond basic charts into real statistical methods used by data scientists and climate researchers.

| Metric | Value |
|--------|-------|
| Records | 270 monthly observations |
| Cities | 5 (across all 5 regions of Nepal) |
| Variables | Temperature, Rainfall, Humidity, Wind, Sunshine, AQI |
| Years | 2020 – 2024 |
| Hottest City | Biratnagar (avg 25.7°C) |
| Wettest City | Pokhara (highest annual rainfall) |
| Peak Monsoon | July (avg 388mm across cities) |
| Warming Rate | +0.39 to +0.47°C per year across all cities |

---

## What Makes This Intermediate

| Feature | Basic Project | This Project |
|---------|--------------|--------------|
| Analysis Method | GroupBy + sum/mean | Z-score anomaly detection, linear regression, Pearson correlation |
| Libraries | pandas, matplotlib | + scipy.stats, numpy signal processing |
| Time Series | Simple line chart | Rolling averages, trend decomposition, year-over-year regression |
| Statistics | Descriptive only | Inferential: r², p-values, significance testing |
| Visualisation | Single charts | Multi-panel subplots, masked heatmaps, bubble overlays |
| Insights | What happened | Why it happened + statistical confidence |

---

## Charts Generated

| # | Chart | Method | Insight |
|---|-------|--------|---------|
| 1 | Temperature Trends per City | Rolling average (3-month) | Seasonal cycles + smoothed trend |
| 2 | Annual Rainfall Comparison | Grouped bar | Year-over-year rainfall changes |
| 3 | Seasonal Climate Heatmaps | Pivot heatmap (year × month) | Temperature and rainfall patterns |
| 4 | Anomaly Detection | Z-score (threshold ±1.8) | Unusually hot/cold months flagged |
| 5 | Correlation Matrix | Pearson r, lower-triangle mask | Relationships between all variables |
| 6 | Warming Trend | Linear regression per city | Climate warming signal with slope |
| 7 | AQI Analysis | Boxplot + seasonal bar | Air quality by city and season |
| 8 | Monsoon Rainfall Intensity | Multi-line (5-yr avg) | Onset, peak, and withdrawal pattern |
| 9 | Humidity vs Rainfall | Scatter + regression line | Statistical relationship with r value |

---

## Key Findings

- 🌡️ **All 5 cities show statistically significant warming** — ranging from +0.39°C/yr (Kathmandu) to +0.47°C/yr (Biratnagar), all with p < 0.05
- 🌧️ **Monsoon peaks in July** — Pokhara receives the highest rainfall (avg ~490mm in July alone)
- 🏭 **Biratnagar has the worst AQI** (avg 180) — consistently in the "Unhealthy" zone
- 🏔️ **Pokhara has the best air quality** (avg AQI 75) — buffered by geography and wind patterns
- 📈 **Strong positive correlation** between humidity and rainfall (r > 0.85)
- ❄️ **Temperature anomalies** detected in winter months (2022–2023) — colder than normal
- ☀️ **Sunshine hours collapse during Monsoon** — dropping below 3 hrs/day in July–August

---

## Statistical Methods Used

### Z-Score Anomaly Detection
Identifies months where temperature deviates more than 1.8 standard deviations from the mean.
```python
from scipy import stats
df["z_score"] = stats.zscore(df["avg_temp_c"])
df["anomaly"] = df["z_score"].abs() > 1.8
```

### Linear Regression (Warming Trend)
Fits a trend line to annual average temperatures to detect climate warming signals.
```python
slope, intercept, r, p, se = stats.linregress(years, temps)
# slope = °C change per year
# r² = goodness of fit
# p = statistical significance
```

### Pearson Correlation Matrix
Measures strength of linear relationships between all climate variables.
```python
corr_matrix = df[climate_cols].corr()
# Values: -1 (perfect negative) to +1 (perfect positive)
```

### Rolling Average
Smooths monthly noise to reveal underlying seasonal trends.
```python
df["rolling_avg"] = df["avg_temp_c"].rolling(window=3, center=True).mean()
```

---

## Power BI Guide

Three CSVs exported to `output_data/`:

| File | Contents |
|------|----------|
| `powerbi_climate_full.csv` | All 270 records with engineered features |
| `powerbi_annual_summary.csv` | City × Year aggregated metrics |
| `powerbi_anomalies.csv` | Flagged anomaly records only |

**Recommended Power BI visuals:**
1. **Line chart** — Avg temp over time, filtered by city slicer
2. **Map** — AQI by city (colour-coded: green → red)
3. **Matrix table** — Year × Month temperature (conditional formatting)
4. **Bar chart** — Annual rainfall per city
5. **KPI Cards** — Avg AQI, Max Temp, Total Annual Rainfall
6. **Scatter** — Humidity vs Rainfall with trend line (Analytics pane)

---

## Project Structure

```
climate-analytics/
│
├── data/
│   └── nepal_climate_2020_2024.csv     270 monthly records, 13 variables
│
├── output_charts/
│   ├── 01_temperature_trends.png
│   ├── 02_annual_rainfall.png
│   ├── 03_seasonal_heatmaps.png
│   ├── 04_anomaly_detection.png
│   ├── 05_correlation_matrix.png
│   ├── 06_temperature_warming_trend.png
│   ├── 07_aqi_analysis.png
│   ├── 08_monsoon_rainfall.png
│   └── 09_humidity_vs_rainfall.png
│
├── output_data/
│   ├── powerbi_climate_full.csv
│   ├── powerbi_annual_summary.csv
│   └── powerbi_anomalies.csv
│
├── analysis.py                         Main analytics script
├── requirements.txt                    Python dependencies
└── README.md                           This file
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/RijanBhattarai/climate-analytics.git
cd climate-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis
python analysis.py
```

---

## Skills Demonstrated

- **Statistical Analysis** — Z-score anomaly detection, linear regression, Pearson correlation, p-value significance testing
- **Time Series Analysis** — Rolling averages, trend decomposition, year-over-year comparison
- **Advanced Visualisation** — Multi-panel subplots, masked heatmaps, regression overlays, dual-axis charts
- **Feature Engineering** — Season mapping, heat index, temperature range, AQI categories
- **Scientific Python** — scipy.stats for inferential statistics beyond descriptive analysis
- **Power BI Pipeline** — 3 enriched CSVs with different granularities for dashboard building
- **Environmental Analytics** — Applied domain knowledge in climate and AQI data interpretation

--
