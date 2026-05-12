"""
================================================================
  Nepal Climate & Weather Analytics (2020–2024)
  Author   : Rijan
  Level    : Intermediate
  Tools    : Python, Pandas, Matplotlib, Seaborn, SciPy, NumPy
  Features : Trend analysis, anomaly detection, seasonal
             decomposition, correlation matrix, rolling averages,
             AQI analysis, and Power BI export pipeline
================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import warnings, os

warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────
os.makedirs("output_charts", exist_ok=True)
os.makedirs("output_data",   exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

NAVY   = "#1F497D"; GREEN  = "#2A9D8F"; RED    = "#C73E1D"
AMBER  = "#F4A261"; PURPLE = "#6A0572"; TEAL   = "#2E86AB"
CITY_COLORS = {
    "Kathmandu": NAVY, "Pokhara": GREEN,
    "Biratnagar": RED, "Chitwan": AMBER, "Butwal": PURPLE
}

# ── Load & Engineer ────────────────────────────────────────────
df = pd.read_csv("data/nepal_climate_2020_2024.csv", parse_dates=["date"])
df["year"]         = df["date"].dt.year
df["month_num"]    = df["date"].dt.month
df["month_name"]   = df["date"].dt.strftime("%b")
df["year_month"]   = df["date"].dt.to_period("M").astype(str)
df["season"]       = df["month_num"].map({
    12:"Winter",1:"Winter",2:"Winter",
    3:"Spring",4:"Spring",5:"Spring",
    6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
    10:"Autumn",11:"Autumn"
})
df["temp_range"]   = df["max_temp_c"] - df["min_temp_c"]
df["heat_index"]   = df["avg_temp_c"] + 0.33 * (df["humidity_pct"] * 0.1) - 4.0

# ── Summary ────────────────────────────────────────────────────
print("=" * 65)
print("     NEPAL CLIMATE ANALYTICS (2020–2024) — REPORT")
print("=" * 65)
print(f"  Records          : {len(df)} monthly observations")
print(f"  Cities           : {df['city'].nunique()} ({', '.join(df['city'].unique())})")
print(f"  Years Covered    : 2020 – 2024 (5 years)")
print(f"  Hottest City     : {df.groupby('city')['avg_temp_c'].mean().idxmax()}")
print(f"  Wettest City     : {df.groupby('city')['rainfall_mm'].mean().idxmax()}")
print(f"  Worst AQI City   : {df.groupby('city')['aqi'].mean().idxmax()}")
print(f"  Peak Monsoon     : July ({df[df['month_num']==7]['rainfall_mm'].mean():.0f} mm avg)")
print("=" * 65)


# ════════════════════════════════════════════════════════════════
# CHART 1 — Temperature Trend per City with Rolling Average
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle("Monthly Average Temperature Trends (2020–2024)\nwith 3-Month Rolling Average",
             fontsize=15, fontweight="bold", y=1.01)

for ax, (city, grp) in zip(axes, df.groupby("city")):
    grp = grp.sort_values("date")
    ax.plot(grp["date"], grp["avg_temp_c"], alpha=0.4,
            color=CITY_COLORS[city], linewidth=1.2, label="Monthly")
    # 3-month rolling average
    rolling = grp["avg_temp_c"].rolling(window=3, center=True).mean()
    ax.plot(grp["date"], rolling, color=CITY_COLORS[city],
            linewidth=2.5, label="3-mo rolling avg")
    ax.fill_between(grp["date"], grp["min_temp_c"], grp["max_temp_c"],
                    alpha=0.1, color=CITY_COLORS[city], label="Min-Max range")
    ax.set_ylabel("°C", fontsize=9)
    ax.set_title(city, fontsize=11, fontweight="bold", color=CITY_COLORS[city])
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}°C"))

axes[-1].set_xlabel("Date", fontsize=11)
plt.tight_layout()
plt.savefig("output_charts/01_temperature_trends.png", bbox_inches="tight")
plt.close()
print("  ✔ Chart 1: Temperature Trends with Rolling Average")


# ════════════════════════════════════════════════════════════════
# CHART 2 — Annual Rainfall Comparison (All Cities, Grouped Bar)
# ════════════════════════════════════════════════════════════════
annual_rain = df.groupby(["year","city"])["rainfall_mm"].sum().unstack()
x     = np.arange(len(annual_rain))
width = 0.15

fig, ax = plt.subplots(figsize=(13, 6))
for i, city in enumerate(annual_rain.columns):
    bars = ax.bar(x + i*width, annual_rain[city], width,
                  label=city, color=CITY_COLORS[city], edgecolor="white", alpha=0.9)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(annual_rain.index, fontsize=11)
ax.set_title("Annual Total Rainfall by City (2020–2024)", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Total Rainfall (mm)", fontsize=11)
ax.legend(title="City", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f} mm"))
plt.tight_layout()
plt.savefig("output_charts/02_annual_rainfall.png")
plt.close()
print("  ✔ Chart 2: Annual Rainfall Comparison")


# ════════════════════════════════════════════════════════════════
# CHART 3 — Seasonal Climate Profile (Radar / Heatmap)
# ════════════════════════════════════════════════════════════════
seasonal = df.groupby(["season","month_num"]).agg(
    avg_temp=("avg_temp_c","mean"),
    rainfall=("rainfall_mm","mean"),
    humidity=("humidity_pct","mean"),
    sunshine=("sunshine_hours","mean")
).reset_index()

# Month-order heatmap of avg temp for Kathmandu (illustrative)
ktm = df[df["city"]=="Kathmandu"].copy()
pivot_temp = ktm.pivot_table(index="year", columns="month_num", values="avg_temp_c")
pivot_temp.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Temperature heatmap
sns.heatmap(pivot_temp, annot=True, fmt=".1f", cmap="RdYlBu_r",
            ax=axes[0], linewidths=0.5,
            cbar_kws={"label":"Avg Temp (°C)"},
            annot_kws={"size":9})
axes[0].set_title("Kathmandu — Monthly Avg Temp Heatmap (°C)\n(Year × Month)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Month"); axes[0].set_ylabel("Year")

# Rainfall heatmap
pivot_rain = ktm.pivot_table(index="year", columns="month_num", values="rainfall_mm")
pivot_rain.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
sns.heatmap(pivot_rain, annot=True, fmt=".0f", cmap="Blues",
            ax=axes[1], linewidths=0.5,
            cbar_kws={"label":"Rainfall (mm)"},
            annot_kws={"size":9})
axes[1].set_title("Kathmandu — Monthly Rainfall Heatmap (mm)\n(Year × Month)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("")

plt.suptitle("Seasonal Climate Heatmaps — Kathmandu", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("output_charts/03_seasonal_heatmaps.png", bbox_inches="tight")
plt.close()
print("  ✔ Chart 3: Seasonal Climate Heatmaps")


# ════════════════════════════════════════════════════════════════
# CHART 4 — Temperature Anomaly Detection (Z-Score)
# ════════════════════════════════════════════════════════════════
ktm_ts = df[df["city"]=="Kathmandu"].sort_values("date").copy()
ktm_ts["z_score"]   = stats.zscore(ktm_ts["avg_temp_c"])
ktm_ts["anomaly"]   = ktm_ts["z_score"].abs() > 1.8
ktm_ts["anomaly_type"] = ktm_ts.apply(
    lambda r: "Hot Anomaly" if r["z_score"] > 1.8 else ("Cold Anomaly" if r["z_score"] < -1.8 else "Normal"), axis=1
)

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Top: temperature line with anomalies highlighted
axes[0].plot(ktm_ts["date"], ktm_ts["avg_temp_c"], color=NAVY,
             linewidth=1.8, alpha=0.8, label="Avg Temp")
hot   = ktm_ts[ktm_ts["anomaly_type"] == "Hot Anomaly"]
cold  = ktm_ts[ktm_ts["anomaly_type"] == "Cold Anomaly"]
axes[0].scatter(hot["date"],  hot["avg_temp_c"],  color=RED,   s=90, zorder=5, label="Hot Anomaly",  marker="^")
axes[0].scatter(cold["date"], cold["avg_temp_c"], color=TEAL,  s=90, zorder=5, label="Cold Anomaly", marker="v")
axes[0].set_ylabel("Avg Temp (°C)", fontsize=11)
axes[0].set_title("Kathmandu Temperature Anomaly Detection (Z-Score > ±1.8)", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}°C"))

# Bottom: Z-score bar chart
bar_colors = [RED if z > 1.8 else (TEAL if z < -1.8 else NAVY) for z in ktm_ts["z_score"]]
axes[1].bar(ktm_ts["date"], ktm_ts["z_score"], color=bar_colors,
            width=20, alpha=0.85)
axes[1].axhline(1.8,  color=RED,  linestyle="--", linewidth=1.2, alpha=0.7, label="+1.8 threshold")
axes[1].axhline(-1.8, color=TEAL, linestyle="--", linewidth=1.2, alpha=0.7, label="-1.8 threshold")
axes[1].axhline(0,    color="grey", linewidth=0.8, alpha=0.5)
axes[1].set_ylabel("Z-Score", fontsize=11)
axes[1].set_xlabel("Date", fontsize=11)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("output_charts/04_anomaly_detection.png")
plt.close()
print("  ✔ Chart 4: Temperature Anomaly Detection")


# ════════════════════════════════════════════════════════════════
# CHART 5 — Correlation Matrix (Climate Variables)
# ════════════════════════════════════════════════════════════════
corr_cols = ["avg_temp_c","max_temp_c","min_temp_c","rainfall_mm",
             "humidity_pct","wind_speed_kmh","sunshine_hours","aqi","temp_range"]
corr_matrix = df[corr_cols].corr().round(2)

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, square=True,
            cbar_kws={"label":"Pearson r", "shrink": 0.8},
            annot_kws={"size": 9})
ax.set_title("Climate Variables — Correlation Matrix\n(Lower Triangle, All Cities Combined)",
             fontsize=13, fontweight="bold", pad=15)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("output_charts/05_correlation_matrix.png")
plt.close()
print("  ✔ Chart 5: Correlation Matrix")


# ════════════════════════════════════════════════════════════════
# CHART 6 — Year-over-Year Temperature Change (Linear Regression)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

for city, grp in df.groupby("city"):
    annual_avg = grp.groupby("year")["avg_temp_c"].mean().reset_index()
    # Linear regression
    slope, intercept, r, p, se = stats.linregress(annual_avg["year"], annual_avg["avg_temp_c"])
    x_line  = np.array(annual_avg["year"])
    y_line  = slope * x_line + intercept
    trend   = f"(+{slope:.2f}°C/yr)" if slope > 0 else f"({slope:.2f}°C/yr)"

    ax.scatter(annual_avg["year"], annual_avg["avg_temp_c"],
               color=CITY_COLORS[city], s=80, zorder=4)
    ax.plot(annual_avg["year"], annual_avg["avg_temp_c"],
            color=CITY_COLORS[city], linewidth=1.5, alpha=0.5)
    ax.plot(x_line, y_line, color=CITY_COLORS[city],
            linestyle="--", linewidth=2.0,
            label=f"{city} {trend}")

ax.set_title("Annual Average Temperature Trend per City\n(Linear Regression — Climate Warming Signal)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Annual Avg Temperature (°C)", fontsize=11)
ax.legend(fontsize=9, title="City (slope)", title_fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}°C"))
ax.set_xticks(df["year"].unique())
plt.tight_layout()
plt.savefig("output_charts/06_temperature_warming_trend.png")
plt.close()
print("  ✔ Chart 6: Year-over-Year Temperature Warming Trend")


# ════════════════════════════════════════════════════════════════
# CHART 7 — AQI Analysis by City and Season
# ════════════════════════════════════════════════════════════════
def aqi_category(v):
    if v <= 50:   return "Good"
    elif v <= 100: return "Moderate"
    elif v <= 150: return "Unhealthy for Sensitive"
    elif v <= 200: return "Unhealthy"
    else:          return "Very Unhealthy"

df["aqi_category"] = df["aqi"].apply(aqi_category)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot: AQI per city
city_order = df.groupby("city")["aqi"].median().sort_values(ascending=False).index
city_palette = {c: CITY_COLORS[c] for c in city_order}
sns.boxplot(data=df, x="city", y="aqi", order=city_order,
            palette=city_palette, ax=axes[0], width=0.6)
axes[0].axhline(100, color=AMBER, linestyle="--", linewidth=1.5, label="Moderate threshold (100)")
axes[0].axhline(150, color=RED,   linestyle="--", linewidth=1.5, label="Unhealthy threshold (150)")
axes[0].set_title("AQI Distribution by City", fontsize=12, fontweight="bold")
axes[0].set_xlabel("City"); axes[0].set_ylabel("AQI")
axes[0].legend(fontsize=8)

# Bar: Average AQI by season
season_order = ["Winter","Spring","Monsoon","Autumn"]
season_aqi   = df.groupby("season")["aqi"].mean().reindex(season_order)
season_colors= [TEAL, GREEN, NAVY, AMBER]
bars = axes[1].bar(season_aqi.index, season_aqi.values,
                   color=season_colors, edgecolor="white")
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.0f}", ha="center", fontsize=10, fontweight="bold")
axes[1].set_title("Average AQI by Season", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Season"); axes[1].set_ylabel("Average AQI")

plt.suptitle("Air Quality Index (AQI) Analysis", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("output_charts/07_aqi_analysis.png", bbox_inches="tight")
plt.close()
print("  ✔ Chart 7: AQI Analysis")


# ════════════════════════════════════════════════════════════════
# CHART 8 — Monsoon Onset & Rainfall Intensity (Multi-line)
# ════════════════════════════════════════════════════════════════
monsoon_months = df[df["month_num"].isin([5,6,7,8,9])].copy()
monsoon_city   = monsoon_months.groupby(["city","month_num"])["rainfall_mm"].mean().reset_index()
month_labels   = {5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep"}
monsoon_city["month_label"] = monsoon_city["month_num"].map(month_labels)

fig, ax = plt.subplots(figsize=(12, 6))
for city, grp in monsoon_city.groupby("city"):
    grp = grp.sort_values("month_num")
    ax.plot(grp["month_label"], grp["rainfall_mm"],
            marker="o", linewidth=2.5, markersize=9,
            color=CITY_COLORS[city], label=city)
    ax.annotate(f"{grp['rainfall_mm'].max():.0f}mm",
                (grp.loc[grp["rainfall_mm"].idxmax(), "month_label"],
                 grp["rainfall_mm"].max()),
                fontsize=8, xytext=(4, 6), textcoords="offset points",
                color=CITY_COLORS[city])

ax.set_title("Monsoon Season Rainfall Intensity by City\n(5-Year Average, May–Sep)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Average Rainfall (mm)", fontsize=11)
ax.legend(title="City", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f} mm"))
plt.tight_layout()
plt.savefig("output_charts/08_monsoon_rainfall.png")
plt.close()
print("  ✔ Chart 8: Monsoon Rainfall Intensity")


# ════════════════════════════════════════════════════════════════
# CHART 9 — Humidity vs Rainfall Scatter (with regression)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))
for city, grp in df.groupby("city"):
    ax.scatter(grp["humidity_pct"], grp["rainfall_mm"],
               color=CITY_COLORS[city], alpha=0.65, s=60, label=city)

# Overall regression line
slope, intercept, r, p, _ = stats.linregress(df["humidity_pct"], df["rainfall_mm"])
x_r = np.linspace(df["humidity_pct"].min(), df["humidity_pct"].max(), 100)
ax.plot(x_r, slope*x_r + intercept, color="black",
        linestyle="--", linewidth=2, alpha=0.7,
        label=f"Regression (r={r:.2f}, p={'<0.001' if p<0.001 else f'{p:.3f}'})")

ax.set_title("Humidity vs Rainfall — All Cities (2020–2024)\nwith Linear Regression",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Humidity (%)", fontsize=11)
ax.set_ylabel("Rainfall (mm)", fontsize=11)
ax.legend(title="City", fontsize=9)
plt.tight_layout()
plt.savefig("output_charts/09_humidity_vs_rainfall.png")
plt.close()
print("  ✔ Chart 9: Humidity vs Rainfall Regression")


# ════════════════════════════════════════════════════════════════
# POWER BI EXPORTS
# ════════════════════════════════════════════════════════════════
# 1 — Full enriched dataset
df.to_csv("output_data/powerbi_climate_full.csv", index=False)

# 2 — Annual city summary
annual_summary = df.groupby(["year","city","region"]).agg(
    avg_temp     =("avg_temp_c",    "mean"),
    max_temp     =("max_temp_c",    "max"),
    min_temp     =("min_temp_c",    "min"),
    total_rain   =("rainfall_mm",   "sum"),
    avg_humidity =("humidity_pct",  "mean"),
    avg_aqi      =("aqi",           "mean"),
    avg_sunshine =("sunshine_hours","mean"),
    avg_wind     =("wind_speed_kmh","mean")
).round(2).reset_index()
annual_summary.to_csv("output_data/powerbi_annual_summary.csv", index=False)

# 3 — Anomaly records only
anomalies = df[df["city"]=="Kathmandu"].copy()
anomalies["z_score"] = stats.zscore(anomalies["avg_temp_c"])
anomalies["is_anomaly"] = anomalies["z_score"].abs() > 1.8
anomalies[anomalies["is_anomaly"]].to_csv("output_data/powerbi_anomalies.csv", index=False)

print("\n  ✔ Power BI: powerbi_climate_full.csv")
print("  ✔ Power BI: powerbi_annual_summary.csv")
print("  ✔ Power BI: powerbi_anomalies.csv")


# ════════════════════════════════════════════════════════════════
# PRINTED ANALYTICAL REPORT
# ════════════════════════════════════════════════════════════════
print()
print("── Annual Avg Temperature by City ───────────────────────")
print(df.groupby(["city","year"])["avg_temp_c"].mean().round(1).unstack().to_string())

print()
print("── Warming Trend (Linear Regression Slope °C/year) ──────")
for city, grp in df.groupby("city"):
    annual = grp.groupby("year")["avg_temp_c"].mean()
    slope, _, r, p, _ = stats.linregress(annual.index, annual.values)
    sig = "***" if p < 0.05 else "n.s."
    print(f"  {city:<15}: {slope:+.3f} °C/yr  (r²={r**2:.3f}) {sig}")

print()
print("── Average AQI by City ───────────────────────────────────")
print(df.groupby("city")["aqi"].mean().round(1).sort_values(ascending=False).to_string())

print()
print(f"  All charts → output_charts/")
print(f"  Power BI CSVs → output_data/")
print("=" * 65)
