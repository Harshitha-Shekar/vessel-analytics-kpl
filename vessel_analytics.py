"""
Vessel Timings Analytics - Kamarajar Port Limited (KPL) 2024
============================================================
Run this script to generate all charts and insights.
Output: charts/ folder with all PNG files
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FILE = "Vessel_Timings_KPL_24.xlsx"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "Coal":         "#2C3E50",
    "Liquid":       "#2980B9",
    "Multi cargo":  "#27AE60",
    "Container":    "#E67E22",
    "Automobiles":  "#8E44AD",
    "Iron Ore/Coal":"#C0392B",
}
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE, engine="openpyxl")

# Normalise month column to abbreviated names
month_map = {
    "January":"Jan","February":"Feb","March":"Mar","April":"Apr",
    "May":"May","June":"Jun","July":"Jul","August":"Aug",
    "September":"Sep","October":"Oct","November":"Nov","December":"Dec",
}
df["Month"] = df["Month"].replace(month_map)
df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)

# Helper: save figure
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔  {name}")

print("\n=== Vessel Analytics – KPL 2024 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Monthly Vessel Count
# ══════════════════════════════════════════════════════════════════════════════
monthly = df.groupby("Month", observed=True).size().reset_index(name="count")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(monthly["Month"], monthly["count"],
              color="#2980B9", edgecolor="white", width=0.6)
ax.bar_label(bars, padding=4, fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_title("Monthly Vessel Traffic – KPL 2024", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Number of Vessels", fontsize=12)
ax.set_ylim(0, monthly["count"].max() * 1.18)
ax.spines[["top","right"]].set_visible(False)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
fig.tight_layout()
save(fig, "01_monthly_vessel_count.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Cargo Type Distribution (Pie)
# ══════════════════════════════════════════════════════════════════════════════
cat_counts = df["Berth_Category"].value_counts()
colors = [PALETTE.get(c, "#BDC3C7") for c in cat_counts.index]

fig, ax = plt.subplots(figsize=(8, 7))
wedges, texts, autotexts = ax.pie(
    cat_counts, labels=None, autopct="%1.1f%%",
    colors=colors, startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=1.5),
    pctdistance=0.82
)
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight("bold"); at.set_color("white")

ax.legend(wedges, [f"{l} ({v})" for l,v in zip(cat_counts.index, cat_counts)],
          title="Cargo Type", loc="lower center", bbox_to_anchor=(0.5, -0.08),
          ncol=3, fontsize=9, title_fontsize=10)
ax.set_title("Cargo Type Distribution\n(Total Vessel Calls)", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "02_cargo_type_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Avg Berth Duration by Cargo Category
# ══════════════════════════════════════════════════════════════════════════════
avg_bd = df.groupby("Berth_Category")["Berth_Duration"].mean().sort_values()
colors_bd = [PALETTE.get(c, "#BDC3C7") for c in avg_bd.index]

fig, ax = plt.subplots(figsize=(10, 5))
hbars = ax.barh(avg_bd.index, avg_bd.values, color=colors_bd, edgecolor="white", height=0.6)
ax.bar_label(hbars, fmt="%.0f h", padding=5, fontsize=10, fontweight="bold")
ax.set_title("Average Berth Duration by Cargo Type (Hours)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Average Berth Duration (hours)", fontsize=12)
ax.spines[["top","right"]].set_visible(False)
ax.set_xlim(0, avg_bd.max() * 1.18)
fig.tight_layout()
save(fig, "03_avg_berth_duration_category.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Top Berths by Average Berth Duration
# ══════════════════════════════════════════════════════════════════════════════
berth_stats = df.groupby("Berth")["Berth_Duration"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
hbars = ax.barh(berth_stats.index[::-1], berth_stats.values[::-1],
                color="#2C3E50", edgecolor="white", height=0.6)
ax.bar_label(hbars, fmt="%.0f h", padding=5, fontsize=9, fontweight="bold")
ax.set_title("Average Berth Duration by Terminal (Hours)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Average Berth Duration (hours)", fontsize=12)
ax.spines[["top","right"]].set_visible(False)
ax.set_xlim(0, berth_stats.max() * 1.2)
fig.tight_layout()
save(fig, "04_avg_berth_duration_terminal.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Seasonal Analysis – Vessel Count & Avg Berth Duration
# ══════════════════════════════════════════════════════════════════════════════
season_order = ["Monsoon", "Post-Monsoon", "Winter"]
season_counts = df.groupby("Season").size().reindex(season_order)
season_avg_bd = df.groupby("Season")["Berth_Duration"].mean().reindex(season_order)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sc = ["#2980B9","#27AE60","#E67E22"]
bars1 = ax1.bar(season_order, season_counts, color=sc, edgecolor="white", width=0.55)
ax1.bar_label(bars1, padding=4, fontsize=11, fontweight="bold")
ax1.set_title("Vessel Count by Season", fontsize=13, fontweight="bold")
ax1.set_ylabel("Number of Vessels"); ax1.spines[["top","right"]].set_visible(False)
ax1.set_ylim(0, season_counts.max() * 1.2)

bars2 = ax2.bar(season_order, season_avg_bd, color=sc, edgecolor="white", width=0.55)
ax2.bar_label(bars2, fmt="%.0f h", padding=4, fontsize=11, fontweight="bold")
ax2.set_title("Avg Berth Duration by Season (Hours)", fontsize=13, fontweight="bold")
ax2.set_ylabel("Average Berth Duration (hours)"); ax2.spines[["top","right"]].set_visible(False)
ax2.set_ylim(0, season_avg_bd.max() * 1.2)

fig.suptitle("Seasonal Port Activity Analysis", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "05_seasonal_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Hour-of-Arrival Heatmap Pattern
# ══════════════════════════════════════════════════════════════════════════════
hoa_counts = df.groupby("Hour_of_Arrival").size().reindex(range(24), fill_value=0)

fig, ax = plt.subplots(figsize=(13, 4))
bar_colors = ["#E74C3C" if v == hoa_counts.max() else
              "#F39C12" if v >= hoa_counts.quantile(0.75) else
              "#2980B9" for v in hoa_counts]
bars = ax.bar(hoa_counts.index, hoa_counts.values, color=bar_colors, edgecolor="white", width=0.8)
ax.bar_label(bars, padding=2, fontsize=8, rotation=90, color="#2C3E50")
ax.set_title("Vessel Arrival Distribution by Hour of Day", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Hour of Arrival (0–23)", fontsize=12)
ax.set_ylabel("Number of Vessels", fontsize=12)
ax.set_xticks(range(24))
ax.spines[["top","right"]].set_visible(False)

legend_handles = [
    mpatches.Patch(color="#E74C3C", label="Peak hour"),
    mpatches.Patch(color="#F39C12", label="Top quartile"),
    mpatches.Patch(color="#2980B9", label="Normal"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
fig.tight_layout()
save(fig, "06_hour_of_arrival.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Day-of-Arrival Pattern
# ══════════════════════════════════════════════════════════════════════════════
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_counts = df["Day_of_Arrival"].value_counts().reindex(day_order, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 5))
day_colors = ["#E74C3C" if d in ["Saturday","Sunday"] else "#2980B9" for d in day_order]
bars = ax.bar(day_counts.index, day_counts.values, color=day_colors, edgecolor="white", width=0.6)
ax.bar_label(bars, padding=4, fontsize=11, fontweight="bold")
ax.set_title("Vessel Arrivals by Day of Week", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Day of Week", fontsize=12)
ax.set_ylabel("Number of Vessels", fontsize=12)
ax.set_ylim(0, day_counts.max() * 1.18)
ax.spines[["top","right"]].set_visible(False)
legend_handles = [
    mpatches.Patch(color="#2980B9", label="Weekday"),
    mpatches.Patch(color="#E74C3C", label="Weekend"),
]
ax.legend(handles=legend_handles, fontsize=10)
fig.tight_layout()
save(fig, "07_day_of_arrival.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  Anchorage vs Berth Duration Scatter
# ══════════════════════════════════════════════════════════════════════════════
anc = df.dropna(subset=["Anchorage_Duration"]).copy()
anc["Berth_Duration_h"] = anc["Berth_Duration"]
anc["Anchorage_Duration_h"] = anc["Anchorage_Duration"]

fig, ax = plt.subplots(figsize=(10, 7))
for cat, grp in anc.groupby("Berth_Category"):
    ax.scatter(grp["Anchorage_Duration_h"], grp["Berth_Duration_h"],
               label=cat, alpha=0.7, s=60,
               color=PALETTE.get(cat, "#BDC3C7"), edgecolors="white", linewidths=0.5)

ax.set_title("Anchorage Duration vs Berth Duration", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Anchorage Duration (hours)", fontsize=12)
ax.set_ylabel("Berth Duration (hours)", fontsize=12)
ax.legend(title="Cargo Type", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "08_anchorage_vs_berth_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  Port Duration Distribution (Box Plot by Category)
# ══════════════════════════════════════════════════════════════════════════════
categories = df["Berth_Category"].unique()
data_by_cat = [df[df["Berth_Category"]==c]["Port_Duration"].dropna().values for c in categories]

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(data_by_cat, labels=categories, patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                flierprops=dict(marker="o", markersize=4, alpha=0.4))
for patch, cat in zip(bp["boxes"], categories):
    patch.set_facecolor(PALETTE.get(cat, "#BDC3C7"))
    patch.set_alpha(0.85)

ax.set_title("Port Duration Distribution by Cargo Type", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Cargo Type", fontsize=12)
ax.set_ylabel("Port Duration (hours)", fontsize=12)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "09_port_duration_boxplot.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Monthly Berth Duration (Stacked by Category)
# ══════════════════════════════════════════════════════════════════════════════
pivot = df.groupby(["Month","Berth_Category"], observed=True)["Berth_Duration"].sum().unstack(fill_value=0)
pivot = pivot.reindex([m for m in MONTH_ORDER if m in pivot.index])

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(pivot))
for cat in pivot.columns:
    color = PALETTE.get(cat, "#BDC3C7")
    ax.bar(pivot.index, pivot[cat], bottom=bottom, label=cat,
           color=color, edgecolor="white", linewidth=0.5)
    bottom += pivot[cat].values

ax.set_title("Total Berth Duration by Month & Cargo Type (Hours)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Total Berth Duration (hours)", fontsize=12)
ax.legend(title="Cargo Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "10_monthly_berth_duration_stacked.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. KPI Summary Card
# ══════════════════════════════════════════════════════════════════════════════
total_vessels = len(df)
avg_berth = df["Berth_Duration"].mean()
avg_port  = df["Port_Duration"].mean()
avg_anc   = df["Anchorage_Duration"].dropna().mean()
pct_anc   = (df["Anchorage_Duration"].notna().sum() / total_vessels) * 100
peak_month = df.groupby("Month", observed=True).size().idxmax()
peak_hour  = df["Hour_of_Arrival"].value_counts().idxmax()
top_berth  = df["Berth"].value_counts().idxmax()

kpis = [
    ("Total Vessel Calls",          f"{total_vessels}"),
    ("Avg Berth Duration",          f"{avg_berth:.1f} hrs"),
    ("Avg Port Duration",           f"{avg_port:.1f} hrs"),
    ("Avg Anchorage Duration",      f"{avg_anc:.1f} hrs"),
    ("Vessels w/ Anchorage",        f"{pct_anc:.0f}%"),
    ("Busiest Month",               str(peak_month)),
    ("Peak Arrival Hour",           f"{peak_hour}:00"),
    ("Busiest Terminal",            top_berth.split("(")[0].strip()),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
axes = axes.flatten()
tile_colors = ["#2C3E50","#2980B9","#27AE60","#E67E22",
               "#8E44AD","#C0392B","#16A085","#D35400"]

for ax, (label, value), color in zip(axes, kpis, tile_colors):
    ax.set_facecolor(color)
    fig.patch.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
    ax.text(0.5, 0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax.text(0.5, 0.25, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=9.5, color="#ECF0F1",
            wrap=True, multialignment="center")

fig.suptitle("KPL 2024 – Key Performance Indicators", fontsize=16, fontweight="bold", y=1.01)
plt.subplots_adjust(wspace=0.08, hspace=0.08)
save(fig, "00_kpi_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# Print text summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── KEY INSIGHTS ──────────────────────────────────────")
print(f"  Total vessel calls recorded   : {total_vessels}")
print(f"  Date range                    : Jul – Dec 2024")
print(f"  Avg berth duration            : {avg_berth:.1f} hrs")
print(f"  Avg port stay                 : {avg_port:.1f} hrs")
print(f"  Vessels requiring anchorage   : {pct_anc:.0f}% ({df['Anchorage_Duration'].notna().sum()} vessels)")
print(f"  Peak traffic month            : {peak_month} ({df.groupby('Month', observed=True).size().max()} vessels)")
print(f"  Peak arrival hour             : {peak_hour}:00")
print(f"  Dominant cargo type           : Coal ({(df['Berth_Category']=='Coal').sum()} calls, {(df['Berth_Category']=='Coal').sum()/total_vessels*100:.1f}%)")
print(f"  Busiest terminal              : {top_berth}")
print("────────────────────────────────────────────────────────")
print(f"\n  All charts saved to: ./{OUTPUT_DIR}/\n")
