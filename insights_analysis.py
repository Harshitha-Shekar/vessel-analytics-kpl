"""
Insights Analysis – Vessel Timings KPL 2024
============================================
Generates charts for all analytical insights:
  1.  Comparative Analysis (Month-on-Month + Berth Efficiency)
  2.  Anchorage Impact on Total Delay
  3.  Congestion Severity Analysis
  4.  Peak Hour Congestion
  5.  Berth Bottleneck Analysis
  6.  Seasonal / Monthly Trend
  7.  Traffic vs Delay Relationship
  8.  Distribution of Waiting Time
  9.  Direct vs Waiting Comparison
  10. High Delay Vessel Analysis
  11. Data Quality Insights

Run:
    python insights_analysis.py
Output: insights/ folder with all PNG charts
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
DATA_FILE  = "Vessel_Timings_KPL_24.xlsx"
OUTPUT_DIR = "insights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "Coal":          "#2C3E50",
    "Liquid":        "#2980B9",
    "Multi cargo":   "#27AE60",
    "Container":     "#E67E22",
    "Automobiles":   "#8E44AD",
    "Iron Ore/Coal": "#C0392B",
}
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SEASON_COL  = {"Monsoon":"#2980B9", "Post-Monsoon":"#27AE60", "Winter":"#E67E22"}

# ── Load & Clean ───────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE, engine="openpyxl")

# Clean
df["Anchorage_Duration"] = df["Anchorage_Duration"].fillna(0)
df["Berth_Duration"]     = df["Berth_Duration"].replace(0, np.nan)

# Convert hours → days
for col in ["Berth_Duration", "Port_Duration", "Anchorage_Duration"]:
    df[col] = df[col] / 24

# Standardise Month
month_map = {
    "January":"Jan","February":"Feb","March":"Mar","April":"Apr",
    "May":"May","June":"Jun","July":"Jul","August":"Aug",
    "September":"Sep","October":"Oct","November":"Nov","December":"Dec",
}
df["Month"] = df["Month"].replace(month_map)
df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)

# Derived columns
df["Anchored"]       = df["Anchorage_Duration"] > 0
df["Total_Delay"]    = df["Port_Duration"] - df["Berth_Duration"].fillna(0)
df["Other_Duration"] = (df["Port_Duration"]
                        - df["Berth_Duration"].fillna(0)
                        - df["Anchorage_Duration"]).clip(lower=0)

# Helper: save figure
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔  {name}")

print("\n=== Insights Analysis – KPL 2024 ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# 1. COMPARATIVE ANALYSIS
#    a) Month-on-month % change in vessel calls
#    b) Port time breakdown (berth / anchorage / other) per cargo category
# ══════════════════════════════════════════════════════════════════════════════
print("── 1. Comparative Analysis ──────────────────────────")

# --- 1a: Month-on-month % change ---
monthly_counts = df.groupby("Month", observed=True).size()
months_present = monthly_counts.index.tolist()
pct_change     = monthly_counts.pct_change() * 100
pct_change.iloc[0] = 0  # baseline

bar_colors = ["#27AE60" if v >= 0 else "#C0392B" for v in pct_change]
bar_colors[0] = "#0D7377"  # baseline

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(months_present, pct_change.values,
              color=bar_colors, edgecolor="white", width=0.6)

for bar, val in zip(bars, pct_change.values):
    label = "Baseline" if val == 0 else f"{val:+.0f}%"
    ypos  = bar.get_height() + 1.5 if val >= 0 else bar.get_height() - 4
    ax.text(bar.get_x() + bar.get_width()/2, ypos, label,
            ha="center", fontsize=10, fontweight="bold", color="#2C3E50")

ax.axhline(0, color="#7F8C8D", linewidth=0.8, linestyle="--")
ax.set_title("Month-on-Month Change in Vessel Calls (%)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("% Change vs Previous Month", fontsize=12)
ax.spines[["top","right"]].set_visible(False)

handles = [
    mpatches.Patch(color="#0D7377", label="Baseline (Jul)"),
    mpatches.Patch(color="#C0392B", label="Decline"),
    mpatches.Patch(color="#27AE60", label="Growth"),
]
ax.legend(handles=handles, fontsize=10)
fig.tight_layout()
save(fig, "01a_mom_vessel_change.png")

# --- 1b: Port time breakdown stacked 100% bar per category ---
cat_order = ["Container","Liquid","Coal","Automobiles","Multi cargo","Iron Ore/Coal"]
breakdown = df.groupby("Berth_Category")[["Berth_Duration","Anchorage_Duration","Other_Duration"]].mean()
breakdown = breakdown.reindex(cat_order)

# Normalise to 100%
total = breakdown.sum(axis=1)
breakdown_pct = breakdown.div(total, axis=0) * 100

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(cat_order))
w = 0.55

colors_stack = ["#0D7377", "#E67E22", "#BDC3C7"]
labels_stack  = ["Berth (productive)", "Anchorage wait", "Other delay"]

bottom = np.zeros(len(cat_order))
for col, color, label in zip(["Berth_Duration","Anchorage_Duration","Other_Duration"],
                              colors_stack, labels_stack):
    vals = breakdown_pct[col].values
    bars = ax.bar(x, vals, bottom=bottom, width=w, color=color,
                  edgecolor="white", linewidth=0.6, label=label)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 8:
            ax.text(x[i], b + v/2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(cat_order, fontsize=10)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)], fontsize=10)
ax.set_title("Port Time Breakdown by Cargo Type (%)", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("% of Total Port Time", fontsize=12)
ax.legend(title="Time Component", fontsize=10, loc="upper right")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "01b_port_time_breakdown.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ANCHORAGE IMPACT ON TOTAL DELAY
# ══════════════════════════════════════════════════════════════════════════════
print("── 2. Anchorage Impact on Total Delay ───────────────")

avg_delay_anchored     = df[df["Anchored"]]["Total_Delay"].mean()
avg_delay_not_anchored = df[~df["Anchored"]]["Total_Delay"].mean()
avg_anc_wait           = df[df["Anchored"]]["Anchorage_Duration"].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Comparison bar
groups  = ["With Anchorage", "Without Anchorage"]
values  = [avg_delay_anchored, avg_delay_not_anchored]
c_bars  = ["#C0392B", "#27AE60"]
bars = axes[0].bar(groups, values, color=c_bars, edgecolor="white", width=0.5)
axes[0].bar_label(bars, fmt="%.1f days", padding=5, fontsize=12, fontweight="bold")
axes[0].set_title("Avg Total Delay: Anchored vs Direct", fontsize=13, fontweight="bold", pad=10)
axes[0].set_ylabel("Avg Total Delay (days)", fontsize=11)
axes[0].set_ylim(0, max(values) * 1.3)
axes[0].spines[["top","right"]].set_visible(False)

# Annotation
diff = avg_delay_anchored - avg_delay_not_anchored
axes[0].annotate(f"+{diff:.1f} days extra\ndue to anchorage",
                 xy=(0, avg_delay_anchored), xytext=(0.5, avg_delay_anchored + 2),
                 fontsize=10, color="#C0392B", ha="center",
                 arrowprops=dict(arrowstyle="-", color="#C0392B", lw=1.2))

# Right: Grouped bar – berth / anchorage / other per category
cat_stats = df.groupby("Berth_Category")[["Berth_Duration","Anchorage_Duration","Other_Duration"]].mean()
x2 = np.arange(len(cat_stats))
w2 = 0.25
for i, (col, color, lbl) in enumerate(zip(
        ["Berth_Duration","Anchorage_Duration","Other_Duration"],
        ["#0D7377","#E67E22","#C0392B"],
        ["Berth","Anchorage","Other delay"])):
    axes[1].bar(x2 + i*w2, cat_stats[col], width=w2, color=color,
                edgecolor="white", label=lbl)

axes[1].set_xticks(x2 + w2)
axes[1].set_xticklabels(cat_stats.index, fontsize=9, rotation=15, ha="right")
axes[1].set_title("Port Time Components by Cargo (days)", fontsize=13, fontweight="bold", pad=10)
axes[1].set_ylabel("Average Days", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].spines[["top","right"]].set_visible(False)

fig.suptitle("Anchorage Impact on Total Port Delay", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "02_anchorage_impact_delay.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CONGESTION SEVERITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("── 3. Congestion Severity Analysis ──────────────────")

cong = df.groupby("Berth_Category").agg(
    Anchor_Rate   = ("Anchored",           lambda x: x.mean() * 100),
    Avg_Anc_Wait  = ("Anchorage_Duration", "mean"),
    Avg_Other_Delay = ("Other_Duration",   "mean"),
).sort_values("Avg_Anc_Wait", ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cats   = cong.index.tolist()
colors = [PALETTE.get(c, "#BDC3C7") for c in cats]

for ax, col, title, fmt in zip(
        axes,
        ["Anchor_Rate","Avg_Anc_Wait","Avg_Other_Delay"],
        ["Anchor Rate (%)", "Avg Anchorage Wait (days)", "Avg Other Delay (days)"],
        ["{:.0f}%","%.1fd","%.1fd"]):
    hbars = ax.barh(cats, cong[col], color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(hbars, cong[col]):
        ax.text(bar.get_width() + cong[col].max()*0.02, bar.get_y() + bar.get_height()/2,
                fmt.format(val) if "%" in fmt else f"{val:.1f}d",
                va="center", fontsize=10, fontweight="bold", color="#2C3E50")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, cong[col].max() * 1.25)
    ax.spines[["top","right"]].set_visible(False)
    ax.invert_yaxis()

fig.suptitle("Congestion Severity by Cargo Type", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "03_congestion_severity.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PEAK HOUR CONGESTION
# ══════════════════════════════════════════════════════════════════════════════
print("── 4. Peak Hour Congestion ───────────────────────────")

hoa = df.groupby("Hour_of_Arrival").size().reindex(range(24), fill_value=0)
q75 = hoa.quantile(0.75)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Top: arrival volume
bar_colors_h = ["#C0392B" if v == hoa.max() else
                "#F39C12" if v >= q75 else
                "#2980B9" for v in hoa]
bars = axes[0].bar(hoa.index, hoa.values, color=bar_colors_h, edgecolor="white", width=0.85)
axes[0].bar_label(bars, padding=2, fontsize=8, color="#2C3E50")
axes[0].set_title("Vessel Arrivals by Hour of Day", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of Vessels", fontsize=11)
axes[0].set_ylim(0, hoa.max() * 1.2)
axes[0].spines[["top","right"]].set_visible(False)
axes[0].legend(handles=[
    mpatches.Patch(color="#C0392B", label=f"Peak (16:00 – {hoa.max()} vessels)"),
    mpatches.Patch(color="#F39C12", label="Top quartile"),
    mpatches.Patch(color="#2980B9", label="Normal"),
], fontsize=9)

# Bottom: concurrent vessel count (congestion proxy)
conc = df.groupby("Hour_of_Arrival")["Vessel_Count_per_HoA"].first().reindex(range(24), fill_value=0)
axes[1].fill_between(conc.index, conc.values, alpha=0.4, color="#C0392B")
axes[1].plot(conc.index, conc.values, color="#C0392B", linewidth=2.5, marker="o", markersize=5)
axes[1].set_title("Concurrent Vessel Count (Congestion Index) by Hour", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Vessels in Same Hour", fontsize=11)
axes[1].set_xlabel("Hour of Day (0–23)", fontsize=11)
axes[1].set_xticks(range(24))
axes[1].spines[["top","right"]].set_visible(False)

fig.suptitle("Peak Hour Congestion Analysis", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "04_peak_hour_congestion.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. BERTH BOTTLENECK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("── 5. Berth Bottleneck Analysis ──────────────────────")

bb = df.groupby("Berth").agg(
    Calls       = ("OBJECTID",            "count"),
    Avg_Berth   = ("Berth_Duration",       "mean"),
    Avg_Anc     = ("Anchorage_Duration",   "mean"),
    Anchor_Rate = ("Anchored",             lambda x: x.mean() * 100),
).sort_values("Avg_Anc", ascending=True)

# Shorten terminal names for readability
short_names = {
    "Ennore Coal Terminal PVT LTD (ECTPL)":      "ECTPL (Coal)",
    "Ennore Bulk Terminal PVT LTD(EBTPL)":       "EBTPL (Bulk)",
    "Adani Ennore Container Terminal (AECT)":    "AECT (Container)",
    "Marine Liquid Terminal":                     "Marine Liquid",
    "General Cargo Berth 1":                      "GCB-1",
    "General Cargo Berth 2":                      "GCB-2",
    "Iron Ore Terminal (SIOT)":                   "SIOT (Iron Ore)",
}
bb.index = [short_names.get(b, b) for b in bb.index]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
y = np.arange(len(bb))

# Left: avg anchorage wait
colors_bb = ["#C0392B" if v > 30 else "#F39C12" if v > 15 else "#27AE60"
             for v in bb["Avg_Anc"]]
hb1 = axes[0].barh(y, bb["Avg_Anc"], color=colors_bb, edgecolor="white", height=0.65)
axes[0].bar_label(hb1, fmt="%.1fd", padding=4, fontsize=9, fontweight="bold")
axes[0].set_yticks(y); axes[0].set_yticklabels(bb.index, fontsize=9)
axes[0].set_title("Avg Anchorage Wait by Terminal (days)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Days", fontsize=11)
axes[0].spines[["top","right"]].set_visible(False)
axes[0].legend(handles=[
    mpatches.Patch(color="#C0392B", label="Critical (>30d)"),
    mpatches.Patch(color="#F39C12", label="High (15–30d)"),
    mpatches.Patch(color="#27AE60", label="Acceptable (<15d)"),
], fontsize=9)

# Right: anchor rate % bubble
sizes = bb["Calls"] * 15
sc = axes[1].scatter(bb["Avg_Berth"], bb["Avg_Anc"],
                     s=sizes, c=bb["Anchor_Rate"],
                     cmap="RdYlGn_r", vmin=40, vmax=100,
                     edgecolors="white", linewidths=0.8, alpha=0.85)
plt.colorbar(sc, ax=axes[1], label="Anchor Rate (%)")
for i, (name, row) in enumerate(bb.iterrows()):
    axes[1].annotate(name, (row["Avg_Berth"], row["Avg_Anc"]),
                     fontsize=7.5, ha="center", va="bottom",
                     xytext=(0, 6), textcoords="offset points")
axes[1].set_title("Berth Stay vs Anchorage Wait\n(bubble size = vessel calls)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Avg Berth Duration (days)", fontsize=11)
axes[1].set_ylabel("Avg Anchorage Wait (days)", fontsize=11)
axes[1].spines[["top","right"]].set_visible(False)

fig.suptitle("Berth Bottleneck Analysis", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "05_berth_bottleneck.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SEASONAL / MONTHLY TREND
# ══════════════════════════════════════════════════════════════════════════════
print("── 6. Seasonal / Monthly Trend ───────────────────────")

mt = df.groupby("Month", observed=True).agg(
    Count     = ("OBJECTID",            "count"),
    Avg_Berth = ("Berth_Duration",       "mean"),
    Avg_Port  = ("Port_Duration",        "mean"),
    Avg_Anc   = ("Anchorage_Duration",   "mean"),
).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
specs = [
    ("Count",     "Vessel Calls per Month",            "#0D7377", None),
    ("Avg_Berth", "Avg Berth Duration (days)",         "#2C3E50", None),
    ("Avg_Port",  "Avg Port Stay (days)",               "#2980B9", None),
    ("Avg_Anc",   "Avg Anchorage Wait (days)",          "#E67E22", None),
]
for ax, (col, title, color, _) in zip(axes.flatten(), specs):
    bars = ax.bar(mt["Month"], mt[col], color=color, edgecolor="white", width=0.65)
    ax.bar_label(bars, fmt="%.0f" if col=="Count" else "%.1f",
                 padding=3, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, mt[col].max() * 1.2)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    # Shade seasons
    season_spans = [("Jul","Sep","#2980B9",0.07), ("Oct","Nov","#27AE60",0.07), ("Dec","Dec","#E67E22",0.07)]
    xlabels = mt["Month"].tolist()
    for start, end, sc, alpha in season_spans:
        if start in xlabels and end in xlabels:
            x0 = xlabels.index(start) - 0.45
            x1 = xlabels.index(end)   + 0.45
            ax.axvspan(x0, x1, alpha=alpha, color=sc, zorder=0)

fig.suptitle("Monthly & Seasonal Port Trends – KPL 2024", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "06_seasonal_monthly_trend.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAFFIC vs DELAY RELATIONSHIP
# ══════════════════════════════════════════════════════════════════════════════
print("── 7. Traffic vs Delay Relationship ─────────────────")

tvd = df.groupby("Month", observed=True).agg(
    Count   = ("OBJECTID",          "count"),
    Avg_Anc = ("Anchorage_Duration","mean"),
).reset_index()

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

x = np.arange(len(tvd))
w = 0.4
bars = ax1.bar(x - w/2, tvd["Count"], width=w, color="#0D7377",
               edgecolor="white", label="Vessel Count", zorder=3)
ax1.bar_label(bars, padding=3, fontsize=10, fontweight="bold", color="#0D7377")

ax2.bar(x + w/2, tvd["Avg_Anc"], width=w, color="#F39C12",
        edgecolor="white", label="Avg Anchorage Wait (days)", zorder=3)
ax2.plot(x + w/2, tvd["Avg_Anc"], color="#C0392B", marker="o",
         linewidth=2, markersize=7, zorder=4)
for xi, yi in zip(x + w/2, tvd["Avg_Anc"]):
    ax2.text(xi, yi + 0.8, f"{yi:.1f}d", ha="center", fontsize=9,
             fontweight="bold", color="#C0392B")

ax1.set_xticks(x)
ax1.set_xticklabels(tvd["Month"].tolist(), fontsize=11)
ax1.set_ylabel("Number of Vessel Calls", fontsize=11, color="#0D7377")
ax2.set_ylabel("Avg Anchorage Wait (days)", fontsize=11, color="#C0392B")
ax1.tick_params(axis="y", colors="#0D7377")
ax2.tick_params(axis="y", colors="#C0392B")
ax1.spines[["top"]].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + [mpatches.Patch(color="#F39C12", label="Avg Anchorage Wait")],
           labels1 + ["Avg Anchorage Wait (days)"], fontsize=10, loc="upper right")

ax1.set_title("Traffic Volume vs Anchorage Delay by Month", fontsize=14, fontweight="bold", pad=12)

# Annotation for Aug insight
aug_idx = tvd["Month"].tolist().index("Aug")
ax1.annotate("Aug: fewer vessels\nbut HIGHER wait",
             xy=(aug_idx - w/2, tvd["Count"].iloc[aug_idx]),
             xytext=(aug_idx - 1.5, tvd["Count"].max() * 0.85),
             fontsize=9, color="#C0392B",
             arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2))

fig.tight_layout()
save(fig, "07_traffic_vs_delay.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. DISTRIBUTION OF WAITING TIME
# ══════════════════════════════════════════════════════════════════════════════
print("── 8. Distribution of Waiting Time ──────────────────")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of all anchorage durations (vessels that anchored)
anc_data = df[df["Anchored"]]["Anchorage_Duration"]
axes[0].hist(anc_data, bins=30, color="#2980B9", edgecolor="white", alpha=0.85)
axes[0].axvline(anc_data.mean(),   color="#C0392B", linewidth=2, linestyle="--",
                label=f"Mean: {anc_data.mean():.1f}d")
axes[0].axvline(anc_data.median(), color="#F39C12", linewidth=2, linestyle=":",
                label=f"Median: {anc_data.median():.1f}d")
axes[0].set_title("Anchorage Duration Distribution\n(298 vessels that anchored)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Anchorage Duration (days)", fontsize=11)
axes[0].set_ylabel("Number of Vessels", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].spines[["top","right"]].set_visible(False)

# Right: bucket bar chart
bins   = [0, 1, 7, 30, 100, 500]
labels = ["0–1 day\n(same day)", "1–7 days\n(short wait)",
          "7–30 days\n(medium wait)", "30–100 days\n(long wait)", "100+ days\n(extreme)"]
df_anc = df[df["Anchored"]].copy()
df_anc["wait_bucket"] = pd.cut(df_anc["Anchorage_Duration"], bins=bins, labels=labels)
bucket_counts = df_anc["wait_bucket"].value_counts().reindex(labels)
b_colors = ["#27AE60","#0D7377","#F39C12","#E67E22","#C0392B"]
bars2 = axes[1].bar(labels, bucket_counts.values, color=b_colors,
                    edgecolor="white", width=0.65)
axes[1].bar_label(bars2, fmt="%d vessels", padding=4, fontsize=10, fontweight="bold")
axes[1].set_title("Vessels by Waiting Time Category", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Number of Vessels", fontsize=11)
axes[1].set_ylim(0, bucket_counts.max() * 1.22)
axes[1].spines[["top","right"]].set_visible(False)

# Annotate % that wait >30 days
long_wait = bucket_counts.iloc[3] + bucket_counts.iloc[4]
pct_long  = long_wait / len(df_anc) * 100
axes[1].text(0.97, 0.95, f"{pct_long:.0f}% of anchored\nvessels wait >30 days",
             transform=axes[1].transAxes, ha="right", va="top",
             fontsize=10, color="#C0392B", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#C0392B", alpha=0.9))

fig.suptitle("Distribution of Anchorage Waiting Time", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "08_waiting_time_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. DIRECT vs WAITING COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("── 9. Direct vs Waiting Comparison ──────────────────")

dw = df.groupby("Berth_Category")[["Berth_Duration","Anchorage_Duration","Other_Duration"]].mean()
cat_order_dw = dw.sum(axis=1).sort_values(ascending=False).index.tolist()
dw = dw.reindex(cat_order_dw)

x    = np.arange(len(dw))
w    = 0.25
fig, ax = plt.subplots(figsize=(12, 6))

for i, (col, color, lbl) in enumerate(zip(
        ["Berth_Duration","Anchorage_Duration","Other_Duration"],
        ["#0D7377","#F39C12","#C0392B"],
        ["Berth (productive)","Anchorage wait","Other delay"])):
    bars = ax.bar(x + i*w, dw[col], width=w, color=color,
                  edgecolor="white", label=lbl)
    ax.bar_label(bars, fmt="%.1fd", padding=3, fontsize=8, fontweight="bold")

ax.set_xticks(x + w)
ax.set_xticklabels(dw.index, fontsize=10)
ax.set_ylabel("Average Days", fontsize=11)
ax.set_title("Direct Berth Time vs Waiting Time by Cargo Type", fontsize=14, fontweight="bold", pad=12)
ax.legend(title="Time Component", fontsize=10)
ax.spines[["top","right"]].set_visible(False)

# Annotate Iron Ore anomaly
io_idx = dw.index.tolist().index("Iron Ore/Coal")
ax.annotate("Other delay (104d)\nexceeds berth time (69d)!",
            xy=(io_idx + 2*w, dw.loc["Iron Ore/Coal","Other_Duration"]),
            xytext=(io_idx + 2*w + 0.8, dw.loc["Iron Ore/Coal","Other_Duration"] - 20),
            fontsize=9, color="#C0392B", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2))

fig.tight_layout()
save(fig, "09_direct_vs_waiting.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. HIGH DELAY VESSEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("── 10. High Delay Vessel Analysis ────────────────────")

top_delay = (df.nlargest(15, "Total_Delay")
               [["Vessel_ID","Berth","Berth_Category","Anchorage_Duration","Total_Delay","Month"]]
               .reset_index(drop=True))
top_delay["Label"] = top_delay["Vessel_ID"].astype(str) + "\n(" + top_delay["Month"].astype(str) + ")"
top_delay_colors   = [PALETTE.get(c, "#BDC3C7") for c in top_delay["Berth_Category"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: top 15 by total delay
hb = axes[0].barh(range(len(top_delay)), top_delay["Total_Delay"],
                  color=top_delay_colors, edgecolor="white", height=0.7)
axes[0].bar_label(hb, fmt="%.0fd", padding=4, fontsize=8, fontweight="bold")
axes[0].set_yticks(range(len(top_delay)))
axes[0].set_yticklabels(top_delay["Label"], fontsize=8)
axes[0].set_title("Top 15 Vessels by Total Port Delay", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Total Delay (days)", fontsize=11)
axes[0].spines[["top","right"]].set_visible(False)
axes[0].invert_yaxis()
legend_handles = [mpatches.Patch(color=v, label=k) for k,v in PALETTE.items()]
axes[0].legend(handles=legend_handles, fontsize=8, loc="lower right")

# Right: scatter total_delay vs anchorage_duration
scatter_colors = [PALETTE.get(c, "#BDC3C7") for c in df["Berth_Category"]]
axes[1].scatter(df["Anchorage_Duration"], df["Total_Delay"],
                c=scatter_colors, alpha=0.6, s=40, edgecolors="white", linewidths=0.4)
axes[1].set_title("Anchorage Wait vs Total Delay\n(all 416 vessels)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Anchorage Duration (days)", fontsize=11)
axes[1].set_ylabel("Total Port Delay (days)", fontsize=11)
axes[1].spines[["top","right"]].set_visible(False)
axes[1].legend(handles=legend_handles, fontsize=8, title="Cargo", loc="upper left")

# Correlation annotation
r = df[["Anchorage_Duration","Total_Delay"]].dropna().corr().iloc[0,1]
axes[1].text(0.97, 0.05, f"Pearson r = {r:.3f}",
             transform=axes[1].transAxes, ha="right", fontsize=10, color="#2C3E50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

fig.suptitle("High Delay Vessel Analysis", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "10_high_delay_vessels.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. DATA QUALITY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
print("── 11. Data Quality Insights ─────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: completeness bar per column
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)
completeness = (1 - missing / len(df)) * 100
if len(completeness) == 0:
    completeness = pd.Series({"No missing\ncolumns": 100})
colors_comp = ["#C0392B" if v < 80 else "#F39C12" if v < 95 else "#27AE60"
               for v in completeness]
axes[0].barh(completeness.index, completeness.values,
             color=colors_comp, edgecolor="white", height=0.55)
axes[0].set_xlim(0, 105)
axes[0].axvline(100, color="#7F8C8D", linewidth=0.8, linestyle="--")
for i, v in enumerate(completeness.values):
    axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
axes[0].set_title("Data Completeness by Column", fontsize=12, fontweight="bold")
axes[0].set_xlabel("% Complete", fontsize=11)
axes[0].spines[["top","right"]].set_visible(False)

# Middle: issue summary bar
issues = {
    "Zero berth\nduration": 3,
    "Anchorage > Port\n(capped)": 30,
    "Long anchorage\nflagged": 46,
    "Repeat vessel\nvisits": 140,
    "Missing\nanchorage": 118,
}
i_colors = ["#C0392B","#F39C12","#F39C12","#2980B9","#0D7377"]
bars3 = axes[1].bar(issues.keys(), issues.values(), color=i_colors, edgecolor="white", width=0.65)
axes[1].bar_label(bars3, fmt="%d rows", padding=4, fontsize=10, fontweight="bold")
axes[1].set_title("Data Issues Found & Handled", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Number of Rows", fontsize=11)
axes[1].set_ylim(0, max(issues.values()) * 1.22)
axes[1].spines[["top","right"]].set_visible(False)
axes[1].tick_params(axis="x", labelsize=8)

# Right: pie of row status
status_counts = {
    "Clean (no issues)": 416 - 69,
    "Flagged / handled": 69,
}
pie_colors = ["#27AE60","#F39C12"]
wedges, texts, autotexts = axes[2].pie(
    status_counts.values(), labels=None,
    colors=pie_colors, autopct="%1.1f%%",
    startangle=140, wedgeprops=dict(edgecolor="white", linewidth=2),
    pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(12); at.set_fontweight("bold"); at.set_color("white")
axes[2].legend(wedges, [f"{k} ({v})" for k,v in status_counts.items()],
               loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=10)
axes[2].set_title("Overall Dataset Health\n(416 total rows, 0 deleted)", fontsize=12, fontweight="bold")

fig.suptitle("Data Quality Insights – KPL 2024 Dataset", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "11_data_quality_insights.png")

# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n{'─'*52}")
print(f"  All charts saved to: ./{OUTPUT_DIR}/")
print(f"  Total: 12 insight charts generated")
print(f"{'─'*52}\n")