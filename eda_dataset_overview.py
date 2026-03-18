"""
Dataset EDA – Vessel Timings KPL 2024
======================================
Deep exploratory analysis — know the data inside out.
Generates: eda/ folder with all PNG charts + prints a full text summary.

Run:
    python eda_dataset_overview.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FILE  = "Vessel_Timings_KPL_24.xlsx"
OUT        = "eda"
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    "Coal":          "#2C3E50",
    "Liquid":        "#2980B9",
    "Multi cargo":   "#27AE60",
    "Container":     "#E67E22",
    "Automobiles":   "#8E44AD",
    "Iron Ore/Coal": "#C0392B",
}
SEASON_COLORS  = {"Monsoon": "#2980B9", "Post-Monsoon": "#27AE60", "Winter": "#E67E22"}
MONTH_ORDER    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Load & basic clean ─────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE, engine="openpyxl")
df["Anchorage_Duration"] = df["Anchorage_Duration"].fillna(0)
df["Berth_Duration"]     = df["Berth_Duration"].replace(0, np.nan)
month_map = {"January":"Jan","February":"Feb","March":"Mar","April":"Apr",
             "May":"May","June":"Jun","July":"Jul","August":"Aug",
             "September":"Sep","October":"Oct","November":"Nov","December":"Dec"}
df["Month"] = pd.Categorical(df["Month"].replace(month_map), categories=MONTH_ORDER, ordered=True)

# Convert hours → days for all duration columns
for col in ["Berth_Duration","Port_Duration","Anchorage_Duration"]:
    df[col] = df[col] / 24

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔  {name}")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 1 – Dataset Overview Snapshot (text printed + saved as figure)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  DATASET OVERVIEW – Vessel Timings KPL 2024")
print("═"*60)
print(f"  Rows         : {len(df)}")
print(f"  Columns      : {df.shape[1]}")
print(f"  Date Range   : {df['Berth_Entry'].min().date()} → {df['Berth_Exit'].max().date()}")
print(f"  Unique Vessels: {df['Vessel_ID'].nunique()}  (276 vessels, 416 calls → repeat visitors)")
print(f"  Terminals    : {df['Berth'].nunique()}")
print(f"  Cargo Types  : {df['Berth_Category'].nunique()}")
print(f"  Seasons      : {', '.join(df['Season'].unique())}")
print(f"  Missing data : Anchorage cols only (118 rows = vessels w/o anchorage)")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 2 – Column-by-Column Summary Table (figure)
# ══════════════════════════════════════════════════════════════════════════════
col_info = []
for col in df.columns:
    dtype  = str(df[col].dtype)
    n_null = int(df[col].isna().sum())
    n_uniq = int(df[col].nunique())
    if df[col].dtype in [np.float64, np.int64]:
        sample = f"min={df[col].min():.2f}  max={df[col].max():.2f}  mean={df[col].mean():.2f}"
    else:
        top = df[col].value_counts().index[0] if n_uniq > 0 else "–"
        sample = f"top = '{top}'"
    col_info.append([col, dtype, n_null, n_uniq, sample])

fig, ax = plt.subplots(figsize=(18, 9))
ax.axis("off")
headers = ["Column", "Dtype", "Nulls", "Unique", "Sample / Range"]
tbl = ax.table(cellText=col_info, colLabels=headers, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.55)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EBF5FB")
    cell.set_edgecolor("#D5D8DC")
ax.set_title("Column-by-Column Summary", fontsize=14, fontweight="bold", pad=20)
fig.tight_layout()
save(fig, "E01_column_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 3 – Missing Values Heatmap
# ══════════════════════════════════════════════════════════════════════════════
missing = df.isnull().astype(int)
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(missing.T, aspect="auto", cmap="RdYlGn_r", interpolation="none", vmin=0, vmax=1)
ax.set_yticks(range(len(df.columns)))
ax.set_yticklabels(df.columns, fontsize=8)
ax.set_xlabel("Row index", fontsize=11)
ax.set_title("Missing Value Map  (Red = missing, Green = present)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.01, pad=0.02,
             ticks=[0,1]).set_ticklabels(["Present","Missing"])
fig.tight_layout()
save(fig, "E02_missing_value_map.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 4 – Duration Distributions (Histograms + KDE)
# ══════════════════════════════════════════════════════════════════════════════
dur_cols  = ["Berth_Duration","Port_Duration","Anchorage_Duration"]
dur_title = ["Berth Duration (days)","Port Duration (days)","Anchorage Duration (days)"]
dur_color = ["#2C3E50","#2980B9","#27AE60"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title, color in zip(axes, dur_cols, dur_title, dur_color):
    data = df[col].dropna()
    data = data[data > 0]
    ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(data.mean(),   color="#E74C3C", linestyle="--", linewidth=1.8, label=f"Mean: {data.mean():.1f}d")
    ax.axvline(data.median(), color="#F39C12", linestyle=":",  linewidth=1.8, label=f"Median: {data.median():.1f}d")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Days", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    stats_txt = f"Std: {data.std():.1f}d\nSkew: {data.skew():.2f}\nMax: {data.max():.1f}d"
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
fig.suptitle("Distribution of Duration Columns", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "E03_duration_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 5 – Duration Box Plots (all 3 side by side, by cargo category)
# ══════════════════════════════════════════════════════════════════════════════
cats = df["Berth_Category"].unique()
cat_colors = [PALETTE.get(c,"#BDC3C7") for c in cats]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, col, title in zip(axes, dur_cols, dur_title):
    data_by_cat = [df[df["Berth_Category"]==c][col].dropna().values for c in cats]
    bp = ax.boxplot(data_by_cat, labels=cats, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp["boxes"], cat_colors):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Days", fontsize=10)
    ax.set_xticklabels(cats, rotation=25, ha="right", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle("Duration Spread by Cargo Category", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "E04_duration_boxplots_by_category.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 6 – Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
corr_cols = ["Berth_Duration","Port_Duration","Anchorage_Duration",
             "Hour_of_Arrival","Hour_of_Departure",
             "Vessel_Count_per_HoA","Vessel_Count_per_Berth"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=9)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr.iloc[i,j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(val) > 0.5 else "#333")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Correlation Matrix – Numeric Variables", fontsize=14, fontweight="bold", pad=14)
fig.tight_layout()
save(fig, "E05_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 7 – Repeat Vessel Visit Frequency
# ══════════════════════════════════════════════════════════════════════════════
visit_freq = df["Vessel_ID"].value_counts()
freq_dist  = visit_freq.value_counts().sort_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: frequency distribution
bars = ax1.bar(freq_dist.index, freq_dist.values, color="#2980B9", edgecolor="white", width=0.7)
ax1.bar_label(bars, padding=3, fontsize=10, fontweight="bold")
ax1.set_title("How Many Times Did Vessels Visit?", fontsize=13, fontweight="bold")
ax1.set_xlabel("Number of Visits", fontsize=11)
ax1.set_ylabel("Number of Vessels", fontsize=11)
ax1.spines[["top","right"]].set_visible(False)

# Right: top 10 most frequent vessels
top10 = visit_freq.head(10)
hbars = ax2.barh(range(len(top10)), top10.values, color="#2C3E50", edgecolor="white")
ax2.bar_label(hbars, padding=3, fontsize=10, fontweight="bold", color="white",
              label_type="center")
ax2.set_yticks(range(len(top10)))
ax2.set_yticklabels([str(v) for v in top10.index], fontsize=9)
ax2.set_title("Top 10 Most Frequent Vessel IDs", fontsize=13, fontweight="bold")
ax2.set_xlabel("Total Visits", fontsize=11)
ax2.spines[["top","right"]].set_visible(False)
ax2.invert_yaxis()

fig.suptitle("Repeat Vessel Visit Analysis\n(276 unique vessels, 416 total calls)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "E06_repeat_vessel_visits.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 8 – Berth × Cargo Category Heatmap
# ══════════════════════════════════════════════════════════════════════════════
crosstab = pd.crosstab(df["Berth"], df["Berth_Category"])
fig, ax = plt.subplots(figsize=(13, 7))
im = ax.imshow(crosstab.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(crosstab.columns))); ax.set_xticklabels(crosstab.columns, fontsize=10)
ax.set_yticks(range(len(crosstab.index)));   ax.set_yticklabels(crosstab.index, fontsize=9)
for i in range(len(crosstab.index)):
    for j in range(len(crosstab.columns)):
        val = crosstab.iloc[i,j]
        if val > 0:
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val > crosstab.values.max()*0.5 else "#2C3E50")
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03, label="Vessel Calls")
ax.set_title("Terminal × Cargo Type — Vessel Call Count", fontsize=14, fontweight="bold", pad=14)
fig.tight_layout()
save(fig, "E07_terminal_cargo_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 9 – Arrival & Departure Hour Patterns
# ══════════════════════════════════════════════════════════════════════════════
arr = df["Hour_of_Arrival"].value_counts().reindex(range(24), fill_value=0)
dep = df["Hour_of_Departure"].value_counts().reindex(range(24), fill_value=0)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, data, label, color in zip(
        axes,
        [arr, dep],
        ["Arrivals", "Departures"],
        ["#2980B9","#E67E22"]):
    q75 = data.quantile(0.75)
    bar_colors = ["#E74C3C" if v == data.max() else
                  "#F39C12" if v >= q75 else color
                  for v in data]
    bars = ax.bar(data.index, data.values, color=bar_colors, edgecolor="white", width=0.85)
    ax.bar_label(bars, padding=2, fontsize=8, color="#333")
    ax.set_ylabel(f"# {label}", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title(f"Vessel {label} by Hour of Day", fontsize=12, fontweight="bold")
    ax.set_ylim(0, data.max()*1.2)
    ax.legend(handles=[
        mpatches.Patch(color="#E74C3C", label="Peak"),
        mpatches.Patch(color="#F39C12", label="High"),
        mpatches.Patch(color=color,     label="Normal"),
    ], fontsize=8, loc="upper left")
axes[1].set_xlabel("Hour of Day (0–23)", fontsize=11)
axes[1].set_xticks(range(24))
fig.suptitle("24-Hour Arrival & Departure Patterns", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "E08_arrival_departure_hourly.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 10 – Day-of-Week: Arrivals vs Departures
# ══════════════════════════════════════════════════════════════════════════════
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
arr_day   = df["Day_of_Arrival"].value_counts().reindex(day_order, fill_value=0)
dep_day   = df["Day_of_Departure"].value_counts().reindex(day_order, fill_value=0)
x = np.arange(len(day_order))

fig, ax = plt.subplots(figsize=(12, 5))
w = 0.38
b1 = ax.bar(x - w/2, arr_day.values, width=w, label="Arrivals",   color="#2980B9", edgecolor="white")
b2 = ax.bar(x + w/2, dep_day.values, width=w, label="Departures", color="#E67E22", edgecolor="white")
ax.bar_label(b1, padding=3, fontsize=9, fontweight="bold", color="#2980B9")
ax.bar_label(b2, padding=3, fontsize=9, fontweight="bold", color="#E67E22")
ax.set_xticks(x); ax.set_xticklabels(day_order, fontsize=10)
ax.set_ylabel("Number of Vessels", fontsize=11)
ax.set_title("Arrivals vs Departures by Day of Week", fontsize=14, fontweight="bold", pad=12)
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=10)
ax.set_ylim(0, max(arr_day.max(), dep_day.max()) * 1.2)
fig.tight_layout()
save(fig, "E09_arrival_vs_departure_day.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 11 – Vessels With vs Without Anchorage (by category)
# ══════════════════════════════════════════════════════════════════════════════
df["Had_Anchorage"] = df["Anchorage_Duration"] > 0

anc_grp = df.groupby("Berth_Category")["Had_Anchorage"].value_counts().unstack(fill_value=0)
anc_grp.columns = ["No Anchorage","Had Anchorage"]
anc_grp["Total"] = anc_grp.sum(axis=1)
anc_grp = anc_grp.sort_values("Total", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(anc_grp))
w = 0.38
b1 = ax.barh(y - w/2, anc_grp["Had Anchorage"], height=w, label="Had Anchorage", color="#2980B9", edgecolor="white")
b2 = ax.barh(y + w/2, anc_grp["No Anchorage"],  height=w, label="No Anchorage",  color="#BDC3C7", edgecolor="white")
ax.bar_label(b1, padding=3, fontsize=9, fontweight="bold")
ax.bar_label(b2, padding=3, fontsize=9, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(anc_grp.index, fontsize=10)
ax.set_xlabel("Number of Vessels", fontsize=11)
ax.set_title("Anchorage Usage by Cargo Type", fontsize=14, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "E10_anchorage_usage_by_category.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 12 – Monthly Stats: Count + Avg Berth + Avg Port + Avg Anchorage
# ══════════════════════════════════════════════════════════════════════════════
mstats = df.groupby("Month", observed=True).agg(
    Vessel_Count   = ("OBJECTID","count"),
    Avg_Berth      = ("Berth_Duration","mean"),
    Avg_Port       = ("Port_Duration","mean"),
    Avg_Anchorage  = ("Anchorage_Duration", lambda x: x[x>0].mean()),
).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
specs = [
    ("Vessel_Count",  "Vessel Calls per Month",             "#2980B9", None),
    ("Avg_Berth",     "Avg Berth Duration per Month (days)","#2C3E50", "days"),
    ("Avg_Port",      "Avg Port Stay per Month (days)",     "#27AE60", "days"),
    ("Avg_Anchorage", "Avg Anchorage Wait per Month (days)","#E67E22", "days"),
]
for ax, (col, title, color, unit) in zip(axes.flatten(), specs):
    bars = ax.bar(mstats["Month"], mstats[col], color=color, edgecolor="white", width=0.6)
    fmt  = "%.1f" if unit else "%d"
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(unit if unit else "Count", fontsize=10)
    ax.set_ylim(0, mstats[col].max() * 1.22)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

fig.suptitle("Month-by-Month Port Statistics", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "E11_monthly_stats_grid.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 13 – Longest vs Shortest Stays (Top 10 each)
# ══════════════════════════════════════════════════════════════════════════════
longest  = df.nlargest(10, "Berth_Duration")[["Vessel_ID","Berth","Berth_Category","Berth_Duration","Month"]].reset_index(drop=True)
shortest = df[df["Berth_Duration"] > (1/24)].nsmallest(10, "Berth_Duration")[["Vessel_ID","Berth","Berth_Category","Berth_Duration","Month"]].reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
colors_l = [PALETTE.get(c,"#BDC3C7") for c in longest["Berth_Category"]]
colors_s = [PALETTE.get(c,"#BDC3C7") for c in shortest["Berth_Category"]]

hb1 = ax1.barh(range(len(longest)), longest["Berth_Duration"], color=colors_l, edgecolor="white")
ax1.set_yticks(range(len(longest)))
ax1.set_yticklabels([f"{r['Berth'].split('(')[0].strip()[:28]}\n({r['Month']})" for _, r in longest.iterrows()], fontsize=8)
ax1.bar_label(hb1, fmt="%.1f d", padding=3, fontsize=9, fontweight="bold")
ax1.set_title("Top 10 Longest Berth Stays", fontsize=12, fontweight="bold")
ax1.set_xlabel("Berth Duration (days)", fontsize=10)
ax1.spines[["top","right"]].set_visible(False)
ax1.invert_yaxis()

hb2 = ax2.barh(range(len(shortest)), shortest["Berth_Duration"] * 24, color=colors_s, edgecolor="white")
ax2.set_yticks(range(len(shortest)))
ax2.set_yticklabels([f"{r['Berth'].split('(')[0].strip()[:28]}\n({r['Month']})" for _, r in shortest.iterrows()], fontsize=8)
ax2.bar_label(hb2, fmt="%.2f h", padding=3, fontsize=9, fontweight="bold")
ax2.set_title("Top 10 Shortest Berth Stays", fontsize=12, fontweight="bold")
ax2.set_xlabel("Berth Duration (hours)", fontsize=10)
ax2.spines[["top","right"]].set_visible(False)
ax2.invert_yaxis()

handles = [mpatches.Patch(color=v, label=k) for k,v in PALETTE.items()]
fig.legend(handles=handles, title="Cargo Type", loc="lower center", ncol=6,
           fontsize=8, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Extreme Berth Duration Cases", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "E12_longest_shortest_stays.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 14 – Port Duration Breakdown: Berth + Anchorage + Other
# ══════════════════════════════════════════════════════════════════════════════
df["Other_Duration"] = (df["Port_Duration"] - df["Berth_Duration"] - df["Anchorage_Duration"]).clip(lower=0)
dur_breakdown = df.groupby("Berth_Category")[["Berth_Duration","Anchorage_Duration","Other_Duration"]].mean()

fig, ax = plt.subplots(figsize=(12, 5))
x     = np.arange(len(dur_breakdown))
w     = 0.22
cols  = ["#2C3E50","#2980B9","#BDC3C7"]
labs  = ["Berth Duration","Anchorage Duration","Other (waiting/transit)"]
for i, (col, lab, c) in enumerate(zip(dur_breakdown.columns, labs, cols)):
    bars = ax.bar(x + (i-1)*w, dur_breakdown[col], width=w, label=lab, color=c, edgecolor="white")
    ax.bar_label(bars, fmt="%.1fd", padding=2, fontsize=7.5, rotation=90, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(dur_breakdown.index, fontsize=10)
ax.set_ylabel("Average Duration (days)", fontsize=11)
ax.set_title("Port Time Breakdown by Cargo Category\n(Berth + Anchorage + Other)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0, dur_breakdown.values.max() * 1.35)
fig.tight_layout()
save(fig, "E13_port_time_breakdown.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 15 – Vessel Count per Hour of Arrival (Polar / Clock chart)
# ══════════════════════════════════════════════════════════════════════════════
hoa = df["Hour_of_Arrival"].value_counts().reindex(range(24), fill_value=0)
angles = np.linspace(0, 2*np.pi, 24, endpoint=False)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.bar(angles, hoa.values, width=2*np.pi/24, color="#2980B9", alpha=0.75,
       edgecolor="white", linewidth=0.8)
ax.set_xticks(angles)
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], fontsize=8)
ax.set_title("Vessel Arrivals – 24-Hour Clock", fontsize=14, fontweight="bold", pad=20)
ax.set_facecolor("#F8F9FA")
# Annotate peak
peak_h = hoa.idxmax()
ax.annotate(f"Peak\n{peak_h}:00\n({hoa.max()} vessels)",
            xy=(angles[peak_h], hoa.max()),
            xytext=(angles[peak_h]+0.3, hoa.max()+15),
            fontsize=9, color="#E74C3C", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E74C3C"))
fig.tight_layout()
save(fig, "E14_arrival_clock_polar.png")

# ══════════════════════════════════════════════════════════════════════════════
# EDA 16 – Scatter: Berth Duration vs Port Duration (coloured by season)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
for season, grp in df.groupby("Season"):
    ax.scatter(grp["Berth_Duration"], grp["Port_Duration"],
               label=season, alpha=0.65, s=55,
               color=SEASON_COLORS.get(season,"#BDC3C7"),
               edgecolors="white", linewidths=0.5)
# perfect line (port = berth)
lim = df["Port_Duration"].max() * 1.05
ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.4, label="Port = Berth (ref)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_title("Berth Duration vs Port Duration  (by Season)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Berth Duration (days)", fontsize=12)
ax.set_ylabel("Port Duration (days)", fontsize=12)
ax.legend(title="Season", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
r = df[["Berth_Duration","Port_Duration"]].dropna().corr().iloc[0,1]
ax.text(0.03, 0.96, f"Pearson r = {r:.3f}", transform=ax.transAxes,
        fontsize=10, color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
fig.tight_layout()
save(fig, "E15_berth_vs_port_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# Print full numeric summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  NUMERIC SUMMARY (durations in days)")
print("═"*60)
for col in ["Berth_Duration","Port_Duration","Anchorage_Duration"]:
    s = df[col].dropna()
    s = s[s > 0]
    print(f"\n  {col}:")
    print(f"    Count  : {len(s)}   Missing : {df[col].isna().sum()}")
    print(f"    Min    : {s.min():.2f}d   Max   : {s.max():.1f}d")
    print(f"    Mean   : {s.mean():.1f}d   Median: {s.median():.1f}d")
    print(f"    Std    : {s.std():.1f}d   Skew  : {s.skew():.2f}")
    print(f"    25th % : {s.quantile(0.25):.1f}d  75th % : {s.quantile(0.75):.1f}d")

print("\n" + "═"*60)
print("  KEY CATEGORICAL FACTS")
print("═"*60)
print(f"\n  Berth Traffic (by terminal):")
for b, n in df["Berth"].value_counts().items():
    print(f"    {b:<45} {n:>3} calls")

print(f"\n  Cargo Mix:")
for c, n in df["Berth_Category"].value_counts().items():
    print(f"    {c:<20} {n:>3} calls  ({n/len(df)*100:.1f}%)")

print(f"\n  Seasonal Split:")
for s, n in df["Season"].value_counts().items():
    print(f"    {s:<15} {n:>3} calls  ({n/len(df)*100:.1f}%)")

print(f"\n  Anchorage Usage:")
print(f"    Vessels that anchored  : {(df['Anchorage_Duration']>0).sum()} ({(df['Anchorage_Duration']>0).mean()*100:.0f}%)")
print(f"    Went straight to berth : {(df['Anchorage_Duration']==0).sum()} ({(df['Anchorage_Duration']==0).mean()*100:.0f}%)")

print(f"\n  Repeat Visitors:")
vc = df["Vessel_ID"].value_counts()
print(f"    Unique vessels         : {len(vc)}")
print(f"    Visited once           : {(vc==1).sum()}")
print(f"    Visited 2-5 times      : {((vc>=2)&(vc<=5)).sum()}")
print(f"    Visited 6+ times       : {(vc>=6).sum()}")
print(f"    Most frequent vessel   : ID {vc.index[0]}  ({vc.iloc[0]} visits)")

print(f"\n  All EDA charts saved to: ./{OUT}/\n")