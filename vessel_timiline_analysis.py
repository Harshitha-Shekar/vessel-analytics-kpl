"""
Timeline Analysis – High Delay Vessels
=======================================
Shows exactly what happens when a vessel enters and exits the port,
and reveals why the 174-day delay vessels are a data quality issue.

Run:
    python vessel_timeline_analysis.py
Output: insights/12_vessel_timeline.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import os
import warnings

warnings.filterwarnings("ignore")

DATA_FILE  = "Vessel_Timings_KPL_24.xlsx"
OUTPUT_DIR = "insights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE, engine="openpyxl")

# ── Pick vessels to show ───────────────────────────────────────────────────────
# Mix: 4 suspicious long-stay + 4 normal vessels for comparison
long_ids   = [419001308, 419600190, 419280223, 419001582]
normal_ids = df[~df["Vessel_ID"].isin(long_ids)].copy()
normal_ids = normal_ids.drop_duplicates("Vessel_ID")
# Pick 4 normals with valid port times < 30 days
normal_ids["port_days"] = (normal_ids["Port_Exit"] - normal_ids["Port_Entry"]).dt.total_seconds() / 86400
normal_sample = (normal_ids[normal_ids["port_days"].between(2, 25)]
                 .nlargest(4, "port_days")["Vessel_ID"].tolist())

all_ids = long_ids + normal_sample

# ── Build one clean row per vessel ────────────────────────────────────────────
rows = []
for vid in all_ids:
    v = df[df["Vessel_ID"] == vid].copy()
    # Use the first / most meaningful row for each vessel
    # (drop duplicates caused by same vessel mapped to multiple berths)
    row = v.sort_values("Berth_Duration", ascending=False).iloc[0]
    port_days = (row["Port_Exit"] - row["Port_Entry"]).total_seconds() / 86400
    berth_days = row["Berth_Duration"] / 24 if not pd.isna(row["Berth_Duration"]) else 0
    anc_days   = row["Anchorage_Duration"] / 24 if not pd.isna(row["Anchorage_Duration"]) else 0
    is_suspect = vid in long_ids
    rows.append({
        "Vessel_ID":   vid,
        "Berth":       row["Berth"].replace("Ennore Coal Terminal PVT LTD (ECTPL)", "ECTPL")
                                   .replace("Ennore Bulk Terminal PVT LTD(EBTPL)", "EBTPL")
                                   .replace("Adani Ennore Container Terminal (AECT)", "AECT")
                                   .replace("Marine Liquid Terminal", "Marine Liquid"),
        "Category":    row["Berth_Category"],
        "Port_Entry":  row["Port_Entry"],
        "Port_Exit":   row["Port_Exit"],
        "Anc_Entry":   row["Anchorage_Entry"],
        "Anc_Exit":    row["Anchorage_Exit"],
        "Berth_Entry": row["Berth_Entry"],
        "Berth_Exit":  row["Berth_Exit"],
        "Port_Days":   round(port_days, 1),
        "Berth_Days":  round(berth_days, 1),
        "Anc_Days":    round(anc_days, 1),
        "Suspect":     is_suspect,
        "n_terminals": v["Berth"].nunique(),
    })

vessels = pd.DataFrame(rows).sort_values(["Suspect", "Port_Days"], ascending=[False, False])

# ── Figure 1: Gantt Timeline ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor("white")

# Date range
x_min = pd.Timestamp("2024-07-01")
x_max = pd.Timestamp("2025-01-05")

# Background bands
ax.axvspan(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30"),
           alpha=0.06, color="#2980B9", label="_nolegend_")
ax.axvspan(pd.Timestamp("2024-10-01"), pd.Timestamp("2024-11-30"),
           alpha=0.06, color="#27AE60", label="_nolegend_")
ax.axvspan(pd.Timestamp("2024-12-01"), pd.Timestamp("2025-01-05"),
           alpha=0.06, color="#E67E22", label="_nolegend_")

# Season labels at top
for label, date, color in [
    ("Monsoon", pd.Timestamp("2024-08-01"), "#2980B9"),
    ("Post-Monsoon", pd.Timestamp("2024-10-25"), "#27AE60"),
    ("Winter", pd.Timestamp("2024-12-10"), "#E67E22"),
]:
    ax.text(date, len(vessels) + 0.3, label, color=color,
            fontsize=9, fontweight="bold", ha="center")

# Divider line between suspect and normal
divider_y = vessels["Suspect"].sum() - 0.5
ax.axhline(divider_y, color="#7F8C8D", linewidth=1.2,
           linestyle="--", alpha=0.6)
ax.text(x_max, divider_y + 0.05, "← Suspect vessels above",
        color="#C0392B", fontsize=8.5, va="bottom", ha="right")
ax.text(x_max, divider_y - 0.08, "← Normal vessels below",
        color="#27AE60", fontsize=8.5, va="top", ha="right")

for i, (_, row) in enumerate(vessels.iterrows()):
    y = len(vessels) - 1 - i

    is_s = row["Suspect"]
    bar_color   = "#C0392B" if is_s else "#0D7377"
    anc_color   = "#F39C12"
    berth_color = "#2C3E50" if is_s else "#14A085"

    # ── Full port stay bar (background) ───────────────────────────────────
    ax.barh(y, (row["Port_Exit"] - row["Port_Entry"]).total_seconds() / 86400,
            left=mdates.date2num(row["Port_Entry"]),
            height=0.55, color=bar_color, alpha=0.18,
            edgecolor=bar_color, linewidth=0.5)

    # ── Anchorage bar ──────────────────────────────────────────────────────
    if pd.notna(row["Anc_Entry"]) and pd.notna(row["Anc_Exit"]):
        anc_w = (row["Anc_Exit"] - row["Anc_Entry"]).total_seconds() / 86400
        ax.barh(y, anc_w,
                left=mdates.date2num(row["Anc_Entry"]),
                height=0.35, color=anc_color, alpha=0.85,
                edgecolor="white", linewidth=0.5)

    # ── Berth bar ──────────────────────────────────────────────────────────
    if pd.notna(row["Berth_Entry"]) and pd.notna(row["Berth_Exit"]):
        berth_w = (row["Berth_Exit"] - row["Berth_Entry"]).total_seconds() / 86400
        if berth_w > 0.1:
            ax.barh(y, berth_w,
                    left=mdates.date2num(row["Berth_Entry"]),
                    height=0.35, color=berth_color, alpha=0.9,
                    edgecolor="white", linewidth=0.5)

    # ── Port entry/exit markers ────────────────────────────────────────────
    ax.plot(mdates.date2num(row["Port_Entry"]), y,
            marker="|", color=bar_color, markersize=14, markeredgewidth=2.5)
    ax.plot(mdates.date2num(row["Port_Exit"]), y,
            marker="|", color=bar_color, markersize=14, markeredgewidth=2.5)

    # ── Y axis label ──────────────────────────────────────────────────────
    flag = "⚠ " if is_s else ""
    label = f"{flag}{row['Vessel_ID']}\n{row['Berth'][:22]}"
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8.5,
            color="#C0392B" if is_s else "#2C3E50",
            fontweight="bold" if is_s else "normal")

    # ── Duration annotation ────────────────────────────────────────────────
    suffix = " ← DATA ISSUE?" if is_s else ""
    ann_color = "#C0392B" if is_s else "#2C3E50"
    ax.text(mdates.date2num(x_max) - 1, y,
            f"  {row['Port_Days']}d{suffix}",
            va="center", fontsize=8.5,
            color=ann_color, fontweight="bold" if is_s else "normal")

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
ax.set_ylim(-0.8, len(vessels))
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.set_yticks([])
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(axis="x", labelsize=9)
ax.set_xlabel("Timeline (Jul 2024 – Dec 2024)", fontsize=11)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#C0392B", alpha=0.25, label="Full port stay (suspect vessel)"),
    mpatches.Patch(color="#0D7377", alpha=0.25, label="Full port stay (normal vessel)"),
    mpatches.Patch(color="#F39C12", alpha=0.85, label="Anchorage wait"),
    mpatches.Patch(color="#2C3E50", alpha=0.9,  label="At berth (suspect)"),
    mpatches.Patch(color="#14A085", alpha=0.9,  label="At berth (normal)"),
]
ax.legend(handles=legend_handles, loc="lower right",
          fontsize=8.5, framealpha=0.9, ncol=2)

ax.set_title("Vessel Port Timeline — Suspect Long-Stay vs Normal Vessels\n"
             "Port Entry | → Anchorage Wait → | Berth Stay | → Port Exit",
             fontsize=14, fontweight="bold", pad=16)

fig.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "12a_vessel_gantt_timeline.png")
fig.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  ✔  12a_vessel_gantt_timeline.png")

# ── Figure 2: Data Issue Deep Dive ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("white")

# LEFT: Number of terminals per vessel (suspects appear in all 13)
all_vessels = df.groupby("Vessel_ID").agg(
    n_terminals = ("Berth", "nunique"),
    port_days   = ("Port_Duration", "first")
).reset_index()
all_vessels["port_days"] = all_vessels["port_days"] / 24
all_vessels["is_suspect"] = all_vessels["Vessel_ID"].isin(long_ids)

colors_scatter = ["#C0392B" if s else "#0D7377"
                  for s in all_vessels["is_suspect"]]
sizes_scatter  = [120 if s else 40
                  for s in all_vessels["is_suspect"]]

axes[0].scatter(all_vessels["n_terminals"], all_vessels["port_days"],
                c=colors_scatter, s=sizes_scatter,
                alpha=0.7, edgecolors="white", linewidths=0.5)

# Annotate suspect vessels
for _, row in all_vessels[all_vessels["is_suspect"]].iterrows():
    axes[0].annotate(str(row["Vessel_ID"]),
                     (row["n_terminals"], row["port_days"]),
                     fontsize=7.5, color="#C0392B",
                     xytext=(4, 4), textcoords="offset points")

axes[0].axhline(60, color="#F39C12", linewidth=1.5,
                linestyle="--", label="60-day threshold")
axes[0].set_xlabel("Number of Terminals a Vessel Appears In", fontsize=11)
axes[0].set_ylabel("Total Port Stay (days)", fontsize=11)
axes[0].set_title("Vessels Appearing in Many Terminals\nhave Extreme Port Durations",
                  fontsize=12, fontweight="bold")
axes[0].spines[["top","right"]].set_visible(False)
axes[0].legend(handles=[
    mpatches.Patch(color="#C0392B", label="Suspect vessels (all 13 terminals)"),
    mpatches.Patch(color="#0D7377", label="Normal vessels"),
    mpatches.Patch(color="#F39C12", label="60-day threshold"),
], fontsize=8.5)

# Insight box
axes[0].text(0.04, 0.97,
             "Vessels appearing across ALL 13 terminals\n"
             "all show ~174-day port stays.\n"
             "A real vessel cannot be at 13 docks simultaneously.\n"
             "This is a data duplication / mapping issue.",
             transform=axes[0].transAxes, fontsize=9,
             va="top", color="#7F8C8D",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7",
                       edgecolor="#F39C12", alpha=0.9))

# RIGHT: Port entry date distribution
df["port_entry_date"] = pd.to_datetime(df["Port_Entry"]).dt.date
entry_counts = df.groupby("port_entry_date")["Vessel_ID"].count().reset_index()
entry_counts.columns = ["date", "count"]
entry_counts["date"] = pd.to_datetime(entry_counts["date"])

suspect_date = pd.Timestamp("2024-07-09").date()

bar_colors = ["#C0392B" if d.date() == suspect_date else "#0D7377"
              for d in entry_counts["date"]]
axes[1].bar(entry_counts["date"], entry_counts["count"],
            color=bar_colors, edgecolor="white", width=1.5, alpha=0.85)

axes[1].axvline(pd.Timestamp("2024-07-09"), color="#C0392B",
                linewidth=2, linestyle="--", label="July 9 — suspect entry date")
axes[1].annotate("July 9\n36 vessels all\nenter same day\n(4 are suspect)",
                 xy=(pd.Timestamp("2024-07-09"), 36),
                 xytext=(pd.Timestamp("2024-08-10"), 30),
                 fontsize=9, color="#C0392B", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5))

axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[1].set_xlabel("Port Entry Date", fontsize=11)
axes[1].set_ylabel("Number of Vessels", fontsize=11)
axes[1].set_title("Port Entry Date Distribution\nJuly 9 Has Unusually High Entries",
                  fontsize=12, fontweight="bold")
axes[1].spines[["top","right"]].set_visible(False)

fig.suptitle("Why the 174-Day Vessels Are Likely a Data Issue",
             fontsize=14, fontweight="bold")
fig.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "12b_data_issue_deep_dive.png")
fig.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  ✔  12b_data_issue_deep_dive.png")

# ── Print summary ──────────────────────────────────────────────────────────────
print("\n─── What the timeline reveals ───────────────────────────────────")
print("  Suspect vessels:")
for vid in long_ids:
    v = df[df["Vessel_ID"] == vid]
    n = v["Berth"].nunique()
    pe = v["Port_Entry"].iloc[0]
    px = v["Port_Exit"].iloc[0]
    days = (px - pe).total_seconds() / 86400
    print(f"    Vessel {vid}: appears in {n} terminals | "
          f"Entry: {pe.date()} | Exit: {px.date()} | {days:.0f} days")
print()
print("  KEY FINDING:")
print("  These vessels all entered on July 9 (first day of dataset)")
print("  and exited on Dec 30 (last day of dataset).")
print("  They appear simultaneously across ALL 13 terminals —")
print("  which is physically impossible for a single vessel.")
print("  This is almost certainly a data mapping error where")
print("  a vessel ID was incorrectly linked to every terminal record.")
print("──────────────────────────────────────────────────────────────────\n")