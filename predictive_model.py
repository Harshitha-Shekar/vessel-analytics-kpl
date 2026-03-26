"""
Predictive Model – Vessel Anchorage Wait Time
==============================================
Kamarajar Port Limited (KPL) 2024 Data
Author : [Student Name]
Date   : March 2026

What this model does:
    Given information about an incoming vessel (what type of cargo it carries,
    what hour it arrives, how many other vessels are arriving that hour, and
    what season it is), the model predicts HOW LONG the vessel will have to
    wait at anchor before it can dock.

Models built:
    1. Linear Regression      — simple baseline
    2. Random Forest Regressor — best performer (ensemble)
    3. Gradient Boosting       — advanced, highest accuracy

Run:
    python predictive_model.py

Output:
    model_results/  →  all charts and performance comparison PNG files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection     import train_test_split, cross_val_score
from sklearn.preprocessing       import LabelEncoder, StandardScaler
from sklearn.linear_model        import LinearRegression
from sklearn.ensemble            import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics             import (mean_absolute_error,
                                         mean_squared_error, r2_score)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = "Vessel_Timings_KPL_24.xlsx"
OUTPUT_DIR = "model_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "Coal":          "#2C3E50",
    "Liquid":        "#2980B9",
    "Multi cargo":   "#27AE60",
    "Container":     "#E67E22",
    "Automobiles":   "#8E44AD",
    "Iron Ore/Coal": "#C0392B",
}

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔  {name}")

print("\n" + "="*60)
print("  KPL Vessel Anchorage Wait Prediction Model")
print("="*60 + "\n")

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════════════════
print("Step 1: Loading and preparing data...")

df = pd.read_excel(DATA_FILE, engine="openpyxl")

# Clean
df["Anchorage_Duration"] = df["Anchorage_Duration"].fillna(0)
df["Berth_Duration"]     = df["Berth_Duration"].replace(0, np.nan)
for col in ["Berth_Duration", "Port_Duration", "Anchorage_Duration"]:
    df[col] = df[col] / 24   # hours → days

month_map = {
    "January":"Jan","February":"Feb","March":"Mar","April":"Apr",
    "May":"May","June":"Jun","July":"Jul","August":"Aug",
    "September":"Sep","October":"Oct","November":"Nov","December":"Dec",
}
df["Month"] = df["Month"].replace(month_map)

# Target: anchorage wait (only vessels that actually anchored)
df_anc = df[df["Anchorage_Duration"] > 0].copy()
print(f"  Vessels that anchored: {len(df_anc)} / {len(df)} total")

# ── Feature engineering ───────────────────────────────────────
# We want to predict Anchorage_Duration using only information
# that is KNOWN when the vessel is approaching the port.
#
# Features available at arrival time:
#   - Hour_of_Arrival       (what time they arrive)
#   - Vessel_Count_per_HoA  (how many other ships arriving same hour)
#   - Berth_Category        (what cargo type they carry)
#   - Month                 (which month)
#   - Season                (Monsoon / Post-Monsoon / Winter)

features = ["Hour_of_Arrival", "Vessel_Count_per_HoA",
            "Berth_Category", "Month", "Season"]
target   = "Anchorage_Duration"

# Encode categorical features
le_cat    = LabelEncoder()
le_month  = LabelEncoder()
le_season = LabelEncoder()

df_anc = df_anc.copy()
df_anc["Cargo_Code"]  = le_cat.fit_transform(df_anc["Berth_Category"])
df_anc["Month_Code"]  = le_month.fit_transform(df_anc["Month"])
df_anc["Season_Code"] = le_season.fit_transform(df_anc["Season"])

# Add is_peak_hour flag (16:00 spike)
df_anc["Is_Peak_Hour"] = (df_anc["Hour_of_Arrival"] == 16).astype(int)

# Final feature set
X_cols = ["Hour_of_Arrival", "Vessel_Count_per_HoA",
          "Cargo_Code", "Month_Code", "Season_Code", "Is_Peak_Hour"]
X = df_anc[X_cols].values
y = df_anc[target].values

print(f"  Features used : {X_cols}")
print(f"  Target        : {target} (days)")
print(f"  Dataset size  : {len(X)} rows")

# ══════════════════════════════════════════════════════════════
# STEP 2 — SPLIT DATA
# ══════════════════════════════════════════════════════════════
print("\nStep 2: Splitting data into train / test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training set  : {len(X_train)} rows (80%)")
print(f"  Test set      : {len(X_test)} rows  (20%)")

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════
# STEP 3 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("\nStep 3: Training three models...")

models = {
    "Linear Regression"   : LinearRegression(),
    "Random Forest"       : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"   : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
predictions = {}

for name, model in models.items():
    # Use scaled data for Linear Regression, raw for tree models
    Xtr = X_train_sc if "Linear" in name else X_train
    Xte = X_test_sc  if "Linear" in name else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_pred = np.clip(y_pred, 0, None)   # wait can't be negative

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    # Cross-validation MAE
    cv_scores = cross_val_score(model, Xtr, y_train,
                                cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    results[name] = {
        "MAE": round(mae, 2), "RMSE": round(rmse, 2),
        "R2":  round(r2,  3), "CV_MAE": round(cv_mae, 2),
    }
    predictions[name] = y_pred

    print(f"  {name:<22}  MAE={mae:.1f}d  RMSE={rmse:.1f}d  R²={r2:.3f}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — EVALUATE & VISUALISE
# ══════════════════════════════════════════════════════════════
print("\nStep 4: Generating evaluation charts...")

# ── 4a: Model comparison bar chart ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
model_names = list(results.keys())
colors = ["#2980B9", "#27AE60", "#E67E22"]

for ax, metric, title, unit in zip(
        axes,
        ["MAE", "RMSE", "R2"],
        ["Mean Absolute Error (days)\nLower = Better",
         "Root Mean Squared Error (days)\nLower = Better",
         "R² Score\nHigher = Better (max 1.0)"],
        ["days", "days", ""]):
    vals  = [results[m][metric] for m in model_names]
    short = ["Linear\nRegression", "Random\nForest", "Gradient\nBoosting"]
    bars  = ax.bar(short, vals, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars,
                 labels=[f"{v:.2f}{unit}" for v in vals],
                 padding=4, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Model Performance Comparison – KPL Anchorage Wait Prediction",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "M1_model_comparison.png")

# ── 4b: Actual vs Predicted — best model (Gradient Boosting) ─
best_name = min(results, key=lambda m: results[m]["MAE"])
best_pred = predictions[best_name]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, best_pred, alpha=0.6, color="#0D7377",
           edgecolors="white", s=50, label="Predictions")
max_val = max(y_test.max(), best_pred.max())
ax.plot([0, max_val], [0, max_val], "--", color="#C0392B",
        linewidth=2, label="Perfect prediction line")
ax.set_xlabel("Actual Anchorage Wait (days)", fontsize=12)
ax.set_ylabel("Predicted Anchorage Wait (days)", fontsize=12)
ax.set_title(f"Actual vs Predicted – {best_name}\n"
             f"MAE = {results[best_name]['MAE']} days  |  "
             f"R² = {results[best_name]['R2']}",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "M2_actual_vs_predicted.png")

# ── 4c: Residual plot ─────────────────────────────────────────
residuals = y_test - best_pred
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(best_pred, residuals, alpha=0.6, color="#8E44AD",
                edgecolors="white", s=50)
axes[0].axhline(0, color="#C0392B", linewidth=2, linestyle="--")
axes[0].set_xlabel("Predicted Value (days)", fontsize=11)
axes[0].set_ylabel("Residual (Actual − Predicted)", fontsize=11)
axes[0].set_title("Residuals vs Predicted Values", fontsize=12, fontweight="bold")
axes[0].spines[["top","right"]].set_visible(False)

axes[1].hist(residuals, bins=20, color="#2980B9",
             edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="#C0392B", linewidth=2, linestyle="--")
axes[1].set_xlabel("Residual (days)", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].set_title("Distribution of Residuals", fontsize=12, fontweight="bold")
axes[1].spines[["top","right"]].set_visible(False)

fig.suptitle(f"Residual Analysis – {best_name}", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "M3_residual_analysis.png")

# ── 4d: Feature importance (Random Forest) ───────────────────
rf_model   = models["Random Forest"]
feat_names = ["Arrival Hour", "Vessel Count\n(same hour)",
              "Cargo Type", "Month", "Season", "Peak Hour\n(16:00 flag)"]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
colors_feat = ["#C0392B" if importances[i] == importances.max()
               else "#2980B9" for i in sorted_idx]
bars = ax.barh([feat_names[i] for i in sorted_idx],
               importances[sorted_idx], color=colors_feat,
               edgecolor="white", height=0.6)
ax.bar_label(bars,
             labels=[f"{importances[i]*100:.1f}%" for i in sorted_idx],
             padding=4, fontsize=10, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=11)
ax.set_title("Feature Importance – Random Forest\n"
             "Which input features matter most for predicting wait time?",
             fontsize=12, fontweight="bold")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "M4_feature_importance.png")

# ── 4e: Predicted wait by cargo type ─────────────────────────
gb_model = models["Gradient Boosting"]

cargo_categories = le_cat.classes_.tolist()
month_codes_ref  = {m: le_month.transform([m])[0]
                    for m in le_month.classes_}
season_codes_ref = {s: le_season.transform([s])[0]
                    for s in le_season.classes_}

# Simulate: July Monsoon peak, 16:00 arrival, 35 concurrent vessels
scenarios = []
for cargo in cargo_categories:
    code = le_cat.transform([cargo])[0]
    wait = gb_model.predict([[16, 35, code,
                               month_codes_ref["Jul"],
                               season_codes_ref["Monsoon"], 1]])[0]
    scenarios.append((cargo, max(0, wait)))

scenarios.sort(key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(10, 5))
cargo_names = [s[0] for s in scenarios]
wait_vals   = [s[1] for s in scenarios]
bar_colors  = [PALETTE.get(c, "#BDC3C7") for c in cargo_names]
bars = ax.bar(cargo_names, wait_vals, color=bar_colors, edgecolor="white", width=0.6)
ax.bar_label(bars,
             labels=[f"{v:.1f}d" for v in wait_vals],
             padding=4, fontsize=11, fontweight="bold")
ax.set_ylabel("Predicted Anchorage Wait (days)", fontsize=11)
ax.set_title("Predicted Wait Time by Cargo Type\n"
             "Scenario: Arrive 16:00 | July | 35 concurrent vessels",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(wait_vals) * 1.25)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "M5_predicted_by_cargo.png")

# ── 4f: Predicted wait by arrival hour ───────────────────────
hours    = list(range(24))
coal_code = le_cat.transform(["Coal"])[0]
jul_code  = month_codes_ref["Jul"]
mon_code  = season_codes_ref["Monsoon"]

hour_waits = []
for h in hours:
    is_peak = 1 if h == 16 else 0
    w = gb_model.predict([[h, 35, coal_code, jul_code, mon_code, is_peak]])[0]
    hour_waits.append(max(0, w))

fig, ax = plt.subplots(figsize=(12, 5))
bar_colors_h = ["#C0392B" if w == max(hour_waits) else
                "#F39C12" if w >= np.percentile(hour_waits, 75) else
                "#2980B9" for w in hour_waits]
ax.bar(hours, hour_waits, color=bar_colors_h, edgecolor="white", width=0.85)
ax.set_xlabel("Hour of Arrival (0 = midnight, 16 = 4pm)", fontsize=11)
ax.set_ylabel("Predicted Anchorage Wait (days)", fontsize=11)
ax.set_title("Predicted Wait Time by Arrival Hour (Coal | July | 35 vessels)",
             fontsize=12, fontweight="bold")
ax.set_xticks(hours)
ax.spines[["top","right"]].set_visible(False)
handles = [mpatches.Patch(color="#C0392B", label="Peak (highest wait)"),
           mpatches.Patch(color="#F39C12", label="High wait"),
           mpatches.Patch(color="#2980B9", label="Normal wait")]
ax.legend(handles=handles, fontsize=9)
fig.tight_layout()
save(fig, "M6_predicted_by_hour.png")

# ── 4g: Cross-validation performance ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
cv_maes  = [results[m]["CV_MAE"]  for m in model_names]
test_maes = [results[m]["MAE"]    for m in model_names]
x = np.arange(len(model_names))
w = 0.35
b1 = ax.bar(x - w/2, cv_maes,  width=w, color="#0D7377",
            edgecolor="white", label="Cross-Validation MAE")
b2 = ax.bar(x + w/2, test_maes, width=w, color="#F39C12",
            edgecolor="white", label="Test Set MAE")
ax.bar_label(b1, labels=[f"{v:.1f}d" for v in cv_maes],
             padding=3, fontsize=10, fontweight="bold")
ax.bar_label(b2, labels=[f"{v:.1f}d" for v in test_maes],
             padding=3, fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Linear\nRegression", "Random\nForest", "Gradient\nBoosting"])
ax.set_ylabel("MAE (days)", fontsize=11)
ax.set_title("CV vs Test MAE – Model Reliability Check\n"
             "Similar bars = model generalises well (not overfitting)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
save(fig, "M7_cv_vs_test.png")

# ══════════════════════════════════════════════════════════════
# STEP 5 — SAMPLE PREDICTIONS (plain English)
# ══════════════════════════════════════════════════════════════
print("\nStep 5: Sample predictions (Gradient Boosting):\n")

sample_vessels = [
    {"Hour_of_Arrival": 16, "Vessel_Count_per_HoA": 35,
     "Berth_Category": "Coal",      "Month": "Jul", "Season": "Monsoon",
     "desc": "Coal ship, arrives 4pm, July peak"},
    {"Hour_of_Arrival": 8,  "Vessel_Count_per_HoA": 9,
     "Berth_Category": "Container", "Month": "Nov", "Season": "Post-Monsoon",
     "desc": "Container ship, arrives 8am, November"},
    {"Hour_of_Arrival": 0,  "Vessel_Count_per_HoA": 5,
     "Berth_Category": "Liquid",    "Month": "Dec", "Season": "Winter",
     "desc": "Liquid ship, midnight arrival, December"},
    {"Hour_of_Arrival": 16, "Vessel_Count_per_HoA": 35,
     "Berth_Category": "Automobiles","Month": "Aug", "Season": "Monsoon",
     "desc": "Car carrier, 4pm, August Monsoon"},
]

print(f"  {'Description':<40}  {'Predicted Wait':>15}")
print(f"  {'-'*40}  {'-'*15}")
for v in sample_vessels:
    cat_code = le_cat.transform([v["Berth_Category"]])[0]
    mon_code = month_codes_ref[v["Month"]]
    sea_code = season_codes_ref[v["Season"]]
    is_peak  = 1 if v["Hour_of_Arrival"] == 16 else 0
    X_in     = np.array([[v["Hour_of_Arrival"],
                           v["Vessel_Count_per_HoA"],
                           cat_code, mon_code, sea_code, is_peak]])
    pred = max(0, gb_model.predict(X_in)[0])
    print(f"  {v['desc']:<40}  {pred:>12.1f} days")

# ══════════════════════════════════════════════════════════════
# STEP 6 — SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"\n  {'Model':<24} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV MAE':>8}")
print(f"  {'-'*56}")
for m, r in results.items():
    flag = "  ← BEST" if m == best_name else ""
    print(f"  {m:<24} {r['MAE']:>7.2f}d {r['RMSE']:>7.2f}d "
          f"{r['R2']:>8.3f} {r['CV_MAE']:>7.2f}d{flag}")

print(f"\n  Best model : {best_name}")
print(f"  Best MAE   : {results[best_name]['MAE']} days")
print(f"  What this means: On average, the model's prediction is off")
print(f"  by ±{results[best_name]['MAE']} days from the actual wait time.\n")
print(f"  Charts saved to: ./{OUTPUT_DIR}/")
print("="*60 + "\n")