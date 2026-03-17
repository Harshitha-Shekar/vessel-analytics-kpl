# Vessel Timings Analytics – Kamarajar Port Limited (KPL) 2024

## Project Overview
Data analytics project for KPL vessel timing dataset covering **July – December 2024**
(416 vessel calls across 13 terminals, 6 cargo categories).

## Setup

### Requirements
```
pandas
matplotlib
numpy
openpyxl
```

Install with:
```bash
pip install pandas matplotlib numpy openpyxl
```

## Run
```bash
python vessel_analytics.py
```
Charts are saved to the `charts/` folder automatically.

---

## Dataset Columns

| Column | Description |
|---|---|
| Vessel_ID | IMO/MMSI vessel identifier |
| Berth | Terminal name |
| Berth_Category | Cargo type (Coal, Liquid, Container, etc.) |
| Berth_Entry / Berth_Exit | Timestamp of berth occupancy |
| Port_Entry / Port_Exit | Timestamp of port entry/exit |
| Anchorage_Entry / Anchorage_Exit | Anchorage waiting timestamps (NaN if not applicable) |
| Berth_Duration | Time at berth (hours) |
| Port_Duration | Total port stay (hours) |
| Anchorage_Duration | Wait time at anchorage (hours) |
| Hour_of_Arrival | Hour (0–23) vessel entered port |
| Day_of_Arrival | Day of week of arrival |
| Month | Month abbreviation |
| Season | Monsoon / Post-Monsoon / Winter |

---

## Generated Charts

| File | Description |
|---|---|
| `00_kpi_summary.png` | KPI dashboard tile summary |
| `01_monthly_vessel_count.png` | Vessel traffic by month |
| `02_cargo_type_distribution.png` | Pie chart of cargo mix |
| `03_avg_berth_duration_category.png` | Avg berth time by cargo type |
| `04_avg_berth_duration_terminal.png` | Avg berth time by terminal |
| `05_seasonal_analysis.png` | Seasonal traffic + duration comparison |
| `06_hour_of_arrival.png` | Arrival pattern by hour of day |
| `07_day_of_arrival.png` | Arrivals by day of week |
| `08_anchorage_vs_berth_scatter.png` | Anchorage vs berth wait correlation |
| `09_port_duration_boxplot.png` | Port duration spread by cargo type |
| `10_monthly_berth_duration_stacked.png` | Monthly berth hours stacked by cargo |

---

## Key Findings

- **416 vessels** recorded, Jul–Dec 2024
- **72% of vessels** required anchorage before berthing
- **July** was the busiest month (136 vessels – Monsoon peak)
- **16:00** is the most common arrival hour (117 vessels)
- **Coal** dominates with 35.1% of total calls
- **Tuesday** sees anomalously high arrivals (169 out of 416)
- **Coal Berths 1–4** have the longest average berth durations (2000–2700 hrs)
- Traffic drops sharply Post-Monsoon → Winter (seasonal congestion pattern)
