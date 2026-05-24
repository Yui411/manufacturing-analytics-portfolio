# Manufacturing Factory Analytics Portfolio

A manufacturing factory analytics project built on Databricks and Tableau.  
This project covers the full data pipeline — from data generation to ML-based defect prediction and visualization.

---

## Architecture

| Service | Role |
| --- | --- |
| Databricks Community Edition | Notebook, Delta Lake, MLflow |
| Delta Lake (Bronze) | Raw sensor data ingestion |
| Delta Lake (Silver) | Data cleansing & feature engineering |
| Delta Lake (Gold) | Daily KPI aggregation |
| MLflow | Experiment tracking & model registry |
| Tableau Desktop | Dashboard and reporting |

---

## Tech Stack

- **Platform**: Databricks Community Edition
- **Visualization**: Tableau Desktop
- **Language**: Python 3.12, SQL
- **Libraries**: PySpark, pandas, numpy, scikit-learn, MLflow

---

## Dataset

Synthetic manufacturing sensor data with realistic defect logic.

| Column | Description |
| --- | --- |
| timestamp | Hourly timestamp (2024-01-01 to 2024-07-29) |
| line_id | Production line (LineA / LineB / LineC) |
| temperature | Sensor temperature (°C) |
| pressure | Pressure (MPa) |
| vibration | Vibration level (mm/s) |
| speed_rpm | Motor speed (RPM) |
| operator_id | Operator ID |
| defect_flag | Defect occurrence (1=defect, 0=normal) |

**Defect logic:** Defect probability increases with higher temperature, pressure, vibration, and speed — simulating realistic manufacturing conditions.

```python
defect_prob = (
    0.03
    + 0.35 * temp_score    # strongest driver
    + 0.25 * press_score
    + 0.20 * vib_score
    + 0.10 * speed_score
)
```

---

## Medallion Architecture

```
Raw Sensor Data
      │
      ▼
┌─────────────┐
│   Bronze    │  Raw data ingestion via Delta Lake
│  Delta Lake │  5,000 records — temperature, pressure,
│             │  vibration, speed, defect flag
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Silver    │  Data cleansing & feature engineering
│  Delta Lake │  Outlier removal, rolling averages (12h),
│             │  time-based features
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Gold     │  Daily KPI aggregation per production line
│  Delta Lake │  Defect rate, yield rate, day-over-day change
└──────┬──────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌─────────────┐      ┌──────────────────┐
│   Tableau   │      │ MLflow + sklearn  │
│  Dashboard  │      │ Defect Prediction │
└─────────────┘      └──────────────────┘
```

---

## Notebooks

| Notebook | Description |
| --- | --- |
| 01_bronze_data_generation.py | Generates synthetic sensor data and saves as Delta table |
| 02_silver_etl.py | Outlier removal, time features, 12h rolling averages |
| 03_gold_kpi.py | Daily KPI aggregation: defect rate, yield rate, day-over-day change |
| 04_ml_defect_prediction.py | Gradient Boosting classifier with MLflow tracking |

---

## ML Model Results

| Metric | Value |
| --- | --- |
| Model | Gradient Boosting Classifier |
| ROC-AUC | 0.68 |
| CV ROC-AUC (5-fold) | 0.86 ± 0.01 |
| PR-AUC | 0.31 |
| Defect Recall | 55% |
| Defect F1 | 0.38 |
| Optimal Threshold | 0.269 |

### Top Features by Importance

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | temperature | 0.348 |
| 2 | pressure | 0.149 |
| 3 | vibration_roll_avg | 0.109 |
| 4 | vibration | 0.102 |
| 5 | speed_rpm | 0.086 |

---

## Dashboard

[View on Tableau Public](https://public.tableau.com/app/profile/bito.yui/viz/Factorydataanalysis/ManufacturingFactoryDashboard?publish=yes)

Built with Tableau Desktop, connected directly to Databricks via Personal Access Token.

### Views

| View | Chart Type | Description |
| --- | --- | --- |
| Daily Defect Rate Trend | Line Chart | Defect rate over time per production line |
| Defect Rate by Line | Bar Chart | Average defect rate comparison across lines |
| Yield Rate Heatmap | Heatmap | Yield rate by date and production line |
| Temperature vs Defect Rate | Scatter Plot | Correlation between temperature and defect rate |

---

## How to Run

### 1. Run Databricks notebooks in order

```
01_bronze_data_generation.py
02_silver_etl.py
03_gold_kpi.py
04_ml_defect_prediction.py
```

### 2. Connect Tableau Desktop to Databricks

```
Connect → To a Server → Databricks
Server:    <your-workspace>.cloud.databricks.com
HTTP Path: <from Compute → Advanced Options → JDBC/ODBC>
Auth:      Personal Access Token
```

### 3. Select table in Tableau

```
Catalog:  workspace
Database: factory_portfolio
Table:    gold_daily_kpi
```

### 4. Open dashboard and refresh data source

---

## Project Structure

```
databricks-manufacturing-portfolio/
│
├── README.md
├── notebooks/
│   ├── 01_bronze_data_generation.py
│   ├── 02_silver_etl.py
│   ├── 03_gold_kpi.py
│   └── 04_ml_defect_prediction.py
└── docs/
    └── architecture.png
```

---

## Notes

- Sample data is synthetically generated within the notebook.  
  Run `01_bronze_data_generation.py` to recreate it.
- Databricks credentials are managed via Personal Access Token.  
  Never commit tokens or connection strings to the repository.
