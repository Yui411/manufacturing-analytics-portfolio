# Databricks notebook source
# ============================================================
# 03_gold_kpi.py
# Layer: Gold — Daily KPI Aggregation
# Input:  factory_portfolio.silver_sensor_clean
# Output: factory_portfolio.gold_daily_kpi
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# ── Validation ──────────────────────────────────────────────
assert spark.catalog.tableExists("factory_portfolio.silver_sensor_clean"), \
    "Please run 02_silver_etl first."

print("OK — silver table confirmed")

# COMMAND ----------

# ── Load Silver ──────────────────────────────────────────────
silver = spark.table("factory_portfolio.silver_sensor_clean")

# ── Daily KPI Aggregation ────────────────────────────────────
daily_kpi = (silver
    .withColumn("date", F.to_date("timestamp"))
    .groupBy("date", "line_id")
    .agg(
        F.count("*")                                       .alias("total_units"),
        F.sum("defect_flag")                               .alias("defect_count"),
        (F.sum("defect_flag") / F.count("*") * 100)       .alias("defect_rate_pct"),
        (1 - F.sum("defect_flag") / F.count("*"))         .alias("yield_rate"),
        F.avg("temperature")                               .alias("avg_temp"),
        F.avg("pressure")                                  .alias("avg_pressure"),
        F.avg("vibration")                                 .alias("avg_vibration"),
        F.avg("speed_rpm")                                 .alias("avg_speed_rpm"),
        F.max("temperature")                               .alias("max_temp"),
        F.min("temperature")                               .alias("min_temp"),
    )
    .orderBy("date", "line_id")
)

# ── Day-over-day defect rate change ─────────────────────────
window_lag = Window.partitionBy("line_id").orderBy("date")

gold = daily_kpi.withColumn(
    "defect_rate_change",
    F.col("defect_rate_pct") - F.lag("defect_rate_pct").over(window_lag)
)

# ── Save as Gold Delta Table ─────────────────────────────────
gold.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("factory_portfolio.gold_daily_kpi")

print(f"Gold table saved. Row count: {gold.count():,}")

# ── Quick check ──────────────────────────────────────────────
display(spark.table("factory_portfolio.gold_daily_kpi").limit(10))