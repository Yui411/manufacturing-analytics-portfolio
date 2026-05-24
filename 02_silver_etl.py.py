# Databricks notebook source
# ============================================================
# 02_silver_etl.py
# Layer: Silver — Data Cleansing & Feature Engineering
# Input:  factory_portfolio.bronze_sensor_raw
# Output: factory_portfolio.silver_sensor_clean
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------


# ── Validation ──────────────────────────────────────────────
assert spark.catalog.tableExists("factory_portfolio.bronze_sensor_raw"), \
    "Please run 01_bronze_data_generation first."

print("OK — bronze table confirmed")


# COMMAND ----------

# ── Load Bronze ──────────────────────────────────────────────
bronze = spark.table("factory_portfolio.bronze_sensor_raw")
print(f"Bronze row count: {bronze.count():,}")

# ── Cleansing: remove outliers ───────────────────────────────
cleaned = bronze.filter(
    (F.col("temperature").between(20, 150)) &
    (F.col("pressure").between(0.5, 8.0))  &
    (F.col("vibration") >= 0)              &
    (F.col("speed_rpm").between(500, 2000))
)
print(f"After outlier removal: {cleaned.count():,}")

# ── Feature Engineering: time-based features ────────────────
featured = (cleaned
    .withColumn("hour",        F.hour("timestamp"))
    .withColumn("day_of_week", F.dayofweek("timestamp"))  # 1=Sun, 7=Sat
    .withColumn("is_weekend",
        F.when(F.dayofweek("timestamp").isin([1, 7]), 1).otherwise(0))
)

# ── Feature Engineering: rolling average (past 12 hours) ────
window_12h = (Window
    .partitionBy("line_id")
    .orderBy(F.col("timestamp").cast("long"))
    .rangeBetween(-43200, 0)   # 12 hours in seconds
)

silver = (featured
    .withColumn("temp_roll_avg",     F.avg("temperature").over(window_12h))
    .withColumn("pressure_roll_avg", F.avg("pressure").over(window_12h))
    .withColumn("vibration_roll_avg",F.avg("vibration").over(window_12h))
)

# ── Save as Silver Delta Table ───────────────────────────────
silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("factory_portfolio.silver_sensor_clean")

print(f"Silver table saved. Row count: {silver.count():,}")

# ── Quick check ──────────────────────────────────────────────
display(spark.table("factory_portfolio.silver_sensor_clean").limit(5))

# COMMAND ----------

bronze.printSchema()