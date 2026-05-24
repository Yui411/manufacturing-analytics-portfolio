# Databricks notebook source
# ============================================================
# 01_bronze_data_generation.py  (v — stronger correlation)
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 5000  # 5000レコード

dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
df = pd.DataFrame({
    "timestamp":     dates,
    "line_id":       np.random.choice(["LineA", "LineB", "LineC"], n),
    "temperature":   np.random.normal(75, 10, n),       # 摂氏
    "pressure":      np.random.normal(3.0, 0.5, n),     # MPa
    "vibration":     np.random.exponential(0.3, n),     # mm/s
    "speed_rpm":     np.random.normal(1200, 100, n),
    "operator_id":   np.random.randint(1, 20, n),
    "defect_flag":   np.random.binomial(1, 0.08, n),    # 不良率8%
})

# Spark-->Delta
# spark_df = spark.createDataFrame(df)
# spark_df.write.format("delta").mode("overwrite") \
#     .save("/tmp/factory/bronze/sensor_raw")

spark.sql("CREATE DATABASE IF NOT EXISTS factory_portfolio")
spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("factory_portfolio.bronze_sensor_raw")

# COMMAND ----------

# ============================================================
# 01_bronze_data_generation.py  (v2 — realistic defect logic)
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 5000

dates      = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
line_id    = np.random.choice(["LineA", "LineB", "LineC"], n)
temperature= np.random.normal(75, 10, n)
pressure   = np.random.normal(3.0, 0.5, n)
vibration  = np.random.exponential(0.3, n)
speed_rpm  = np.random.normal(1200, 100, n)
operator_id= np.random.randint(1, 20, n)

# ── Realistic defect logic ───────────────────────────────────
# Base defect probability: 5%
# +15% if temperature > 85
# +10% if vibration > 0.6
# +10% if pressure > 3.5
defect_prob = (
    0.05
    + 0.15 * (temperature > 85).astype(float)
    + 0.10 * (vibration > 0.6).astype(float)
    + 0.10 * (pressure > 3.5).astype(float)
)
defect_prob = np.clip(defect_prob, 0, 1)
defect_flag = np.random.binomial(1, defect_prob, n)

df = pd.DataFrame({
    "timestamp":   dates,
    "line_id":     line_id,
    "temperature": temperature,
    "pressure":    pressure,
    "vibration":   vibration,
    "speed_rpm":   speed_rpm,
    "operator_id": operator_id,
    "defect_flag": defect_flag,
})

print(f"Defect rate: {defect_flag.mean():.2%}")
print(f"Defect count: {defect_flag.sum()}")

spark_df = spark.createDataFrame(df)
spark.sql("CREATE DATABASE IF NOT EXISTS factory_portfolio")
spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("factory_portfolio.bronze_sensor_raw")

print(f"Bronze table saved. Row count: {spark_df.count():,}")

# COMMAND ----------

# ============================================================
# 01_bronze_data_generation.py  (v3 — stronger correlation)
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 5000

dates      = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
line_id    = np.random.choice(["LineA", "LineB", "LineC"], n)
temperature= np.random.normal(75, 12, n)
pressure   = np.random.normal(3.0, 0.6, n)
vibration  = np.random.exponential(0.3, n)
speed_rpm  = np.random.normal(1200, 120, n)
operator_id= np.random.randint(1, 20, n)

# ── Stronger defect logic ────────────────────────────────────
# Each condition contributes clearly to defect probability
temp_score  = np.clip((temperature - 75) / 20, 0, 1)   # rises above 75
press_score = np.clip((pressure - 3.0) / 2.0, 0, 1)    # rises above 3.0
vib_score   = np.clip((vibration - 0.3) / 1.0, 0, 1)   # rises above 0.3
speed_score = np.clip((speed_rpm - 1200) / 400, 0, 1)  # rises above 1200

defect_prob = (
    0.03
    + 0.35 * temp_score
    + 0.25 * press_score
    + 0.20 * vib_score
    + 0.10 * speed_score
)
defect_prob = np.clip(defect_prob, 0, 0.95)
defect_flag = np.random.binomial(1, defect_prob, n)

df = pd.DataFrame({
    "timestamp":   dates,
    "line_id":     line_id,
    "temperature": temperature,
    "pressure":    pressure,
    "vibration":   vibration,
    "speed_rpm":   speed_rpm,
    "operator_id": operator_id,
    "defect_flag": defect_flag,
})

print(f"Defect rate: {defect_flag.mean():.2%}")

spark_df = spark.createDataFrame(df)
spark.sql("CREATE DATABASE IF NOT EXISTS factory_portfolio")
spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("factory_portfolio.bronze_sensor_raw")

print(f"Bronze table saved. Row count: {spark_df.count():,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN factory_portfolio;
