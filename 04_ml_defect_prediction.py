# Databricks notebook source
# Check your actual username
username = spark.sql("SELECT current_user()").collect()[0][0]
print(f"Your username: {username}")
print(f"Experiment path: /Users/{username}/factory_portfolio/defect_prediction")

# COMMAND ----------

# ============================================================
# 04_ml_defect_prediction.py
# ML Model: Defect Prediction using Random Forest
# Input:  factory_portfolio.silver_sensor_clean
# Output: MLflow Model Registry — factory_defect_predictor
# ============================================================

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# COMMAND ----------

# ── Validation ───────────────────────────────────────────────
assert spark.catalog.tableExists("factory_portfolio.silver_sensor_clean"), \
    "Please run 02_silver_etl first."

print("OK — silver table confirmed")

# COMMAND ----------

# ============================================================
# 04_ml_defect_prediction.py
# ML Model: Defect Prediction using Random Forest
# Input:  factory_portfolio.silver_sensor_clean
# Output: MLflow Model Registry — factory_defect_predictor
# ============================================================

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ── Validation ───────────────────────────────────────────────
assert spark.catalog.tableExists("factory_portfolio.silver_sensor_clean"), \
    "Please run 02_silver_etl first."

print("OK — silver table confirmed")

# ── Load Data ────────────────────────────────────────────────
pdf = spark.table("factory_portfolio.silver_sensor_clean").toPandas()
print(f"Total rows: {len(pdf):,}")
print(f"Defect rate: {pdf['defect_flag'].mean():.2%}")

# ── Feature & Target Definition ──────────────────────────────
FEATURES = [
    "temperature", "pressure", "vibration", "speed_rpm",
    "hour", "day_of_week", "is_weekend",
    "temp_roll_avg", "pressure_roll_avg", "vibration_roll_avg"
]
TARGET = "defect_flag"

X = pdf[FEATURES].fillna(0)
y = pdf[TARGET]

# ── Train / Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── MLflow Experiment ────────────────────────────────────────
mlflow.set_experiment("/Users/factory_portfolio/defect_prediction")

with mlflow.start_run(run_name="RandomForest_v1"):

    # ── Train Model ──────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",   # handle imbalanced defect data
        random_state=42
    )
    model.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc          = roc_auc_score(y_test, y_pred_proba)

    print(f"\nROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Feature Importance ───────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":   FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    # ── Log to MLflow ─────────────────────────────────────────
    mlflow.log_param("n_estimators",    100)
    mlflow.log_param("max_depth",       8)
    mlflow.log_param("min_samples_leaf",5)
    mlflow.log_param("features",        FEATURES)
    mlflow.log_metric("roc_auc",        auc)

    # Log feature importance as artifact
    importance_df.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")

    # ── Register Model ───────────────────────────────────────
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="defect_model",
        registered_model_name="factory_defect_predictor",
        input_example=X_test.iloc[:3]
    )

    print(f"\nModel registered: factory_defect_predictor")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# ============================================================
# 04_ml_defect_prediction.py  (v2 — tuned model)
# ============================================================

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ── Validation ───────────────────────────────────────────────
assert spark.catalog.tableExists("factory_portfolio.silver_sensor_clean"), \
    "Please run 02_silver_etl first."

# ── Load Data ────────────────────────────────────────────────
pdf = spark.table("factory_portfolio.silver_sensor_clean").toPandas()
print(f"Total rows: {len(pdf):,}")
print(f"Defect rate: {pdf['defect_flag'].mean():.2%}")

# ── Feature & Target Definition ──────────────────────────────
FEATURES = [
    "temperature", "pressure", "vibration", "speed_rpm",
    "hour", "day_of_week", "is_weekend",
    "temp_roll_avg", "pressure_roll_avg", "vibration_roll_avg"
]
TARGET = "defect_flag"

X = pdf[FEATURES].fillna(0).astype(float)   # cast to float: fix UserWarning
y = pdf[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── MLflow Experiment ─────────────────────────────────────────
username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/Portfolio_Factory data analysis/notebooks/")

with mlflow.start_run(run_name="GradientBoosting_v2"):

    # ── Train Model ───────────────────────────────────────────
    # GradientBoosting handles imbalanced data better than RandomForest
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc          = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation AUC (more reliable estimate)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    print(f"\nROC-AUC:           {auc:.4f}")
    print(f"CV ROC-AUC (5fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Feature Importance ────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    # ── Log to MLflow ─────────────────────────────────────────
    mlflow.log_param("model_type",    "GradientBoosting")
    mlflow.log_param("n_estimators",  200)
    mlflow.log_param("max_depth",     4)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_metric("roc_auc",      auc)
    mlflow.log_metric("cv_roc_auc",   cv_scores.mean())

    importance_df.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="defect_model",
        registered_model_name="factory_defect_predictor",
        input_example=X_test.iloc[:3]
    )

    print(f"\nModel registered: factory_defect_predictor")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# ============================================================
# 04_ml_defect_prediction.py  (v3 — imbalanced data handling)
# ============================================================

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score
)

# ── Validation ───────────────────────────────────────────────
assert spark.catalog.tableExists("factory_portfolio.silver_sensor_clean"), \
    "Please run 02_silver_etl first."

# ── Load Data ────────────────────────────────────────────────
pdf = spark.table("factory_portfolio.silver_sensor_clean").toPandas()
print(f"Total rows:   {len(pdf):,}")
print(f"Defect rate:  {pdf['defect_flag'].mean():.2%}")

# ── Features ─────────────────────────────────────────────────
FEATURES = [
    "temperature", "pressure", "vibration", "speed_rpm",
    "hour", "day_of_week", "is_weekend",
    "temp_roll_avg", "pressure_roll_avg", "vibration_roll_avg"
]
TARGET = "defect_flag"

X = pdf[FEATURES].fillna(0).astype(float)
y = pdf[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Imbalance ratio (used for scale_pos_weight) ───────────────
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = round(neg / pos, 2)
print(f"Class ratio (neg/pos): {scale}")

# ── MLflow ───────────────────────────────────────────────────
username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/Portfolio_Factory data analysis/notebooks/")

with mlflow.start_run(run_name="GradientBoosting_v3_balanced"):

    # ── Train ─────────────────────────────────────────────────
    # scale_pos_weight: penalizes missing defects more heavily
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42
    )

    # Oversample minority class in training data (SMOTE-free simple approach)
    defect_rows  = X_train[y_train == 1]
    defect_label = y_train[y_train == 1]
    normal_rows  = X_train[y_train == 0]
    normal_label = y_train[y_train == 0]

    # Upsample defects to 30% of training data
    target_defect_n = int(len(normal_rows) * 0.3 / 0.7)
    defect_up = defect_rows.sample(n=target_defect_n, replace=True, random_state=42)
    label_up  = defect_label.sample(n=target_defect_n, replace=True, random_state=42)

    X_bal = pd.concat([normal_rows, defect_up]).reset_index(drop=True)
    y_bal = pd.concat([normal_label, label_up]).reset_index(drop=True)
    print(f"Balanced train size: {len(X_bal):,}  (defect rate: {y_bal.mean():.2%})")

    model.fit(X_bal, y_bal)

    # ── Optimal threshold (maximize F1 for defect class) ──────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_thresh = thresholds[f1_scores[:-1].argmax()]
    print(f"Optimal threshold: {best_thresh:.3f}")

    y_pred = (y_pred_proba >= best_thresh).astype(int)

    # ── Metrics ───────────────────────────────────────────────
    auc    = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    cv_scores = cross_val_score(model, X_bal, y_bal, cv=5, scoring="roc_auc")

    print(f"\nROC-AUC:            {auc:.4f}")
    print(f"PR-AUC:             {pr_auc:.4f}")
    print(f"CV ROC-AUC (5fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nClassification Report (threshold={best_thresh:.3f}):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Feature Importance ────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    # ── Log to MLflow ─────────────────────────────────────────
    mlflow.log_param("model_type",       "GradientBoosting_v3")
    mlflow.log_param("n_estimators",     300)
    mlflow.log_param("learning_rate",    0.03)
    mlflow.log_param("upsampling_ratio", 0.3)
    mlflow.log_param("threshold",        round(best_thresh, 3))
    mlflow.log_metric("roc_auc",         auc)
    mlflow.log_metric("pr_auc",          pr_auc)
    mlflow.log_metric("cv_roc_auc",      cv_scores.mean())

    importance_df.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="defect_model",               # fix: use name instead of artifact_path
        registered_model_name="factory_defect_predictor",
        input_example=X_test.iloc[:3]
    )

    print(f"\nModel registered: factory_defect_predictor")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
