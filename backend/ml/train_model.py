"""
QuantumFleet AI - Train the fuel prediction model (XGBoost).

Predicts fuel_per_day_tons from: speed, cargo_ratio, distance, weather,
fuel_type, vessel efficiency profile.

Also stores:
- feature importances (for the "Why this prediction?" explainability page)
- residual std (used to build a simple confidence interval band)
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "synthetic_voyages.csv")
MODEL_PATH = os.path.join(HERE, "..", "data", "fuel_model.joblib")
META_PATH = os.path.join(HERE, "..", "data", "model_meta.json")

NUMERIC_FEATURES = ["speed_knots", "cargo_ratio", "distance_nm", "capacity_tons"]
CATEGORICAL_FEATURES = ["weather_condition", "fuel_type", "vessel_name"]
TARGET = "fuel_per_day_tons"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    residual_std = float(np.std(y_test.values - preds))

    print(f"Test MAE: {mae:.3f} tons/day")
    print(f"Test R^2: {r2:.4f}")
    print(f"Residual std (for confidence band): {residual_std:.3f}")

    # Feature importance -> map one-hot cols back to readable groups
    ohe = pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_cols = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_cols = NUMERIC_FEATURES + cat_cols
    importances = pipeline.named_steps["model"].feature_importances_

    grouped = {"speed_knots": 0.0, "cargo_ratio": 0.0, "distance_nm": 0.0,
               "capacity_tons": 0.0, "weather_condition": 0.0,
               "fuel_type": 0.0, "vessel_name": 0.0}
    for col, imp in zip(all_cols, importances):
        matched = False
        for base in CATEGORICAL_FEATURES:
            if col.startswith(base):
                grouped[base] += float(imp)
                matched = True
                break
        if not matched:
            grouped[col] += float(imp)

    total = sum(grouped.values()) or 1.0
    grouped_pct = {k: round(100 * v / total, 1) for k, v in grouped.items()}

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        "mae": mae,
        "r2": r2,
        "residual_std": residual_std,
        "feature_importance_pct": grouped_pct,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved model  ->", MODEL_PATH)
    print("Saved meta   ->", META_PATH)
    print("Feature importance:", grouped_pct)


if __name__ == "__main__":
    main()
