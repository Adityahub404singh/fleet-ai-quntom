import os
import json
import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "data", "fuel_model.joblib")
META_PATH = os.path.join(HERE, "..", "data", "model_meta.json")

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


def predict_fuel(vessel_name: str, capacity_tons: float, speed_knots: float,
                  cargo_weight_tons: float, distance_nm: float,
                  weather_condition: str, fuel_type: str):
    model, meta = _load()
    cargo_ratio = min(max(cargo_weight_tons / capacity_tons, 0.05), 1.2)

    row = pd.DataFrame([{
        "speed_knots": speed_knots,
        "cargo_ratio": cargo_ratio,
        "distance_nm": distance_nm,
        "capacity_tons": capacity_tons,
        "weather_condition": weather_condition,
        "fuel_type": fuel_type,
        "vessel_name": vessel_name,
    }])

    fuel_per_day = float(model.predict(row)[0])
    std = meta["residual_std"]

    duration_days = distance_nm / (speed_knots * 24)
    total_trip_fuel = fuel_per_day * duration_days

    return {
        "fuel_per_day_tons": round(fuel_per_day, 2),
        "confidence_low": round(max(fuel_per_day - std, 0), 2),
        "confidence_high": round(fuel_per_day + std, 2),
        "duration_days": round(duration_days, 2),
        "total_trip_fuel_tons": round(total_trip_fuel, 2),
        "feature_importance_pct": meta["feature_importance_pct"],
    }


def get_feature_importance():
    _, meta = _load()
    return meta["feature_importance_pct"]
