"""
QuantumFleet AI - Backend (FastAPI)

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /api/vessels
    POST /api/vessels
    GET  /api/voyages
    POST /api/predict-fuel
    POST /api/optimize
    POST /api/scenario
    GET  /api/explain
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import time

from database import init_db, get_db, Vessel, Voyage, PredictionLog, OptimizationRun
from ml.predict import predict_fuel, get_feature_importance
from optimization.classical import run_classical
from optimization.quantum_inspired import run_quantum_inspired
from config import FUEL_PRICE, EMISSION_FACTOR, LIFECYCLE_FACTOR

app = FastAPI(title="QuantumFleet AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


# ---------------------------------------------------------------- Schemas
class VesselIn(BaseModel):
    name: str
    type: str = "Cargo Ship"
    capacity_tons: float
    engine_type: str = "Diesel"
    fuel_type: str = "Fuel A"
    status: str = "Available"


class PredictRequest(BaseModel):
    vessel_name: str
    capacity_tons: float
    speed_knots: float
    cargo_weight_tons: float
    distance_nm: float
    weather_condition: str = "Moderate"
    fuel_type: str = "Fuel A"


class OptimizeRequest(BaseModel):
    vessel_name: str
    capacity_tons: float
    cargo_weight_tons: float
    distance_nm: float
    weather_condition: str = "Moderate"
    fuel_type: str = "Fuel A"          # current/baseline fuel choice
    current_speed_knots: float = 14
    deadline_days: float = 6


class ScenarioRequest(OptimizeRequest):
    fuel_price_change_pct: float = 0.0
    cargo_change_pct: float = 0.0
    speed_override_knots: Optional[float] = None


# ---------------------------------------------------------------- Vessels
@app.get("/api/vessels")
def list_vessels(db: Session = Depends(get_db)):
    return db.query(Vessel).all()


@app.post("/api/vessels")
def create_vessel(payload: VesselIn, db: Session = Depends(get_db)):
    v = Vessel(**payload.dict())
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


@app.get("/api/voyages")
def list_voyages(db: Session = Depends(get_db)):
    return db.query(Voyage).all()


# ---------------------------------------------------------------- Predict
@app.post("/api/predict-fuel")
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    result = predict_fuel(
        payload.vessel_name, payload.capacity_tons, payload.speed_knots,
        payload.cargo_weight_tons, payload.distance_nm,
        payload.weather_condition, payload.fuel_type,
    )
    log = PredictionLog(
        vessel_name=payload.vessel_name, speed_knots=payload.speed_knots,
        cargo_weight_tons=payload.cargo_weight_tons, distance_nm=payload.distance_nm,
        weather_condition=payload.weather_condition, fuel_type=payload.fuel_type,
        predicted_fuel_tons=result["fuel_per_day_tons"],
        confidence_low=result["confidence_low"], confidence_high=result["confidence_high"],
    )
    db.add(log)
    db.commit()
    return result


@app.get("/api/explain")
def explain():
    return {"feature_importance_pct": get_feature_importance()}


# ------------------------------------------------------------- Optimize
def _current_plan(req: OptimizeRequest):
    pred = predict_fuel(req.vessel_name, req.capacity_tons, req.current_speed_knots,
                         req.cargo_weight_tons, req.distance_nm,
                         req.weather_condition, req.fuel_type)
    fuel = pred["total_trip_fuel_tons"]
    cost = fuel * FUEL_PRICE[req.fuel_type]
    emissions = fuel * EMISSION_FACTOR[req.fuel_type] * LIFECYCLE_FACTOR[req.fuel_type]
    return {
        "algorithm": "current",
        "speed_knots": req.current_speed_knots,
        "fuel_type": req.fuel_type,
        "fuel_tons": round(fuel, 2),
        "cost": round(cost, 2),
        "emissions_tons": round(emissions, 2),
        "duration_days": pred["duration_days"],
        "schedule_met": pred["duration_days"] <= req.deadline_days,
        "runtime_seconds": 0.0,
    }


def _reasons(current, classical, quantum):
    reasons = []
    best = quantum if quantum.get("schedule_met") else classical
    if best.get("schedule_met"):
        reasons.append("Meets the required delivery schedule")
    if best["fuel_tons"] < current["fuel_tons"]:
        pct = round(100 * (current["fuel_tons"] - best["fuel_tons"]) / current["fuel_tons"], 1)
        reasons.append(f"Lower predicted fuel consumption ({pct}% less than current plan)")
    if best["cost"] < current["cost"]:
        reasons.append("Lower estimated operating cost")
    if best["emissions_tons"] < current["emissions_tons"]:
        reasons.append("Lower estimated GHG emissions")
    if best["fuel_type"] != current["fuel_type"]:
        reasons.append(f"Switches fuel to {best['fuel_type']} for a better fuel/cost/emissions balance")
    return reasons


@app.post("/api/optimize")
def optimize(req: OptimizeRequest, db: Session = Depends(get_db)):
    current = _current_plan(req)

    classical = run_classical(
        req.vessel_name, req.capacity_tons, req.cargo_weight_tons, req.distance_nm,
        req.weather_condition, req.fuel_type, req.deadline_days,
        FUEL_PRICE[req.fuel_type], EMISSION_FACTOR[req.fuel_type], LIFECYCLE_FACTOR[req.fuel_type],
    )
    quantum = run_quantum_inspired(
        req.vessel_name, req.capacity_tons, req.cargo_weight_tons, req.distance_nm,
        req.weather_condition, req.deadline_days,
    )

    for algo, plan in [("current", current), ("classical", classical), ("quantum_inspired", quantum)]:
        if plan.get("status") == "infeasible":
            continue
        db.add(OptimizationRun(
            algorithm=algo, fuel_tons=plan["fuel_tons"], cost=plan["cost"],
            emissions_tons=plan["emissions_tons"], runtime_seconds=plan["runtime_seconds"],
            schedule_met=str(plan["schedule_met"]),
        ))
    db.commit()

    recommended = quantum if quantum.get("schedule_met", False) or classical.get("status") == "infeasible" else classical
    reasons = _reasons(current, classical, quantum)

    return {
        "current": current,
        "classical": classical,
        "quantum_inspired": quantum,
        "recommended": recommended,
        "why_recommended": reasons,
    }


# ------------------------------------------------------------- Scenario
@app.post("/api/scenario")
def scenario(req: ScenarioRequest, db: Session = Depends(get_db)):
    """What-if simulator: tweak fuel price / cargo / speed and recompute."""
    adjusted_cargo = req.cargo_weight_tons * (1 + req.cargo_change_pct / 100)
    speed = req.speed_override_knots or req.current_speed_knots

    pred = predict_fuel(req.vessel_name, req.capacity_tons, speed, adjusted_cargo,
                         req.distance_nm, req.weather_condition, req.fuel_type)

    adjusted_price = FUEL_PRICE[req.fuel_type] * (1 + req.fuel_price_change_pct / 100)
    fuel = pred["total_trip_fuel_tons"]
    cost = fuel * adjusted_price
    emissions = fuel * EMISSION_FACTOR[req.fuel_type] * LIFECYCLE_FACTOR[req.fuel_type]

    baseline = _current_plan(req)

    return {
        "baseline": baseline,
        "scenario_result": {
            "speed_knots": speed,
            "cargo_weight_tons": round(adjusted_cargo, 1),
            "fuel_tons": round(fuel, 2),
            "cost": round(cost, 2),
            "emissions_tons": round(emissions, 2),
            "duration_days": pred["duration_days"],
            "schedule_met": pred["duration_days"] <= req.deadline_days,
        },
        "delta": {
            "fuel_pct": round(100 * (fuel - baseline["fuel_tons"]) / baseline["fuel_tons"], 1),
            "cost_pct": round(100 * (cost - baseline["cost"]) / baseline["cost"], 1),
            "emissions_pct": round(100 * (emissions - baseline["emissions_tons"]) / baseline["emissions_tons"], 1),
        },
    }


@app.get("/")
def root():
    return {"status": "QuantumFleet AI backend running", "docs": "/docs"}
