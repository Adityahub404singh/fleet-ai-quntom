"""
Classical baseline optimizer.

Strategy: GREEDY heuristic - among a small set of "sensible" speed choices,
pick the LOWEST speed that still meets the deadline (lower speed => lower
fuel under the cubic law), using the fixed fuel type given in the scenario.
This deliberately does NOT search fuel-type or vessel choice - that's what
makes it a fair, simple baseline to compare the quantum-inspired optimizer
against.
"""
import time
from ml.predict import predict_fuel

# A handful of discrete speed steps a real operator would actually consider
CANDIDATE_SPEEDS = [10, 12, 14, 16, 18, 20]


def run_classical(vessel_name, capacity_tons, cargo_weight_tons, distance_nm,
                   weather_condition, fuel_type, deadline_days,
                   fuel_price, emission_factor, lifecycle_factor):
    start = time.time()

    feasible = []
    for speed in CANDIDATE_SPEEDS:
        pred = predict_fuel(vessel_name, capacity_tons, speed, cargo_weight_tons,
                             distance_nm, weather_condition, fuel_type)
        if pred["duration_days"] <= deadline_days:
            cost = pred["total_trip_fuel_tons"] * fuel_price
            emissions = pred["total_trip_fuel_tons"] * emission_factor * lifecycle_factor
            feasible.append({
                "speed_knots": speed,
                "fuel_type": fuel_type,
                "fuel_tons": pred["total_trip_fuel_tons"],
                "cost": round(cost, 2),
                "emissions_tons": round(emissions, 2),
                "duration_days": pred["duration_days"],
                "schedule_met": True,
            })

    runtime = time.time() - start

    if not feasible:
        return {"status": "infeasible", "runtime_seconds": round(runtime, 4)}

    # Greedy rule: lowest speed that meets deadline = plan already sorted by speed asc
    best = min(feasible, key=lambda p: p["speed_knots"])
    best["algorithm"] = "classical"
    best["runtime_seconds"] = round(runtime, 4)
    return best
