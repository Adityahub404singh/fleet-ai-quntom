"""Run once: python3 seed_data.py -- populates sample vessels & voyages."""
from database import init_db, SessionLocal, Vessel, Voyage

init_db()
db = SessionLocal()

if db.query(Vessel).count() == 0:
    vessels = [
        Vessel(name="Vessel A", type="Bulk Carrier", capacity_tons=5000, engine_type="Diesel", fuel_type="Fuel A", status="Available"),
        Vessel(name="Vessel B", type="Container Ship", capacity_tons=7000, engine_type="Diesel", fuel_type="Fuel B", status="Active"),
        Vessel(name="Vessel C", type="Tanker", capacity_tons=4000, engine_type="Diesel", fuel_type="Fuel A", status="Available"),
        Vessel(name="Vessel D", type="Bulk Carrier", capacity_tons=9000, engine_type="Dual-Fuel", fuel_type="Green Fuel", status="Maintenance"),
    ]
    db.add_all(vessels)
    db.commit()
    print(f"Seeded {len(vessels)} vessels")

if db.query(Voyage).count() == 0:
    va = db.query(Vessel).filter_by(name="Vessel A").first()
    vb = db.query(Vessel).filter_by(name="Vessel B").first()
    voyages = [
        Voyage(vessel_id=va.id, origin="Mumbai", destination="Colombo", distance_nm=780,
               planned_speed_knots=14, cargo_weight_tons=4000, deadline_days=3,
               weather_condition="Moderate", fuel_type="Fuel A"),
        Voyage(vessel_id=vb.id, origin="Chennai", destination="Singapore", distance_nm=1600,
               planned_speed_knots=16, cargo_weight_tons=6000, deadline_days=6,
               weather_condition="Rough", fuel_type="Fuel B"),
    ]
    db.add_all(voyages)
    db.commit()
    print(f"Seeded {len(voyages)} voyages")

db.close()
print("Seed complete.")
