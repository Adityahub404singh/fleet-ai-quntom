"""
QuantumFleet AI - Database layer
SQLite for prototype/demo (swap DATABASE_URL for Postgres in production
without changing any other code, since SQLAlchemy handles both).
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'quantumfleet.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Vessel(Base):
    __tablename__ = "vessels"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, default="Cargo Ship")
    capacity_tons = Column(Float, nullable=False)
    engine_type = Column(String, default="Diesel")
    fuel_type = Column(String, default="Fuel A")
    status = Column(String, default="Available")  # Available / Active / Maintenance

    voyages = relationship("Voyage", back_populates="vessel")


class FuelType(Base):
    __tablename__ = "fuel_types"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    price_per_ton = Column(Float, nullable=False)       # currency / ton
    emission_factor = Column(Float, nullable=False)     # tCO2 / ton fuel
    lifecycle_factor = Column(Float, default=1.0)        # well-to-wake multiplier


class Voyage(Base):
    __tablename__ = "voyages"
    id = Column(Integer, primary_key=True, index=True)
    vessel_id = Column(Integer, ForeignKey("vessels.id"))
    origin = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    distance_nm = Column(Float, nullable=False)          # nautical miles
    planned_speed_knots = Column(Float, nullable=False)
    cargo_weight_tons = Column(Float, nullable=False)
    deadline_days = Column(Float, nullable=False)
    weather_condition = Column(String, default="Moderate")  # Calm / Moderate / Rough
    fuel_type = Column(String, default="Fuel A")

    vessel = relationship("Vessel", back_populates="voyages")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    vessel_name = Column(String)
    speed_knots = Column(Float)
    cargo_weight_tons = Column(Float)
    distance_nm = Column(Float)
    weather_condition = Column(String)
    fuel_type = Column(String)
    predicted_fuel_tons = Column(Float)
    confidence_low = Column(Float)
    confidence_high = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    id = Column(Integer, primary_key=True, index=True)
    algorithm = Column(String)          # current / classical / quantum_inspired
    fuel_tons = Column(Float)
    cost = Column(Float)
    emissions_tons = Column(Float)
    runtime_seconds = Column(Float)
    schedule_met = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
