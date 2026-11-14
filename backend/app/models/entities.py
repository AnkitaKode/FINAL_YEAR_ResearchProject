import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.database.session import Base

def gen_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="researcher")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(String(50), default="calibrated_synthetic")
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    total_records = Column(Integer, default=0)
    sampling_rate = Column(String(50), default="1-hour")
    is_preprocessed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class ModelRecord(Base):
    __tablename__ = "models"
    
    id = Column(String(36), primary_key=True, default=gen_uuid)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)
    hyperparameters = Column(JSON, default={})
    artifact_path = Column(String(512), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class OptimizationRecord(Base):
    __tablename__ = "optimization_runs"
    
    id = Column(String(36), primary_key=True, default=gen_uuid)
    algorithm = Column(String(100), default="Occupancy_Aware_MPC")
    cost_baseline = Column(Float, nullable=False)
    cost_optimized = Column(Float, nullable=False)
    energy_savings_pct = Column(Float, nullable=False)
    peak_demand_reduction_pct = Column(Float, nullable=False)
    comfort_violation_hours = Column(Float, default=0.0)
    co2_emissions_kg = Column(Float, nullable=False)
    schedule_details = Column(JSON, default={})
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
