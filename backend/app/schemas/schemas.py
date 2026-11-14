from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: Optional[str] = "researcher"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: str
    role: str

class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool

class ForecastRequest(BaseModel):
    model_type: Optional[str] = "Proposed_DualHorizon_PINN"
    horizon_hours: Optional[int] = 24
    start_hour_offset: Optional[int] = 0

class OptimizationRequest(BaseModel):
    horizon_hours: Optional[int] = 24
    initial_indoor_temp: Optional[float] = 22.0
    initial_soc_pct: Optional[float] = 50.0
    bess_capacity_kwh: Optional[float] = 60.0
    co2_factor: Optional[float] = 0.72

class AnomalyDetectionRequest(BaseModel):
    contamination: Optional[float] = 0.03
    max_rows: Optional[int] = 500
