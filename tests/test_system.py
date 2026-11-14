"""
Comprehensive Test Suite for Physics-Informed Microgrid Platform.
Tests ML tensor shapes, physics residual exactness, chronological data leakage prevention,
MPC optimization constraints, sensor anomaly detection, and FastAPI endpoints.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.preprocessing.preprocessor import MicrogridDataPreprocessor
from ml.pinn.pinn_model import PhysicsInformedNN, PINNTrainer
from ml.lstm.lstm_model import LSTMForecaster
from ml.transformer.transformer_model import DualHorizonTransformer
from ml.anomaly_detection.anomaly_detector import SensorAnomalyDetector
from ml.optimization.optimizer import MicrogridOptimizer
from backend.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# -------------------------------------------------------------
# 1. MACHINE LEARNING & TENSOR SHAPE TESTS
# -------------------------------------------------------------
def test_pinn_model_shapes():
    model = PhysicsInformedNN(input_dim=9, hidden_dim=32, forecast_horizon=24, output_dim=4)
    dummy_x = torch.randn(8, 24, 9)
    out = model(dummy_x)
    assert out.shape == (8, 24, 4), f"PINN output shape mismatch: {out.shape}"

def test_lstm_model_shapes():
    model = LSTMForecaster(input_dim=9, hidden_dim=32, num_layers=2, forecast_horizon=24, output_dim=4)
    dummy_x = torch.randn(8, 24, 9)
    out = model(dummy_x)
    assert out.shape == (8, 24, 4), f"LSTM output shape mismatch: {out.shape}"

def test_transformer_model_shapes():
    model = DualHorizonTransformer(input_dim=9, d_model=32, nhead=2, forecast_horizon=24, output_dim=4, use_dual_horizon=True)
    dummy_x = torch.randn(8, 24, 9)
    out = model(dummy_x)
    assert out.shape == (8, 24, 4), f"DualHorizonTransformer output shape mismatch: {out.shape}"

# -------------------------------------------------------------
# 2. PHYSICS RESIDUAL EXACTNESS TEST
# -------------------------------------------------------------
def test_physics_residual_computation():
    model = PhysicsInformedNN(thermal_capacitance_cz=40.0, thermal_resistance_renv=2.5)
    
    # Construct exact thermal equilibrium where T_in = T_out = 20°C and Q_hvac = Q_int = Q_solar = 0
    # In equilibrium, dT/dt should be 0, and ODE RHS is 0 -> residual should be approximately 0.0
    t_in = torch.tensor([[20.0, 20.0]], dtype=torch.float32)
    t_out = torch.tensor([[20.0, 20.0]], dtype=torch.float32)
    hvac = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    occ = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    light = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    plug = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    
    residual = model.compute_physics_residual(
        predicted_indoor_temp=t_in,
        outdoor_temp=t_out,
        hvac_power_kw=hvac,
        occupancy_count=occ,
        lighting_power_kw=light,
        base_plug_kw=plug
    )
    assert torch.abs(residual).item() < 1e-4, f"Physics residual not zero in equilibrium: {residual.item()}"

# -------------------------------------------------------------
# 3. CHRONOLOGICAL SPLIT & DATA LEAKAGE TEST
# -------------------------------------------------------------
def test_chronological_data_split_no_leakage():
    dates = pd.date_range("2024-01-01", periods=1000, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "outdoor_temperature": np.random.normal(15, 5, 1000),
        "indoor_humidity": np.random.normal(50, 10, 1000),
        "solar_generation_kw": np.zeros(1000),
        "wind_generation_kw": np.zeros(1000),
        "occupancy_count": np.zeros(1000),
        "hvac_power_kw": np.zeros(1000),
        "lighting_power_kw": np.zeros(1000),
        "base_plug_load_kw": np.zeros(1000),
        "electricity_price": np.zeros(1000),
        "indoor_temperature": np.random.normal(21, 1, 1000),
        "total_demand_kw": np.random.normal(20, 5, 1000)
    })
    
    preproc = MicrogridDataPreprocessor()
    train_df, val_df, test_df = preproc.chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    
    # Train must precede Val, and Val must precede Test
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Data leakage between Train and Val!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Data leakage between Val and Test!"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Sample count mismatch!"

# -------------------------------------------------------------
# 4. SENSOR ANOMALY DETECTION TEST
# -------------------------------------------------------------
def test_anomaly_detector():
    detector = SensorAnomalyDetector()
    df = pd.DataFrame({
        "indoor_temperature": [21.0, 21.2, 55.0, 21.3, 21.3, 21.3, 21.3, 21.3, 21.3, 21.3], # contains spike (55) & stuck sensor
        "outdoor_temperature": [18.0] * 10,
        "total_demand_kw": [15.0] * 10,
        "solar_generation_kw": [5.0] * 10
    })
    detector.fit(df, ["indoor_temperature", "outdoor_temperature", "total_demand_kw", "solar_generation_kw"])
    res = detector.detect_anomalies(df)
    
    assert res["is_anomaly"].sum() > 0, "Failed to detect injected sensor anomalies"
    # Row 2 (55.0°C) must be flagged
    assert res.loc[2, "is_anomaly"] == True, "Out-of-bounds spike was not detected"

# -------------------------------------------------------------
# 5. OPTIMIZATION & BATTERY CONSTRAINT TEST
# -------------------------------------------------------------
def test_mpc_optimizer_constraints():
    opt = MicrogridOptimizer()
    horizon = 24
    outdoor = np.array([25.0] * horizon)
    occ = np.array([50] * horizon)
    solar = np.array([20.0] * horizon)
    wind = np.array([5.0] * horizon)
    price = np.array([0.16] * horizon)
    
    res = opt.optimize_schedule(outdoor, occ, solar, wind, price)
    
    # Check that energy savings are non-negative
    assert res["energy_savings_pct"] >= 0, "Energy savings cannot be negative"
    assert res["comfort_violation_hours"] == 0, "Comfort violation occurred"
    
    # Check SoC bounds
    socs = res["schedule"]["opt_soc_pct"]
    for s in socs:
        assert 19.9 <= s <= 90.1, f"Battery SoC violated physical bounds: {s}%"

# -------------------------------------------------------------
# 6. FASTAPI ENDPOINT INTEGRATION TESTS
# -------------------------------------------------------------
def test_api_root_and_kpis():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Physics-Informed" in res_root.json()["service"]

    res_kpis = client.get("/api/v1/analytics/kpis")
    assert res_kpis.status_code == 200
    assert "current_demand_kw" in res_kpis.json()

def test_api_forecast_and_optimization():
    res_fc = client.post("/api/v1/forecast/generate", json={"model_type": "Proposed_DualHorizon_PINN", "horizon_hours": 24})
    assert res_fc.status_code == 200
    assert len(res_fc.json()["forecast"]) == 24

    res_opt = client.post("/api/v1/optimization/run-mpc", json={"horizon_hours": 24, "initial_indoor_temp": 22.0})
    assert res_opt.status_code == 200
    assert res_opt.json()["cost_optimized_usd"] > 0
