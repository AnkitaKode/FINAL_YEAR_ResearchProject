-- ==============================================================================
-- PRODUCTION POSTGRESQL 18 DDL SCHEMA
-- Project: Physics-Informed Deep Learning Architecture for Dynamic Building Energy Optimization
-- ==============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Users and Authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'researcher' CHECK (role IN ('admin', 'researcher', 'operator', 'viewer')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Registered Telemetry Datasets
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    source_type VARCHAR(50) CHECK (source_type IN ('real_world', 'calibrated_synthetic', 'benchmark')),
    file_path VARCHAR(512) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    total_records INTEGER NOT NULL,
    sampling_rate VARCHAR(50) NOT NULL,
    time_start TIMESTAMP WITH TIME ZONE,
    time_end TIMESTAMP WITH TIME ZONE,
    is_preprocessed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Telemetry and Sensor Readings
CREATE TABLE IF NOT EXISTS sensor_readings (
    id BIGSERIAL PRIMARY KEY,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    indoor_temp NUMERIC(5, 2),
    outdoor_temp NUMERIC(5, 2),
    indoor_humidity NUMERIC(5, 2),
    solar_generation_kw NUMERIC(8, 3),
    wind_generation_kw NUMERIC(8, 3),
    occupancy_count INTEGER,
    hvac_power_kw NUMERIC(8, 3),
    lighting_power_kw NUMERIC(8, 3),
    base_plug_load_kw NUMERIC(8, 3),
    total_demand_kw NUMERIC(8, 3),
    battery_soc NUMERIC(5, 2),
    grid_power_kw NUMERIC(8, 3),
    electricity_price NUMERIC(6, 4),
    is_synthetic BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sensor_dataset_ts ON sensor_readings (dataset_id, timestamp);

-- 4. Sensor Anomalies and Imputations
CREATE TABLE IF NOT EXISTS sensor_anomalies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    sensor_name VARCHAR(100) NOT NULL,
    raw_value NUMERIC(10, 3),
    imputed_value NUMERIC(10, 3),
    anomaly_type VARCHAR(100) NOT NULL,
    severity NUMERIC(4, 3) CHECK (severity BETWEEN 0 AND 1),
    is_resolved BOOLEAN DEFAULT FALSE,
    detection_algorithm VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_anomalies_ts ON sensor_anomalies (timestamp);

-- 5. Machine Learning Models Registry
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) CHECK (model_type IN ('PINN', 'LSTM', 'Transformer', 'DualHorizonTransformer', 'BaselinePersistence', 'BaselineMLP')),
    hyperparameters JSONB NOT NULL,
    artifact_path VARCHAR(512) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Training Experiments & Quantitative Metrics
CREATE TABLE IF NOT EXISTS training_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    epochs_trained INTEGER NOT NULL,
    training_time_seconds NUMERIC(8, 2) NOT NULL,
    train_loss NUMERIC(10, 6) NOT NULL,
    val_loss NUMERIC(10, 6) NOT NULL,
    test_mae NUMERIC(10, 6) NOT NULL,
    test_rmse NUMERIC(10, 6) NOT NULL,
    test_mape NUMERIC(8, 4),
    test_r2 NUMERIC(6, 4),
    physics_residual_error NUMERIC(10, 6),
    comfort_violation_rate NUMERIC(6, 4),
    loss_curve JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Forecast Runs History
CREATE TABLE IF NOT EXISTS forecast_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    forecast_horizon_hours INTEGER NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    predictions JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. Energy Optimization Runs
CREATE TABLE IF NOT EXISTS optimization_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    algorithm VARCHAR(100) NOT NULL,
    forecast_id UUID REFERENCES forecast_results(id) ON DELETE SET NULL,
    cost_baseline NUMERIC(10, 2) NOT NULL,
    cost_optimized NUMERIC(10, 2) NOT NULL,
    energy_savings_pct NUMERIC(5, 2) NOT NULL,
    peak_demand_reduction_pct NUMERIC(5, 2) NOT NULL,
    comfort_violation_hours NUMERIC(6, 2) NOT NULL,
    co2_emissions_kg NUMERIC(10, 2) NOT NULL,
    schedule_details JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 9. Audit Logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100),
    ip_address VARCHAR(45),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
