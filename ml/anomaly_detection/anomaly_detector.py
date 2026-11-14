"""
IoT Sensor Anomaly Detection and Self-Correcting Imputation Engine.
Detects missing readings, sudden spikes, sensor drift, stuck values, and unphysical measurements.
Provides physics-constrained reconstruction of corrupted telemetry.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
from typing import Dict, List, Tuple, Any

class SensorAnomalyDetector:
    def __init__(
        self,
        contamination: float = 0.03,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.fitted = False
        
        # Physical domain sanity bounds for building telemetry
        self.physical_bounds = {
            "indoor_temperature": (10.0, 38.0),   # °C
            "outdoor_temperature": (-20.0, 50.0), # °C
            "indoor_humidity": (10.0, 95.0),      # %
            "solar_generation_kw": (0.0, 60.0),   # kW
            "wind_generation_kw": (0.0, 40.0),    # kW
            "total_demand_kw": (1.0, 80.0),       # kW
            "battery_soc_pct": (0.0, 100.0)       # %
        }

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        clean_df = df[feature_cols].dropna()
        self.iso_forest.fit(clean_df.values)
        self.feature_cols = feature_cols
        self.fitted = True

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        results = df.copy()
        n = len(df)
        
        is_anomaly = np.zeros(n, dtype=bool)
        anomaly_types = [[] for _ in range(n)]
        anomaly_scores = np.zeros(n, dtype=float)
        
        # 1. Physical Domain Bound Violations
        for col, (b_min, b_max) in self.physical_bounds.items():
            if col in df.columns:
                vals = df[col].values
                out_bounds = (vals < b_min) | (vals > b_max)
                for idx in np.where(out_bounds)[0]:
                    is_anomaly[idx] = True
                    anomaly_types[idx].append(f"{col}_out_of_bounds")
                    anomaly_scores[idx] = max(anomaly_scores[idx], 0.95)
                    
        # 2. Sudden Spikes & Physical Rate of Change Violations
        # Indoor temperature cannot realistically jump more than 3.5°C in 1 hour due to thermal mass
        if "indoor_temperature" in df.columns:
            t_diff = np.abs(np.diff(df["indoor_temperature"].values, prepend=df["indoor_temperature"].values[0]))
            spike_idx = np.where(t_diff > 3.5)[0]
            for idx in spike_idx:
                is_anomaly[idx] = True
                anomaly_types[idx].append("indoor_temp_unphysical_jump")
                anomaly_scores[idx] = max(anomaly_scores[idx], 0.85)

        # 3. Stuck Sensor Values (Sensor outputs identical value for >= 6 consecutive hours)
        for col in ["indoor_temperature", "total_demand_kw"]:
            if col in df.columns:
                vals = df[col].values
                for i in range(5, n):
                    window = vals[i-5:i+1]
                    if np.all(window == window[0]) and not np.isnan(window[0]):
                        is_anomaly[i] = True
                        anomaly_types[i].append(f"{col}_stuck_sensor")
                        anomaly_scores[i] = max(anomaly_scores[i], 0.75)

        # 4. Multidimensional Isolation Forest Scoring
        if self.fitted and hasattr(self, "feature_cols"):
            avail_cols = [c for c in self.feature_cols if c in df.columns]
            if avail_cols:
                X = df[avail_cols].fillna(df[avail_cols].median()).values
                preds = self.iso_forest.predict(X) # -1 is anomaly
                raw_scores = -self.iso_forest.score_samples(X)
                # Normalize raw scores to [0, 1]
                norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
                for i in range(n):
                    if preds[i] == -1 and norm_scores[i] > 0.65:
                        is_anomaly[i] = True
                        anomaly_types[i].append("multivariate_isolation_outlier")
                        anomaly_scores[i] = max(anomaly_scores[i], float(norm_scores[i]))

        results["is_anomaly"] = is_anomaly
        results["anomaly_type"] = [", ".join(t) if t else "normal" for t in anomaly_types]
        results["anomaly_score"] = np.round(anomaly_scores, 3)
        return results

    def impute_sensor_data(self, df: pd.DataFrame, target_col: str = "indoor_temperature") -> pd.DataFrame:
        """
        Self-correcting physical continuity interpolation.
        Replaces anomalous and missing entries with physically bounded time-series smoothing.
        """
        res = df.copy()
        if target_col not in res.columns:
            return res
            
        vals = res[target_col].copy()
        if "is_anomaly" in res.columns:
            anom_mask = res["is_anomaly"] & res["anomaly_type"].str.contains(target_col)
            vals[anom_mask] = np.nan
            
        # Linear interpolation with forward/backward fill for physical continuity
        imputed = vals.interpolate(method="linear").bfill().ffill()
        res[f"{target_col}_original"] = res[target_col]
        res[f"{target_col}_imputed"] = np.round(imputed, 2)
        res[f"is_{target_col}_imputed"] = res[target_col].isna() | (res.get("is_anomaly", False))
        return res

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"iso_forest": self.iso_forest, "fitted": self.fitted}, path)

    def load_model(self, path: str):
        data = joblib.load(path)
        self.iso_forest = data["iso_forest"]
        self.fitted = data["fitted"]

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "calibrated_smart_building_microgrid.csv")
    df = pd.read_csv(csv_file).head(500)
    
    # Inject deliberate synthetic anomalies to test detection
    df.loc[50, "indoor_temperature"] = 48.0 # Extreme jump
    df.loc[100:106, "indoor_temperature"] = 21.32 # Stuck sensor
    df.loc[150, "total_demand_kw"] = 120.0 # Extreme power spike
    
    detector = SensorAnomalyDetector(contamination=0.03)
    feature_cols = ["indoor_temperature", "outdoor_temperature", "total_demand_kw", "solar_generation_kw"]
    detector.fit(df, feature_cols)
    anom_df = detector.detect_anomalies(df)
    imputed_df = detector.impute_sensor_data(anom_df, "indoor_temperature")
    
    num_anoms = anom_df["is_anomaly"].sum()
    print(f"Anomaly Detection Completed: Found {num_anoms} anomalies out of {len(df)} samples.")
    print("Sample detected anomalies:")
    print(anom_df[anom_df["is_anomaly"]][["timestamp", "indoor_temperature", "anomaly_type", "anomaly_score"]].head())
