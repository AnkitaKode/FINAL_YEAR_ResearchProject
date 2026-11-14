"""
Data Preprocessing, Chronological Splitting, and Time-Series Windowing Pipeline.
Guarantees strict prevention of data leakage by fitting scalers only on training data.
"""

import numpy as np
import pandas as pd
import os
import joblib
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler

FEATURE_COLUMNS = [
    "outdoor_temperature",
    "indoor_humidity",
    "solar_generation_kw",
    "wind_generation_kw",
    "occupancy_count",
    "hvac_power_kw",
    "lighting_power_kw",
    "base_plug_load_kw",
    "electricity_price"
]

TARGET_COLUMNS = [
    "indoor_temperature",
    "total_demand_kw",
    "solar_generation_kw",
    "wind_generation_kw"
]

class MicrogridDataPreprocessor:
    def __init__(
        self,
        lookback_window: int = 24,
        forecast_horizon: int = 24,
        scaler_type: str = "standard"
    ):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.scaler_type = scaler_type
        self.scaler_X = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.scaler_Y = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.feature_cols = FEATURE_COLUMNS
        self.target_cols = TARGET_COLUMNS
        self.fitted = False

    def load_and_validate(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        df = pd.read_csv(file_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def chronological_split(
        self, df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df

    def fit(self, train_df: pd.DataFrame):
        X_train = train_df[self.feature_cols].values
        Y_train = train_df[self.target_cols].values
        self.scaler_X.fit(X_train)
        self.scaler_Y.fit(Y_train)
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on training data first!")
        X = self.scaler_X.transform(df[self.feature_cols].values)
        Y = self.scaler_Y.transform(df[self.target_cols].values)
        return X, Y

    def create_sliding_windows(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = len(X) - self.lookback_window - self.forecast_horizon + 1
        if num_samples <= 0:
            raise ValueError("Sequence length is too short for chosen lookback and forecast horizons.")
        
        X_windows = np.zeros((num_samples, self.lookback_window, X.shape[1]), dtype=np.float32)
        Y_windows = np.zeros((num_samples, self.forecast_horizon, Y.shape[1]), dtype=np.float32)
        
        for i in range(num_samples):
            X_windows[i] = X[i : i + self.lookback_window]
            Y_windows[i] = Y[i + self.lookback_window : i + self.lookback_window + self.forecast_horizon]
            
        return X_windows, Y_windows

    def inverse_transform_targets(self, Y_scaled: np.ndarray) -> np.ndarray:
        orig_shape = Y_scaled.shape
        if len(orig_shape) == 3:
            # [samples, horizon, features]
            flat = Y_scaled.reshape(-1, orig_shape[-1])
            unscaled = self.scaler_Y.inverse_transform(flat)
            return unscaled.reshape(orig_shape)
        return self.scaler_Y.inverse_transform(Y_scaled)

    def save_artifacts(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.scaler_X, os.path.join(directory, "scaler_X.pkl"))
        joblib.dump(self.scaler_Y, os.path.join(directory, "scaler_Y.pkl"))
        joblib.dump({
            "lookback_window": self.lookback_window,
            "forecast_horizon": self.forecast_horizon,
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols
        }, os.path.join(directory, "preprocessor_metadata.pkl"))

    def load_artifacts(self, directory: str):
        self.scaler_X = joblib.load(os.path.join(directory, "scaler_X.pkl"))
        self.scaler_Y = joblib.load(os.path.join(directory, "scaler_Y.pkl"))
        meta = joblib.load(os.path.join(directory, "preprocessor_metadata.pkl"))
        self.lookback_window = meta["lookback_window"]
        self.forecast_horizon = meta["forecast_horizon"]
        self.feature_cols = meta["feature_cols"]
        self.target_cols = meta["target_cols"]
        self.fitted = True

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "calibrated_smart_building_microgrid.csv")
    preproc = MicrogridDataPreprocessor(lookback_window=24, forecast_horizon=24)
    df = preproc.load_and_validate(csv_file)
    train_df, val_df, test_df = preproc.chronological_split(df)
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    preproc.fit(train_df)
    X_train, Y_train = preproc.transform(train_df)
    X_win, Y_win = preproc.create_sliding_windows(X_train, Y_train)
    print(f"Window shapes: X_win={X_win.shape}, Y_win={Y_win.shape}")
    art_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    preproc.save_artifacts(art_dir)
    print("Preprocessor artifacts saved successfully.")
