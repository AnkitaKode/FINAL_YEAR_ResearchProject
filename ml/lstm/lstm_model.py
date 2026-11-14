"""
Multi-Horizon LSTM Forecasting Model for Building Energy Load and Renewables.
Provides sequential time-series baseline with multi-feature inputs and configurable horizons.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Dict, Tuple

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        output_dim: int = 4,
        dropout: float = 0.2
    ):
        super(LSTMForecaster, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon * output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback, input_dim]
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :] # [batch, hidden_dim]
        out = self.fc(last_step)
        return out.view(-1, self.forecast_horizon, self.output_dim)

class LSTMTrainer:
    def __init__(
        self,
        model: LSTMForecaster,
        lr: float = 0.001,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        pred = self.model(batch_x)
        loss = self.criterion(pred, batch_y)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def evaluate(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            pred = self.model(batch_x)
            loss = self.criterion(pred, batch_y)
        return float(loss.item())

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

if __name__ == "__main__":
    model = LSTMForecaster(input_dim=9, hidden_dim=32, num_layers=2, forecast_horizon=24, output_dim=4)
    trainer = LSTMTrainer(model)
    x = torch.randn(8, 24, 9)
    y = torch.randn(8, 24, 4)
    loss = trainer.train_step(x, y)
    print("LSTM train step loss:", loss)
