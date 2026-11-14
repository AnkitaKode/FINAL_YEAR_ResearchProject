"""
Dual-Horizon Attention Transformer & Standard Transformer Baseline for Multi-Horizon Microgrid Forecasting.
Decouples intra-day transient dynamics from macro diurnal cycles with cross-attention gating.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from typing import Dict, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class DualHorizonTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 9,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        output_dim: int = 4,
        dropout: float = 0.1,
        use_dual_horizon: bool = True
    ):
        super(DualHorizonTransformer, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.d_model = d_model
        self.use_dual_horizon = use_dual_horizon
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Long-Horizon Global Attention
        encoder_layer_long = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.long_horizon_encoder = nn.TransformerEncoder(encoder_layer_long, num_layers=num_layers)
        
        # Short-Horizon Local Transient Conv/Attention
        if use_dual_horizon:
            self.short_horizon_conv = nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1
            )
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        
        self.decoder_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, forecast_horizon * output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback, input_dim]
        b, l, _ = x.shape
        proj = self.input_proj(x) # [batch, lookback, d_model]
        emb = self.pos_encoder(proj)
        
        long_feat = self.long_horizon_encoder(emb) # [batch, lookback, d_model]
        
        if self.use_dual_horizon:
            # short-term transient representation via 1D temporal convolution
            trans_short = self.short_horizon_conv(emb.transpose(1, 2)).transpose(1, 2)
            # combine last representations
            combined = torch.cat([long_feat[:, -1, :], trans_short[:, -1, :]], dim=-1)
            g = self.gate(combined) # [batch, d_model]
            fused = g * trans_short[:, -1, :] + (1.0 - g) * long_feat[:, -1, :]
        else:
            fused = long_feat[:, -1, :]
            
        out = self.decoder_proj(fused)
        return out.view(-1, self.forecast_horizon, self.output_dim)

class TransformerTrainer:
    def __init__(
        self,
        model: DualHorizonTransformer,
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
    model = DualHorizonTransformer(input_dim=9, d_model=32, nhead=2, forecast_horizon=24, output_dim=4, use_dual_horizon=True)
    trainer = TransformerTrainer(model)
    x = torch.randn(8, 24, 9)
    y = torch.randn(8, 24, 4)
    loss = trainer.train_step(x, y)
    print("DualHorizonTransformer train step loss:", loss)
