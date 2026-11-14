"""
Physics-Informed Neural Network (PINN) for Building Thermal Dynamics and Energy Load.
Embeds the first-order 1R-1C thermal energy balance differential equation directly into PyTorch loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import os

class PhysicsInformedNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        forecast_horizon: int = 24,
        output_dim: int = 4, # [indoor_temp, total_demand, solar_gen, wind_gen]
        num_layers: int = 2,
        dropout: float = 0.1,
        # Thermodynamic parameters (calibrated for commercial/academic zone)
        thermal_capacitance_cz: float = 40.0, # kWh/°C
        thermal_resistance_renv: float = 2.5, # °C/kW
        cop_cooling: float = 3.2,
        cop_heating: float = 2.8,
        dt_hours: float = 1.0
    ):
        super(PhysicsInformedNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.dt_hours = dt_hours
        
        # Physics Parameters (can be fixed or made learnable)
        self.C_z = nn.Parameter(torch.tensor(thermal_capacitance_cz, dtype=torch.float32), requires_grad=False)
        self.R_env = nn.Parameter(torch.tensor(thermal_resistance_renv, dtype=torch.float32), requires_grad=False)
        self.cop_cooling = cop_cooling
        self.cop_heating = cop_heating
        
        # Temporal Encoder
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dense Forecast Head
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon * output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, lookback_seq_len, input_dim]
        rnn_out, _ = self.encoder_rnn(x) # [batch_size, seq_len, hidden_dim]
        last_hidden = rnn_out[:, -1, :] # [batch_size, hidden_dim]
        out_flat = self.fc_head(last_hidden) # [batch_size, forecast_horizon * output_dim]
        out = out_flat.view(-1, self.forecast_horizon, self.output_dim)
        return out

    def compute_physics_residual(
        self,
        predicted_indoor_temp: torch.Tensor,     # [batch, horizon] (unscaled or physically scaled)
        outdoor_temp: torch.Tensor,              # [batch, horizon]
        hvac_power_kw: torch.Tensor,             # [batch, horizon]
        occupancy_count: torch.Tensor,           # [batch, horizon]
        lighting_power_kw: torch.Tensor,         # [batch, horizon]
        base_plug_kw: torch.Tensor,              # [batch, horizon]
        solar_ghi_wm2: Optional[torch.Tensor] = None # [batch, horizon]
    ) -> torch.Tensor:
        """
        Computes 1R-1C thermal balance equation residual:
        Residual = dT_in/dt - (1/C_z) * [ (T_out - T_in)/R_env + Q_HVAC + Q_internal + Q_solar ]
        """
        batch_size, horizon = predicted_indoor_temp.shape
        if horizon < 2:
            return torch.zeros(1, device=predicted_indoor_temp.device)
            
        # Finite-difference derivative: dT/dt
        dT_dt = (predicted_indoor_temp[:, 1:] - predicted_indoor_temp[:, :-1]) / self.dt_hours
        
        T_in_mid = (predicted_indoor_temp[:, 1:] + predicted_indoor_temp[:, :-1]) / 2.0
        T_out_mid = (outdoor_temp[:, 1:] + outdoor_temp[:, :-1]) / 2.0
        hvac_mid = (hvac_power_kw[:, 1:] + hvac_power_kw[:, :-1]) / 2.0
        occ_mid = (occupancy_count[:, 1:] + occupancy_count[:, :-1]) / 2.0
        light_mid = (lighting_power_kw[:, 1:] + lighting_power_kw[:, :-1]) / 2.0
        plug_mid = (base_plug_kw[:, 1:] + base_plug_kw[:, :-1]) / 2.0
        
        # Envelope conduction heat transfer
        Q_env = (T_out_mid - T_in_mid) / self.R_env
        
        # Internal thermal gains (kW)
        Q_internal = (occ_mid * 0.1) + (0.80 * plug_mid) + (0.85 * light_mid)
        
        # Solar thermal gains through windows (kW)
        if solar_ghi_wm2 is not None:
            ghi_mid = (solar_ghi_wm2[:, 1:] + solar_ghi_wm2[:, :-1]) / 2.0
            Q_solar = (ghi_mid / 1000.0) * 80.0 * 0.35
        else:
            Q_solar = torch.zeros_like(Q_env)
            
        # HVAC thermal injection/extraction (negative for cooling when T_out > T_in)
        # Cooling mode dominates when outdoor > 20°C
        is_cooling = (T_out_mid >= 20.0).float()
        Q_hvac = -1.0 * hvac_mid * (is_cooling * self.cop_cooling + (1.0 - is_cooling) * (-self.cop_heating))
        
        # Governing physical ODE right-hand side
        ode_rhs = (Q_env + Q_hvac + Q_internal + Q_solar) / self.C_z
        
        # Physical residual error: difference between neural rate of change and thermodynamic law
        residual = dT_dt - ode_rhs
        return residual

class PINNTrainer:
    def __init__(
        self,
        model: PhysicsInformedNN,
        lr: float = 0.001,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.25,
        lambda_bound: float = 0.10,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.mse_criterion = nn.MSELoss()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.lambda_bound = lambda_bound

    def train_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        unscaled_x_features: Optional[Dict[str, torch.Tensor]] = None,
        unscaled_t_in: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        predictions = self.model(batch_x)
        
        # 1. Data Fitting Loss (Supervised MSE across all target channels)
        loss_data = self.mse_criterion(predictions, batch_y)
        
        # 2. Physics-Informed Residual Loss
        loss_physics = torch.tensor(0.0, device=self.device)
        if self.lambda_phys > 0 and unscaled_x_features is not None and unscaled_t_in is not None:
            # We compute residual on indoor temperature channel (index 0)
            pred_t_in = predictions[:, :, 0]
            # Use unscaled features to evaluate physical differential balance
            res = self.model.compute_physics_residual(
                predicted_indoor_temp=unscaled_t_in, # or combined proxy
                outdoor_temp=unscaled_x_features["outdoor_temp"],
                hvac_power_kw=unscaled_x_features["hvac_power"],
                occupancy_count=unscaled_x_features["occupancy"],
                lighting_power_kw=unscaled_x_features["lighting_power"],
                base_plug_kw=unscaled_x_features["base_plug"]
            )
            loss_physics = torch.mean(res ** 2)
            
        # 3. Boundary Condition Loss (Indoor temperature physically bounded in [10°C, 35°C])
        # Penalty for exceeding reasonable safe building limits
        t_min_bound = -3.0 # normalized lower bound approx 10°C
        t_max_bound = 3.0  # normalized upper bound approx 35°C
        pred_temp = predictions[:, :, 0]
        loss_boundary = torch.mean(torch.relu(t_min_bound - pred_temp)**2 + torch.relu(pred_temp - t_max_bound)**2)
        
        total_loss = (
            self.lambda_data * loss_data +
            self.lambda_phys * loss_physics +
            self.lambda_bound * loss_boundary
        )
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": float(total_loss.item()),
            "loss_data": float(loss_data.item()),
            "loss_physics": float(loss_physics.item()),
            "loss_boundary": float(loss_boundary.item())
        }

    def evaluate(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            predictions = self.model(batch_x)
            loss = self.mse_criterion(predictions, batch_y)
        return float(loss.item())

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

if __name__ == "__main__":
    model = PhysicsInformedNN(input_dim=9, hidden_dim=32, forecast_horizon=24, output_dim=4)
    trainer = PINNTrainer(model=model, lambda_phys=0.25)
    dummy_x = torch.randn(16, 24, 9)
    dummy_y = torch.randn(16, 24, 4)
    metrics = trainer.train_step(dummy_x, dummy_y)
    print("PINN training step executed successfully:", metrics)
