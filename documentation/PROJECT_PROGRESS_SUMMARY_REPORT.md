1. **Parameter Calibration Dependency**: The 1R-1C thermal model assumes effective lumped parameters ($C_z, R_{\text{env}}$). In real-world buildings with non-homogeneous zones, parameter estimation errors can bias the physical loss if not periodically re-calibrated.
2. **Computational Scaling on Edge Hardware**: Training PINNs with continuous automatic differentiation requires non-negligible CPU/GPU memory. Our discretized finite-difference formulation resolves this, but multi-zone buildings with $>50$ thermal zones require distributed model-parallel training.
3. **Weather Forecast Horizon Degradation**: Solar and wind forecasts naturally degrade beyond 48-hour horizons due to atmospheric chaotic dynamics.
---
## 16. Chapter 15: Future Research Directions & Hardware-in-the-Loop Scope
1. **Multi-Zone Heat Transfer (2R-2C Formulation)**: Extending the lumped single-zone ODE to coupled multi-zone partial differential heat equations with inter-zone convective heat transfer.
2. **Hardware-in-the-Loop (HIL) Integration**: Interfacing the FastAPI backend with industrial BACnet/IP and Modbus TCP programmable logic controllers (PLCs) in live commercial testbeds.
3. **Reinforcement Learning with Physics Action Masks**: Investigating Deep Q-Networks (DQN) or Proximal Policy Optimization (PPO) where action spaces are physically constrained by PINN surrogate boundaries.
---
