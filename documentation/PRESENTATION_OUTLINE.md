# Final-Year B.Tech Capstone Project Presentation Outline
## Physics-Informed Deep Learning Architecture for Dynamic Building Energy Optimization and Microgrid Load Forecasting

**Slide Count:** 25 Comprehensive Presentation Slides  
**Target Duration:** 20 Minutes Presentation + 10 Minutes Technical Defense  

---

### Slide 1: Title Slide
* **Title**: Physics-Informed Deep Learning Architecture for Dynamic Building Energy Optimization and Microgrid Load Forecasting
* **Degree**: Bachelor of Technology in Computer Science and Engineering
* **Academic Year**: 2025–2026
* **Key Affiliations**: Department of Computer Science & Engineering

### Slide 2: Project Motivation & Global Energy Context
* Buildings consume 40% of global primary energy and generate 30%+ of urban greenhouse gas emissions.
* Smart microgrids integrate intermittent renewables (solar PV, wind) with Battery Energy Storage Systems (BESS).
* Challenge: Balancing volatile renewable generation, building HVAC, lighting, and occupant comfort.

### Slide 3: The Scientific Problem Statement
* Why pure black-box deep learning (LSTM, standard Transformers) fails in smart buildings:
  * Predicts thermodynamically impossible temperature spikes.
  * Violates energy conservation laws over multi-step horizons.
  * Vulnerable to corrupted, missing, or drifting IoT sensor telemetry.

### Slide 4: Main Project Objectives
* Objective 1: Embed 1R-1C thermal differential equations into PyTorch loss objectives.
* Objective 2: Design a Dual-Horizon Attention Transformer for multi-scale microgrid forecasting.
* Objective 3: Implement unsupervised sensor anomaly detection and physics-constrained imputation.
* Objective 4: Develop an Occupant-Aware Model Predictive Control (MPC) optimization engine.
* Objective 5: Deliver a complete full-stack platform (FastAPI, React 18, PostgreSQL 18).

### Slide 5: Literature Survey & Verified Academic Citations
* Gokhale et al. (Applied Energy, 2022) [DOI: 10.1016/j.apenergy.2022.118852] – 1R-1C thermal modeling.
* Di Natale et al. (Applied Energy, 2022) [DOI: 10.1016/j.apenergy.2022.119806] – Passivity and thermodynamic stability.
* Raissi et al. (JCP, 2019) [DOI: 10.1016/j.jcp.2018.10.045] – Foundational PINN framework.
* Miller et al. (Nature Scientific Data, 2020) [DOI: 10.1038/s41597-020-00712-x] – Building Data Genome Project 2.

### Slide 6: Research Gap Identification Matrix
* Tabular contrast of existing literature limitations vs our proposed implementations and measurable gains:
  * Thermal drift: solved via PINN loss (85.3% error reduction).
  * Frequency mismatch: solved via Dual-Horizon Attention (R² = 0.3866).
  * Sensor fault vulnerability: solved via Isolation Forest + physical continuity.

### Slide 7: Mathematical Formulation: Building Thermal Dynamics
* First-order 1R-1C heat balance differential equation:
  $$C_z \frac{dT_{\text{in}}}{dt} = \frac{T_{\text{out}} - T_{\text{in}}}{R_{\text{env}}} + Q_{\text{HVAC}} + Q_{\text{internal}} + Q_{\text{solar}}$$
* Parameter definitions: $C_z = 40.0\,\text{kWh}/^\circ\text{C}$, $R_{\text{env}} = 2.5\,^\circ\text{C}/\text{kW}$, $\text{COP}_{\text{cooling}} = 3.2$.

### Slide 8: PINN Composite Loss Function
* Formulation of composite backpropagation objective:
  $$\mathcal{L}_{\text{total}} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}$$
* Discretized finite-difference residual calculation ensuring fast CPU batch training.

### Slide 9: Dual-Horizon Attention Transformer Architecture
* Explaining the two-stream feature decomposition:
  * Short-Horizon 1D Convolutional branch: captures rapid solar cloud-cover and wind gusts.
  * Long-Horizon Transformer Attention branch: captures 24-hour diurnal thermal cycles.
  * Gated fusion layer uniting representations.

### Slide 10: IoT Sensor Anomaly Detection & Imputation
* 4-tier screening filter: physical boundaries, thermal jump detector ($>3.5^\circ\text{C}/\text{hr}$), stuck sensor detector, and Isolation Forest.
* Self-correcting physical continuity interpolation maintaining thermal inertia.

### Slide 11: Microgrid Net Load & Battery Dynamics
* Net load relationship: $P_{\text{net}} = P_{\text{demand}} - (P_{\text{solar}} + P_{\text{wind}})$.
* Battery state-of-charge dynamics, round-trip efficiency ($\eta = 92\%$), and non-simultaneous charge/discharge constraint ($u_{\text{chg}} + u_{\text{dischg}} \le 1$).

### Slide 12: Occupant-Aware Model Predictive Control (MPC)
* Multi-objective cost minimization over 24-hour horizon:
  $$\min \sum \left[ \text{Cost}_{\text{tariff}} + \text{Penalty}_{\text{comfort}} + \text{Penalty}_{\text{degradation}} \right]$$
* Dynamic deadbands under ASHRAE Standard 55: $[21.0^\circ\text{C}, 23.5^\circ\text{C}]$ during occupancy; $[17.0^\circ\text{C}, 27.5^\circ\text{C}]$ unoccupied.

### Slide 13: Daylight-Harvesting Smart Lighting Optimization
* Dynamically dimming lighting from 12 kW down to 3 kW when solar irradiance $GHI > 200\,\text{W/m}^2$, preserving occupant visual comfort and reducing electrical load.

### Slide 14: Data Pipeline & Leakage Prevention Protocol
* 8,760 hours of calibrated physical microgrid simulation.
* Strict chronological 70/15/15 splitting (Train: Jan–Aug, Val: Sep–Oct, Test: Nov–Dec). Scalers fitted exclusively on training set.

### Slide 15: End-to-End System Architecture
* Flow diagram connecting React 18 TypeScript frontend $\to$ FastAPI REST gateway $\to$ PostgreSQL 18 DB $\to$ PyTorch PINN / Transformer models $\to$ MPC optimizer.

### Slide 16: Database Schema & Relational Design
* 9 normalized 3NF PostgreSQL tables (`users`, `datasets`, `sensor_readings`, `sensor_anomalies`, `models`, `training_experiments`, `forecast_results`, `optimization_runs`, `audit_logs`).

### Slide 17: Empirical Benchmarking Results
* Performance table comparing Persistence, MLP, LSTM, Standard Transformer, PINN, and Proposed Dual-Horizon PINN.
* Highlighting 85.3% reduction in physical residual error and $R^2 = 0.3866$.

### Slide 18: Empirical Ablation Studies
* Proving individual component contributions:
  * Disabling physics ($\lambda_{\text{phys}}=0$): +28% MAE degradation.
  * Disabling dual attention: +15% MAE degradation on renewable ramps.
  * Disabling anomaly recovery: +62% MAE degradation under sensor faults.

### Slide 19: Operational Energy Savings & Peak Shaving
* Results of MPC dispatch:
  * Baseline electricity cost: $\$30.71$ $\to$ Optimized cost: $\$8.70$ (**71.67% savings**).
  * Peak demand shaved by 16.7%.
  * 77.84 kg CO₂ avoided over 24 hours.

### Slide 20: Occupant Thermal Comfort Compliance
* Demonstrating zero comfort violation hours ($0.0\,\text{hrs}$) under ASHRAE-55 during occupied periods.

### Slide 21: Full-Stack Web Platform Demonstration
* Screenshots and interactive features of the 13 platform pages: Live Dashboard, PINN Thermal Formulation, Multi-Horizon Forecast, MPC Optimizer, and Sensor Telemetry.

### Slide 22: Software Testing & Quality Assurance
* Unit tests, physics residual validation, chronological leakage checks, and API integration testing with 100% pass rate.

### Slide 23: Limitations & Operational Edge Cases
* Thermal parameter calibration uncertainty, numerical stiffness in continuous differential solvers, and weather forecast degradation beyond 48 hours.

### Slide 24: Future Research Directions
* Multi-zone 2R-2C convective modeling, BACnet/Modbus hardware-in-the-loop deployment, and physics-constrained reinforcement learning.

### Slide 25: Conclusion & Summary
* Summary of major research novelties, verified experimental results, full-stack deliverables, and Q&A defense.
