# Comprehensive Viva Voce Examination Defense Guide: 100 Technical Q&As
## Physics-Informed Deep Learning Architecture for Dynamic Building Energy Optimization and Microgrid Load Forecasting

**Target:** Final-Year B.Tech CSE Project Defense, External Review Board & Technical Viva  
**Coverage:** PINNs, Thermodynamics, Heat Transfer, Transformers, Microgrids, MPC, Optimization, Evaluation Metrics, Software Architecture.

---

### SECTION 1: Physics-Informed Neural Networks & Thermodynamics (Questions 1–25)

**1. What is the fundamental difference between standard deep learning and a Physics-Informed Neural Network (PINN)?**  
*Answer:* Standard deep learning maps inputs to outputs solely by minimizing empirical prediction error ($\mathcal{L}_{\text{data}}$) on labeled observations, behaving as an unconstrained black box. A PINN incorporates domain-specific governing differential equations (e.g., heat transfer or Navier-Stokes equations) directly into the loss function: $\mathcal{L}_{\text{total}} = \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \lambda_{\text{phys}}\mathcal{L}_{\text{phys}} + \lambda_{\text{bound}}\mathcal{L}_{\text{bound}}$, penalizing deviations from physical conservation laws.

**2. What is the governing thermodynamic differential equation used in your building model?**  
*Answer:* The 1R-1C lumped capacitance thermal balance equation:
$$C_z \frac{dT_{\text{in}}(t)}{dt} = \frac{T_{\text{out}}(t) - T_{\text{in}}(t)}{R_{\text{env}}} + Q_{\text{HVAC}}(t) + Q_{\text{internal}}(t) + Q_{\text{solar}}(t)$$

**3. What does $C_z$ represent and what are its physical units?**  
*Answer:* $C_z$ represents the effective lumped indoor zone thermal capacitance ($\text{kWh}/^\circ\text{C}$ or $\text{kJ}/^\circ\text{C}$). It measures the amount of thermal energy required to change the indoor zone temperature by one degree Celsius. In our model, it is calibrated to $40.0\,\text{kWh}/^\circ\text{C}$.

**4. What does $R_{\text{env}}$ represent and how is it related to the overall heat transfer coefficient $UA$?**  
*Answer:* $R_{\text{env}}$ represents the effective thermal resistance of the building envelope ($^\circ\text{C}/\text{kW}$). It is inversely proportional to the overall heat transfer coefficient and surface area: $R_{\text{env}} = \frac{1}{UA}$. In our building, $R_{\text{env}} = 2.5\,^\circ\text{C}/\text{kW}$, meaning $UA = 0.4\,\text{kW}/^\circ\text{C}$.

**5. How is $Q_{\text{HVAC}}$ defined, and why does its sign change between cooling and heating?**  
*Answer:* $Q_{\text{HVAC}}$ is the thermal heat rate injected into or extracted from the zone. For cooling (when $T_{\text{out}} \ge 20^\circ\text{C}$), heat is extracted from the building, so $Q_{\text{HVAC}} = -\text{COP}_{\text{cooling}} \cdot P_{\text{HVAC,elec}}$. For heating, heat is injected, so $Q_{\text{HVAC}} = +\text{COP}_{\text{heating}} \cdot P_{\text{HVAC,elec}}$.

**6. What is the Coefficient of Performance (COP)?**  
*Answer:* COP is the ratio of useful thermal energy transferred to the electrical energy consumed by the compressor/heat pump: $\text{COP} = \frac{|Q_{\text{thermal}}|}{P_{\text{electrical}}}$. Our model uses $\text{COP}_{\text{cooling}} = 3.2$ and $\text{COP}_{\text{heating}} = 2.8$.

**7. How did you compute $\frac{dT_{\text{in}}}{dt}$ inside your PyTorch PINN model?**  
*Answer:* We evaluated $\frac{dT_{\text{in}}}{dt}$ using a differentiable finite-difference layer: $\frac{\hat{T}_{\text{in}}(t_{k+1}) - \hat{T}_{\text{in}}(t_k)}{\Delta t}$ on discrete batched sequence tensors.

**8. Why did you use discrete finite differences instead of continuous PyTorch autograd?**  
*Answer:* Continuous autograd requires evaluating network gradients with respect to continuous coordinate input $t$ at sampled collocation points, which increases training latency on CPU by $\approx 10\times$. Finite difference operates directly on batched sequence slices, maintaining vectorization and numerical stability.

**9. What is the physical residual $R_{\text{therm}}$?**  
*Answer:* The physical residual is the algebraic error between the rate of temperature change predicted by the neural network and the right-hand side of the thermodynamic ODE:
$$R_{\text{therm}}(t_k) = \frac{\hat{T}_{\text{in}}(t_{k+1}) - \hat{T}_{\text{in}}(t_k)}{\Delta t} - \frac{1}{C_z} \left[ \frac{\bar{T}_{\text{out}} - \bar{T}_{\text{in}}}{R_{\text{env}}} + \bar{Q}_{\text{HVAC}} + \bar{Q}_{\text{int}} + \bar{Q}_{\text{sol}} \right]$$

**10. How does the PINN loss enforce conservation of energy?**  
*Answer:* By minimizing $\mathcal{L}_{\text{phys}} = \text{MSE}(R_{\text{therm}}, 0)$, any trajectory where indoor temperature warms without corresponding heat gain ($Q_{\text{env}} + Q_{\text{HVAC}} + Q_{\text{int}} + Q_{\text{sol}} > 0$) is heavily penalized during gradient descent.

**11. What is the boundary condition loss $\mathcal{L}_{\text{bound}}$?**  
*Answer:* It enforces thermodynamic passivity and physical viability by penalizing temperatures outside safe operational limits ($10^\circ\text{C}$ to $38^\circ\text{C}$):
$$\mathcal{L}_{\text{bound}} = \text{mean}\left(\text{ReLU}(10^\circ\text{C} - \hat{T}_{\text{in}})^2 + \text{ReLU}(\hat{T}_{\text{in}} - 38^\circ\text{C})^2\right)$$

**12. How were the loss weights ($\lambda_{\text{data}}, \lambda_{\text{phys}}, \lambda_{\text{bound}}$) selected?**  
*Answer:* Through systematic grid search on validation data: $\lambda_{\text{data}} = 1.0$, $\lambda_{\text{phys}} = 0.25$, $\lambda_{\text{bound}} = 0.10$. Setting $\lambda_{\text{phys}} > 0.5$ caused underfitting of empirical meter readings.

**13. What was the observed reduction in physical residual error achieved by your PINN?**  
*Answer:* The physical residual error dropped from 0.842 in standard MLP down to 0.124 in the PINN—an **85.3% reduction in physical violation**.

**14. What are internal heat gains $Q_{\text{internal}}$ comprised of?**  
*Answer:* Metabolic heat from occupants ($0.1\,\text{kW/person}$), plug load heat dissipation ($80\%$ of plug power), and lighting heat dissipation ($85\%$ of lighting power).

**15. How do you calculate solar heat gain $Q_{\text{solar}}$?**  
*Answer:* $Q_{\text{solar}} = \frac{I_{\text{solar}}}{1000} \cdot A_{\text{win}} \cdot \text{SHGC}$, where $A_{\text{win}} = 80.0\,\text{m}^2$ and $\text{SHGC} = 0.35$.

**16. What is the thermal time constant of your building envelope?**  
*Answer:* The thermal time constant $\tau = R_{\text{env}} \cdot C_z = 2.5\,^\circ\text{C}/\text{kW} \times 40.0\,\text{kWh}/^\circ\text{C} = 100\,\text{hours}$. This indicates high thermal inertia, meaning the envelope retains heat or coolness for multiple days.

**17. What is stiffness in ordinary differential equations, and does it occur here?**  
*Answer:* Stiffness occurs when dynamic time constants differ by multiple orders of magnitude. Because building thermal capacitance has a large time constant ($\tau = 100\,\text{h}$) while sampling is $\Delta t = 1.0\,\text{h}$, the system is moderately stiff, requiring normalized loss gradients to prevent instability.

**18. What is Fourier's Law of Heat Conduction?**  
*Answer:* Heat conduction rate is proportional to the negative temperature gradient: $q = -k \nabla T$. In 1D building envelopes, this simplifies to $Q_{\text{env}} = \frac{T_{\text{out}} - T_{\text{in}}}{R_{\text{env}}}$.

**19. Why does a standard LSTM violate thermodynamics over multi-step horizons?**  
*Answer:* Because standard LSTMs have no internal concept of heat conservation or thermal capacitance. In recursive multi-step forecasting, small errors compound, causing the hidden states to drift toward non-physical values.

**20. What is the difference between a forward problem and an inverse problem in PINNs?**  
*Answer:* In a forward problem, physical parameters ($C_z, R_{\text{env}}$) are known, and the PINN solves for the state field ($T_{\text{in}}(t)$). In an inverse problem, the PINN simultaneously infers unknown physical parameters from sparse sensor observations.

**21. Can your model be adapted to an inverse problem?**  
*Answer:* Yes. $C_z$ and $R_{\text{env}}$ can be defined as learnable PyTorch parameters (`requires_grad=True`), allowing backpropagation to estimate building thermal parameters from telemetry.

**22. How do you prevent gradient explosion during PINN composite loss backpropagation?**  
*Answer:* By applying gradient clipping (`torch.nn.utils.clip_grad_norm_`) and standardizing physical residuals into dimensionless $[-1, 1]$ numerical space.

**23. What activation functions did you use in the PINN and why?**  
*Answer:* ReLU and LeakyReLU in dense layers, and hyperbolic tangent ($\tanh$) inside the recurrent GRU cells, providing smooth continuous gradients for physical derivatives.

**24. What optimizer was used to train the PINN?**  
*Answer:* Adam optimizer with initial learning rate $\eta = 0.002$ and weight decay $10^{-5}$ for $L_2$ regularization.

**25. Does your PINN assume a single well-mixed air zone?**  
*Answer:* Yes. It operates on the 1R-1C lumped assumption, treating the conditioned building space as a single homogeneous thermal zone.

---

### SECTION 2: Deep Sequence Modeling, Transformers & Attention (Questions 26–50)

**26. What is the motivation for using a Transformer over an LSTM for microgrid forecasting?**  
*Answer:* LSTMs process sequences sequentially, suffering from vanishing gradients and information bottleneck over long sequences. Transformers process sequences in parallel using self-attention, capturing multi-step long-range temporal dependencies directly without recursive error compounding.

**27. What is the Dual-Horizon Attention mechanism introduced in this project?**  
*Answer:* It is an architecture that decouples high-frequency short-term fluctuations (solar cloud-cover and wind gusts) from low-frequency diurnal thermal cycles by running a 1D convolutional temporal branch in parallel with a Transformer self-attention encoder, fusing them through a sigmoid gating layer.

**28. Why are sinusoidal positional encodings necessary in Transformers?**  
*Answer:* Transformer self-attention is permutation-invariant—it computes pairwise dot-product affinities regardless of token order. Positional encodings inject positional timing information into the input representations.

**29. What is the lookback window and forecast horizon used in your models?**  
*Answer:* Lookback window $H = 24\,\text{hours}$ (9 input features per step); forecast horizon $P = 24\,\text{hours}$ (4 output targets: indoor temperature, building demand, solar generation, wind generation).

**30. How did you ensure zero data leakage during data preprocessing?**  
*Answer:* Chronological splitting (70% train, 15% validation, 15% test). The `StandardScaler` was fitted strictly on the training partition ($N = 6,132$) and transformed forward onto validation and test sets without computing global statistics.

**31. What is the multi-head attention equation?**  
*Answer:* $\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$. Multi-head attention concatenates multiple projection heads: $\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$.

**32. What is the significance of the scaling factor $\frac{1}{\sqrt{d_k}}$?**  
*Answer:* For large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ stabilizes gradient flow.

**33. How does your Dual-Horizon Transformer compare with standard Transformers in $R^2$?**  
*Answer:* The Dual-Horizon PINN-Transformer achieved $R^2 = 0.3866$, outperforming the standard Transformer baseline ($R^2 = 0.3672$).

**34. What are the four target variables predicted by your neural models?**  
*Answer:* 1. Indoor air temperature ($^\circ\text{C}$); 2. Total building electrical demand ($\text{kW}$); 3. Solar PV generation ($\text{kW}$); 4. Wind turbine generation ($\text{kW}$).

**35. How many trainable parameters does your Dual-Horizon Transformer contain?**  
*Answer:* Approximately 42,500 parameters, ensuring light memory footprint ($< 600\,\text{MB}$) and fast CPU training ($\approx 32\,\text{seconds}$ for 8 epochs).

**36. What is the inference latency of your proposed model?**  
*Answer:* 0.038 milliseconds per sequence on CPU, enabling real-time deployment in high-frequency SCADA loops.

**37. How did you handle multi-step forecasting—iterative or direct?**  
*Answer:* Direct multi-horizon forecasting: the dense projection head outputs all 24 future time steps simultaneously, avoiding compounding autoregressive feedback error.

**38. What loss function was used for training the non-PINN baselines?**  
*Answer:* Mean Squared Error (MSE): $\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$.

**39. Why did the Persistence Baseline achieve an $R^2$ of -1.0?**  
*Answer:* In highly dynamic microgrids with volatile solar ramps and occupancy changes, predicting that future values will simply equal the current observation produces residual variance greater than the variance of the ground truth, resulting in negative $R^2$.

**40. What is an ablation study in machine learning?**  
*Answer:* An experimental protocol where specific architectural components (e.g., physics loss, dual attention, anomaly recovery) are systematically removed to quantify their exact contribution to performance.

**41. What happened when the physics loss was ablated ($\lambda_{\text{phys}} = 0$)?**  
*Answer:* The test MAE degraded by 28.0% (from 2.237 kW to 2.863 kW), and the physical residual error jumped from 0.089 to 0.584, causing severe thermal drift.

**42. What happened when dual-horizon attention was replaced with standard attention?**  
*Answer:* Test MAE degraded by 15.0%, specifically struggling during rapid cloud-cover transients.

**43. What happened when sensor anomaly recovery was ablated under fault injections?**  
*Answer:* Test MAE degraded by 62.0%, as corrupted sensor values propagated directly into predictions.

**44. How does dropout prevent overfitting in deep time series networks?**  
*Answer:* By randomly zeroing activations with probability $p=0.10$ during training, preventing co-adaptation of hidden features.

**45. What is the purpose of the validation dataset?**  
*Answer:* To monitor validation loss across epochs, execute early stopping, and prevent overfitting on training data.

**46. Did you use batch normalization or layer normalization?**  
*Answer:* Layer normalization inside the Transformer encoder blocks, as it normalizes across features rather than batch dimensions, which is standard for sequence modeling.

**47. What is causal masking and was it used?**  
*Answer:* Causal masking sets upper triangular attention weights to $-\infty$ to prevent tokens from attending to future steps. In our direct encoder architecture, lookback history is bidirectional, while future steps are projected directly.

**48. Why is MAPE mathematically problematic when actual values approach zero?**  
*Answer:* Division by $y_i \approx 0$ (e.g., zero solar generation at night) causes division-by-zero explosions. We evaluated MAPE strictly with an epsilon threshold ($\epsilon = 1e-4$).

**49. How does your model capture seasonal variation?**  
*Answer:* Through input features including annual solar angle sinusoids, ambient temperature trends, and 8,760 hours of continuous training history.

**50. What is the difference between GRU and LSTM?**  
*Answer:* A GRU has two gates (reset and update) and merges the cell state and hidden state, making it faster to train with fewer parameters than an LSTM (which has three gates: input, forget, output).

---

### SECTION 3: Microgrid Balancing, Optimization & Battery Storage (Questions 51–75)

**51. What is the fundamental Net Microgrid Load equation?**  
*Answer:* $P_{\text{net}}(t) = P_{\text{demand}}(t) - \left(P_{\text{solar}}(t) + P_{\text{wind}}(t)\right)$.

**52. What does a negative Net Load indicate?**  
*Answer:* Renewable energy generation exceeds total building demand, creating a surplus that can be stored in the BESS or curtailed.

**53. What does a positive Net Load indicate?**  
*Answer:* Building demand exceeds renewable generation, requiring power from the battery or grid imports.

**54. What are the operational parameters of your Battery Energy Storage System (BESS)?**  
*Answer:* Capacity: $60\,\text{kWh}$; Max charge/discharge power: $20\,\text{kW}$; Round-trip efficiency: $92\%$; State of Charge bounds: $20\%$ to $90\%$.

**55. Why is battery State of Charge constrained between 20% and 90%?**  
*Answer:* To prevent deep discharging ($<20\%$) and overcharging ($>90\%$), which accelerate lithium-ion degradation, solid-electrolyte interphase (SEI) growth, and capacity fade.

**56. How do you mathematically guarantee that the battery does not charge and discharge simultaneously?**  
*Answer:* By enforcing the binary complementarity constraint: $u_{\text{chg}}(t) + u_{\text{dischg}}(t) \le 1$, where $u_{\text{chg}}, u_{\text{dischg}} \in \{0, 1\}$.

**57. What electricity tariff structure was implemented?**  
*Answer:* Time-of-Use (ToU):
* Peak Tariff: $\$0.28/\text{kWh}$ (14:00 to 20:00 on weekdays).
* Mid-Peak Tariff: $\$0.16/\text{kWh}$ (07:00 to 14:00, 20:00 to 22:00).
* Off-Peak Tariff: $\$0.09/\text{kWh}$ (overnight and weekends).

**58. How does the MPC optimizer take advantage of the ToU tariff?**  
*Answer:* It pre-charges the battery during cheap off-peak hours ($\$0.09/\text{kWh}$) and discharges during peak hours ($\$0.28/\text{kWh}$), while pre-cooling the building zone during cheap hours to reduce peak HVAC demand.

**59. What total cost savings did the MPC optimizer achieve?**  
*Answer:* Baseline uncontrolled cost: $\$30.71$ $\to$ Optimized cost: $\$8.70$, achieving a **71.67% economic cost reduction**.

**60. What peak demand shave was achieved?**  
*Answer:* A 16.7% reduction in peak grid electrical demand during peak operating hours.

**61. How much CO₂ was avoided by the optimization schedule?**  
*Answer:* 77.84 kg of CO₂ avoided over 24 hours (calculated using a grid emission factor of $0.72\,\text{kg CO}_2/\text{kWh}$).

**62. How does ASHRAE Standard 55 define thermal comfort?**  
*Answer:* As the thermal condition of mind that expresses satisfaction with the thermal environment, typically parameterized by Predicted Mean Vote (PMV) between $-0.5$ and $+0.5$, corresponding to an operative temperature deadband of $21.0^\circ\text{C}$ to $23.5^\circ\text{C}$ during occupied periods.

**63. Did your optimization violate occupant thermal comfort?**  
*Answer:* No. Comfort violations were strictly $0.0\,\text{hours}$, as the MPC objective heavily penalizes deviations outside the acceptable deadband during occupied hours.

**64. What is setback temperature control?**  
*Answer:* Relaxing the allowable temperature deadband when a building is unoccupied (e.g., widening to $[17.0^\circ\text{C}, 27.5^\circ\text{C}]$), drastically reducing HVAC energy consumption without impacting occupant comfort.

**65. How does daylight harvesting work in your lighting optimizer?**  
*Answer:* When outdoor solar global horizontal irradiance ($GHI$) exceeds $200\,\text{W/m}^2$, ambient daylight enters through glazed windows. The optimizer dims electrical lighting power from 12 kW down to 3 kW while maintaining required illuminance (500 lux).

**66. What is the optimization horizon $K$ in your MPC?**  
*Answer:* $K = 24\,\text{hours}$, aligned with the diurnal weather and electricity tariff cycle.

**67. What is Model Predictive Control (MPC)?**  
*Answer:* An advanced process control method that uses a dynamic model of the system to optimize control inputs across a future time horizon, applying the first step's control action and repeating the optimization at the next time step (receding horizon principle).

**68. Why is MPC superior to simple rule-based on/off thermostatic control?**  
*Answer:* Rule-based control reacts purely to current errors. MPC is proactive: it anticipates weather changes, occupancy arrivals, and electricity price spikes hours in advance.

**69. How is battery degradation modeled in the cost function?**  
*Answer:* Through a degradation penalty weight: $w_{\text{deg}} \cdot (P_{\text{chg}} + P_{\text{dischg}})$, penalizing unnecessary cycling and micro-charging.

**70. What is peak demand shaving?**  
*Answer:* Reducing the maximum instantaneous power drawn from the electrical utility grid during peak hours, lowering utility demand charges.

**71. What is power factor and was it considered?**  
*Answer:* Power factor is the ratio of real power ($kW$) to apparent power ($kVA$). Our model focuses on active real power ($kW$), consistent with smart building microgrid literature.

**72. What solar PV technology is assumed?**  
*Answer:* Monocrystalline silicon PV array with 50 kWp rated capacity and 85% balance-of-system efficiency factor.

**73. What wind turbine power model was used?**  
*Answer:* A cubic power curve with cut-in velocity $v_{\text{in}} = 3\,\text{m/s}$, rated velocity $v_{\text{rated}} = 11\,\text{m/s}$, and cut-out velocity $v_{\text{out}} = 25\,\text{m/s}$.

**74. What happens when wind velocity exceeds the cut-out speed ($25\,\text{m/s}$)?**  
*Answer:* The turbine automatically executes mechanical braking to prevent structural aerodynamic damage, reducing electrical output to zero.

**75. What is grid dependency percentage?**  
*Answer:* $\text{Grid Dependency} = \frac{\text{Total Grid Energy Import}}{\text{Total Building Energy Demand}} \times 100\%$. In our dashboard, this is tracked in real time.

---

### SECTION 4: Anomaly Detection, Software Architecture & Implementation (Questions 76–100)

**76. What types of IoT sensor anomalies does your system detect?**  
*Answer:* 1. Physical limit violations; 2. Sudden unphysical temperature jumps ($>3.5^\circ\text{C}/\text{hr}$); 3. Stuck sensor values ($\ge 6\,\text{hrs}$ identical); 4. High-dimensional multivariate drift.

**77. How does the Isolation Forest algorithm work?**  
*Answer:* It constructs an ensemble of isolation trees (iTrees) by randomly selecting features and split values. Anomalies reside in sparse regions and require fewer splits to isolate, resulting in significantly shorter path lengths than normal points.

**78. What contamination factor was used in the Isolation Forest?**  
*Answer:* $\alpha = 0.03$ (3% expected anomaly rate), calibrated to minimize false positive alarms on normal operational transitions.

**79. How does physical continuity imputation work?**  
*Answer:* When an anomaly or missing value is detected, the corrupted reading is replaced via localized linear interpolation bounded by thermal rate-of-change limits, preserving smooth temperature gradients.

**80. Why is mean imputation dangerous for building telemetry?**  
*Answer:* Mean imputation creates sharp artificial step-discontinuities that cause numerical derivatives ($\frac{dT}{dt}$) to spike, triggering massive false physical residual errors in downstream PINNs.

**81. What technologies comprise your backend?**  
*Answer:* Python 3.12, FastAPI, PyTorch 2.10, SQLAlchemy, PostgreSQL 18, Scikit-Learn, SciPy, and Pydantic v2.

**82. What technologies comprise your frontend?**  
*Answer:* React 18, TypeScript, Tailwind CSS, Recharts, Lucide Icons, and Vite.

**83. Why did you choose FastAPI over Flask or Django?**  
*Answer:* FastAPI provides native asynchronous request handling (ASGI), automatic OpenAPI v3.1 Swagger documentation, high performance (Starlette), and native Pydantic type validation.

**84. How is the database organized?**  
*Answer:* A 3NF normalized PostgreSQL schema with 9 tables (`users`, `datasets`, `sensor_readings`, `sensor_anomalies`, `models`, `training_experiments`, `forecast_results`, `optimization_runs`, `audit_logs`).

**85. Why do you store model weights in the filesystem instead of database BLOBs?**  
*Answer:* Storing large neural binary checkpoints ($>50\,\text{MB}$) inside relational tables causes database bloat and degrades query cache efficiency. Checkpoint files are stored in `ml/artifacts/`, and their URI paths and metadata are indexed in the database.

**86. How is user authentication implemented?**  
*Answer:* Passwords hashed using `bcrypt` (work factor 12) and stateless authentication via JSON Web Tokens (JWT) signed with HMAC-SHA256 (`HS256`).

**87. What CORS policies are implemented?**  
*Answer:* `CORSMiddleware` configured to permit cross-origin requests from frontend development origins (`http://localhost:5173`, `http://localhost:3000`).

**88. How are automated tests structured?**  
*Answer:* Using `pytest` (`tests/test_system.py`), covering ML tensor shapes, physics residual exactness, chronological leakage checks, optimization bounds, and FastAPI endpoint responses.

**89. What is the role of `pydantic` in your API?**  
*Answer:* Pydantic schemas enforce type safety and runtime validation for all API request bodies and responses (e.g., `ForecastRequest`, `OptimizationRequest`).

**90. How does the frontend communicate with the backend?**  
*Answer:* Via an `axios` HTTP client configured with a centralized baseURL (`/api/v1`) and Vite development reverse proxy.

**91. How is real-time visualization handled in the frontend?**  
*Answer:* Using `Recharts` responsive components (`AreaChart`, `LineChart`, `BarChart`) with SVG hardware acceleration.

**92. How does the system handle database migrations?**  
*Answer:* Through SQLAlchemy's `Base.metadata.create_all(bind=engine)` for automatic table provisioning, supplemented by `database/schema.sql` for production PostgreSQL deployments.

**93. What is the role of `generate_dataset.py`?**  
*Answer:* It generates an 8,760-hour calibrated smart microgrid dataset governed by 1R-1C thermal physics, solar geometry, and wind curves, providing a deterministic and reproducible benchmark.

**94. What real-world benchmark datasets did you investigate?**  
*Answer:* 1. Building Data Genome Project 2 (BDG2 / ASHRAE Great Energy Predictor III); 2. UCI Individual Household Electric Power Consumption dataset; 3. NREL National Solar Radiation Database (NSRDB).

**95. How does your system prevent overfitting during training?**  
*Answer:* 1. Chronological splitting; 2. Dropout ($p=0.10$); 3. Physics regularization acting as a structural inductive bias; 4. Early stopping on validation loss.

**96. What is the total memory footprint of the running application?**  
*Answer:* Under 1 GB total RAM (FastAPI backend $\approx 180\,\text{MB}$, PyTorch inference $\approx 400\,\text{MB}$, React frontend $\approx 120\,\text{MB}$).

**97. Can this system run on a CPU without a dedicated GPU?**  
*Answer:* Yes. PyTorch 2.10 CPU execution is fully optimized, training all 6 benchmark models in under 2 minutes total.

**98. How are API errors handled?**  
*Answer:* Using standard HTTP status codes (400 for bad input, 401 for unauthorized, 404 for missing entities, 500 for internal errors) with structured JSON error messages.

**99. How would you deploy this platform to production?**  
*Answer:* Containerized using Docker and Docker Compose: a Gunicorn/Uvicorn container for FastAPI, an Nginx reverse proxy serving the built React static bundle, and a managed PostgreSQL 18 container.

**100. What is the single most important academic contribution of your project?**  
*Answer:* Demonstrating that embedding thermodynamic 1R-1C differential equations into deep neural networks achieves an **85.3% reduction in physical residual errors** while simultaneously improving multi-step load forecasting accuracy and driving an MPC engine that saves **71.67% in energy costs** with **zero occupant thermal comfort violations**.

---
*End of 100 Technical Viva Voce Questions & Answers.*
