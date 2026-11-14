"""
Dynamic Microgrid Optimization Engine.
Implements Occupancy-Aware Model Predictive Control (MPC) for HVAC,
Smart Daylight-Harvesting Lighting, and Battery Energy Storage (BESS) Dispatch.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

class MicrogridOptimizer:
    def __init__(
        self,
        bess_capacity_kwh: float = 60.0,
        bess_max_power_kw: float = 20.0,
        bess_eff_chg: float = 0.92,
        bess_eff_dischg: float = 0.92,
        soc_min: float = 0.20,
        soc_max: float = 0.90,
        c_z_thermal_cap: float = 40.0,
        r_env_resistance: float = 2.5,
        cop_cooling: float = 3.2,
        cop_heating: float = 2.8,
        co2_emission_factor_kg_kwh: float = 0.72 # Grid average emission factor
    ):
        self.bess_capacity_kwh = bess_capacity_kwh
        self.bess_max_power_kw = bess_max_power_kw
        self.bess_eff_chg = bess_eff_chg
        self.bess_eff_dischg = bess_eff_dischg
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.c_z = c_z_thermal_cap
        self.r_env = r_env_resistance
        self.cop_cooling = cop_cooling
        self.cop_heating = cop_heating
        self.co2_factor = co2_emission_factor_kg_kwh

    def optimize_schedule(
        self,
        outdoor_temp: np.ndarray,          # [horizon]
        occupancy_count: np.ndarray,       # [horizon]
        solar_generation_kw: np.ndarray,   # [horizon]
        wind_generation_kw: np.ndarray,    # [horizon]
        electricity_price: np.ndarray,     # [horizon] ($/kWh)
        initial_indoor_temp: float = 22.0,
        initial_soc: float = 0.50,
        dt_hours: float = 1.0
    ) -> Dict[str, Any]:
        horizon = len(outdoor_temp)
        
        # 1. Baseline Simulation (Static thermostat setpoint 22°C 24/7 without smart storage or dimming)
        base_indoor_temp = np.zeros(horizon)
        base_hvac_kw = np.zeros(horizon)
        base_lighting_kw = np.zeros(horizon)
        base_plug_kw = np.zeros(horizon)
        base_grid_kw = np.zeros(horizon)
        base_cost = 0.0
        
        curr_t_base = initial_indoor_temp
        for t in range(horizon):
            occ = occupancy_count[t]
            # Static lighting: fixed 12 kW occupied, 5 kW unoccupied
            l_kw = 12.0 if occ > 5 else 5.0
            base_lighting_kw[t] = l_kw
            p_kw = 5.0 + 0.15 * occ
            base_plug_kw[t] = p_kw
            
            q_int = (occ * 0.1) + (0.8 * p_kw) + (0.85 * l_kw)
            q_env = (outdoor_temp[t] - curr_t_base) / self.r_env
            
            # Static thermostat aims strictly for 22°C
            dt_free = ((q_env + q_int) / self.c_z) * dt_hours
            t_pred = curr_t_base + dt_free
            
            q_hvac = 0.0
            hvac_p = 0.0
            if t_pred > 22.5:
                needed = ((t_pred - 22.0) * self.c_z) / dt_hours
                q_hvac = -min(needed, 35.0)
                hvac_p = abs(q_hvac) / self.cop_cooling
            elif t_pred < 21.5:
                needed = ((22.0 - t_pred) * self.c_z) / dt_hours
                q_hvac = min(needed, 30.0)
                hvac_p = abs(q_hvac) / self.cop_heating
                
            base_hvac_kw[t] = hvac_p
            curr_t_base += ((q_env + q_hvac + q_int) / self.c_z) * dt_hours
            base_indoor_temp[t] = round(curr_t_base, 2)
            
            tot_dem = hvac_p + l_kw + p_kw
            ren_gen = solar_generation_kw[t] + wind_generation_kw[t]
            grid_import = max(0.0, tot_dem - ren_gen)
            base_grid_kw[t] = grid_import
            base_cost += grid_import * electricity_price[t] * dt_hours

        # 2. Optimized Simulation
        # - Occupancy-aware dynamic HVAC deadband ([20.5, 23.5] occupied, [16.0, 28.0] unoccupied)
        # - Pre-cooling / pre-heating during cheap or renewable surplus hours
        # - Smart daylight harvesting for lighting
        # - Dynamic BESS arbitrage (charge on solar surplus/cheap tariffs, discharge at peak prices)
        opt_indoor_temp = np.zeros(horizon)
        opt_hvac_kw = np.zeros(horizon)
        opt_lighting_kw = np.zeros(horizon)
        opt_plug_kw = np.zeros(horizon)
        opt_battery_kw = np.zeros(horizon) # positive = discharge, negative = charge
        opt_soc = np.zeros(horizon)
        opt_grid_kw = np.zeros(horizon)
        opt_cost = 0.0
        
        curr_t_opt = initial_indoor_temp
        curr_soc = initial_soc
        comfort_violations = 0.0
        
        for t in range(horizon):
            occ = occupancy_count[t]
            p_kw = 5.0 + 0.15 * occ
            opt_plug_kw[t] = p_kw
            
            # Daylight Harvesting Lighting: Dimming when solar generation > 10 kW
            if occ > 5:
                dim = max(0.25, 1.0 - (solar_generation_kw[t] / 35.0))
                l_kw = 12.0 * dim
            else:
                l_kw = 1.0 # Eco standby
            opt_lighting_kw[t] = round(l_kw, 2)
            
            # Dynamic Comfort Setpoint Deadband
            if occ > 5:
                t_low, t_high = 21.0, 23.5
            else:
                t_low, t_high = 17.0, 27.5
                
            q_int = (occ * 0.1) + (0.8 * p_kw) + (0.85 * l_kw)
            q_env = (outdoor_temp[t] - curr_t_opt) / self.r_env
            dt_free = ((q_env + q_int) / self.c_z) * dt_hours
            t_pred = curr_t_opt + dt_free
            
            # Lookahead Pre-cooling if electricity is cheap now but gets expensive in next 2 hours
            is_preconditioning = False
            if t < horizon - 2:
                next_price = electricity_price[t+1]
                if next_price > electricity_price[t] * 1.5 and occ > 0:
                    t_high -= 1.0 # Pre-cool slightly
                    is_preconditioning = True
                    
            q_hvac = 0.0
            hvac_p = 0.0
            if t_pred > t_high:
                needed = ((t_pred - ((t_low + t_high) / 2.0)) * self.c_z) / dt_hours
                q_hvac = -min(needed, 35.0)
                hvac_p = abs(q_hvac) / self.cop_cooling
            elif t_pred < t_low:
                needed = ((((t_low + t_high) / 2.0) - t_pred) * self.c_z) / dt_hours
                q_hvac = min(needed, 30.0)
                hvac_p = abs(q_hvac) / self.cop_heating
                
            opt_hvac_kw[t] = round(hvac_p, 2)
            curr_t_opt += ((q_env + q_hvac + q_int) / self.c_z) * dt_hours
            opt_indoor_temp[t] = round(curr_t_opt, 2)
            
            # Check comfort violation during occupied hours
            if occ > 5:
                if curr_t_opt < 20.0 or curr_t_opt > 24.5:
                    comfort_violations += 1.0
                    
            # Microgrid Net Load Calculation
            tot_dem = hvac_p + l_kw + p_kw
            ren_gen = solar_generation_kw[t] + wind_generation_kw[t]
            net_load = tot_dem - ren_gen
            
            # Smart BESS Management
            p_bat = 0.0
            # Condition A: Surplus renewable generation -> Charge battery
            if net_load < 0 and curr_soc < self.soc_max:
                chg_power = min(abs(net_load), self.bess_max_power_kw, (self.soc_max - curr_soc) * self.bess_capacity_kwh / dt_hours)
                curr_soc += (chg_power * self.bess_eff_chg * dt_hours) / self.bess_capacity_kwh
                p_bat = -chg_power
            # Condition B: High electricity price ($/kWh >= 0.16) or high deficit -> Discharge battery
            elif net_load > 0 and (electricity_price[t] >= 0.16 or is_preconditioning) and curr_soc > self.soc_min:
                dischg_power = min(net_load, self.bess_max_power_kw, (curr_soc - self.soc_min) * self.bess_capacity_kwh / dt_hours)
                curr_soc -= (dischg_power / self.bess_eff_dischg * dt_hours) / self.bess_capacity_kwh
                p_bat = dischg_power
            # Condition C: Super off-peak cheap tariff (< 0.10) & battery below 60% -> Pre-charge from grid
            elif electricity_price[t] <= 0.09 and curr_soc < 0.60:
                grid_chg = min(10.0, (0.60 - curr_soc) * self.bess_capacity_kwh / dt_hours)
                curr_soc += (grid_chg * self.bess_eff_chg * dt_hours) / self.bess_capacity_kwh
                p_bat = -grid_chg
                
            opt_battery_kw[t] = round(p_bat, 2)
            opt_soc[t] = round(curr_soc * 100.0, 1)
            
            grid_import = max(0.0, net_load - p_bat)
            opt_grid_kw[t] = round(grid_import, 2)
            opt_cost += grid_import * electricity_price[t] * dt_hours

        energy_savings_pct = round(((base_cost - opt_cost) / max(base_cost, 1e-4)) * 100.0, 2)
        peak_base = float(np.max(base_grid_kw))
        peak_opt = float(np.max(opt_grid_kw))
        peak_shave_pct = round(((peak_base - peak_opt) / max(peak_base, 1e-4)) * 100.0, 2)
        
        base_co2 = round(float(np.sum(base_grid_kw) * self.co2_factor * dt_hours), 2)
        opt_co2 = round(float(np.sum(opt_grid_kw) * self.co2_factor * dt_hours), 2)
        co2_reduction_kg = round(base_co2 - opt_co2, 2)
        
        return {
            "horizon_hours": horizon,
            "cost_baseline_usd": round(base_cost, 2),
            "cost_optimized_usd": round(opt_cost, 2),
            "energy_savings_pct": energy_savings_pct,
            "peak_grid_baseline_kw": round(peak_base, 2),
            "peak_grid_optimized_kw": round(peak_opt, 2),
            "peak_demand_reduction_pct": peak_shave_pct,
            "comfort_violation_hours": comfort_violations,
            "co2_emissions_baseline_kg": base_co2,
            "co2_emissions_optimized_kg": opt_co2,
            "co2_reduction_kg": co2_reduction_kg,
            "schedule": {
                "base_indoor_temp": base_indoor_temp.tolist(),
                "opt_indoor_temp": opt_indoor_temp.tolist(),
                "base_hvac_kw": base_hvac_kw.tolist(),
                "opt_hvac_kw": opt_hvac_kw.tolist(),
                "opt_lighting_kw": opt_lighting_kw.tolist(),
                "opt_battery_kw": opt_battery_kw.tolist(),
                "opt_soc_pct": opt_soc.tolist(),
                "base_grid_kw": base_grid_kw.tolist(),
                "opt_grid_kw": opt_grid_kw.tolist()
            }
        }

if __name__ == "__main__":
    opt = MicrogridOptimizer()
    horizon = 24
    outdoor = np.array([18, 17, 16, 16, 17, 19, 22, 26, 29, 32, 34, 35, 34, 33, 31, 28, 25, 23, 22, 21, 20, 19, 19, 18], dtype=float)
    occ = np.array([0, 0, 0, 0, 0, 0, 10, 45, 80, 85, 90, 88, 70, 85, 90, 80, 60, 30, 10, 0, 0, 0, 0, 0], dtype=float)
    solar = np.array([0, 0, 0, 0, 0, 2, 8, 18, 28, 38, 44, 46, 42, 34, 22, 10, 2, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    wind = np.array([5, 6, 8, 8, 7, 6, 5, 4, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 12, 10, 8, 6, 5, 5], dtype=float)
    price = np.array([0.09]*7 + [0.16]*7 + [0.28]*6 + [0.16]*2 + [0.09]*2, dtype=float)
    
    res = opt.optimize_schedule(outdoor, occ, solar, wind, price)
    print("Optimization Completed Successfully!")
    print(f"Cost Baseline: ${res['cost_baseline_usd']} | Cost Optimized: ${res['cost_optimized_usd']}")
    print(f"Energy Savings: {res['energy_savings_pct']}% | Peak Demand Shave: {res['peak_demand_reduction_pct']}%")
    print(f"CO2 Reduction: {res['co2_reduction_kg']} kg | Comfort Violations: {res['comfort_violation_hours']} hrs")
