"""
Calibrated Smart Building and Microgrid Dataset Generator.
Synthesizes a physically consistent, calibrated 1-year (8,760 hours) smart building and microgrid dataset
governed by 1R-1C thermal thermodynamics, solar PV geometry, wind turbine power curves, and stochastic occupancy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_calibrated_microgrid_data(
    start_date: str = "2024-01-01 00:00:00",
    hours: int = 8760,
    time_step_mins: int = 60,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    
    steps = int(hours * (60 / time_step_mins))
    dt_hours = time_step_mins / 60.0
    dt_seconds = dt_hours * 3600.0
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    timestamps = [start_dt + timedelta(minutes=i * time_step_mins) for i in range(steps)]
    
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps])
    hour_of_day = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    is_weekend = np.array([1 if ts.weekday() >= 5 else 0 for ts in timestamps])
    
    # 1. Outdoor Temperature Model (Annual Sinusoid + Diurnal Sinusoid + Stochastic Weather Fronts)
    # Annual mean = 15°C, amplitude = 12°C. Diurnal amplitude = 6°C
    annual_cycle = 15.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.0)
    diurnal_cycle = 6.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
    weather_noise = np.random.normal(0, 1.2, steps)
    outdoor_temp = annual_cycle + diurnal_cycle + weather_noise
    outdoor_temp = np.round(outdoor_temp, 2)
    
    # 2. Outdoor Humidity (%)
    base_humidity = 60.0 - 15.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
    indoor_humidity = np.clip(base_humidity + np.random.normal(0, 4.0, steps), 25.0, 85.0)
    indoor_humidity = np.round(indoor_humidity, 2)
    
    # 3. Solar Global Horizontal Irradiance (GHI in W/m^2) and Solar PV Output (kW)
    # Peak solar noon is around hour 12. Solar generation only during daylight (6am - 6pm)
    solar_zenith_approx = np.clip(np.sin(np.pi * (hour_of_day - 6) / 12.0), 0, 1)
    cloud_cover = np.clip(np.random.beta(2, 5, steps), 0, 0.8)
    # Winter vs summer solar irradiance factor
    seasonal_solar_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    ghi = 1000.0 * solar_zenith_approx * seasonal_solar_factor * (1.0 - cloud_cover)
    ghi[hour_of_day < 6] = 0.0
    ghi[hour_of_day > 18] = 0.0
    
    # Solar PV capacity = 50 kWp, efficiency factor = 0.82
    pv_capacity_kw = 50.0
    solar_generation_kw = np.round((ghi / 1000.0) * pv_capacity_kw * 0.85, 3)
    
    # 4. Wind Speed (m/s) and Wind Turbine Power (kW)
    # Weibull-distributed wind speed
    wind_speed = np.random.weibull(2.1, steps) * 4.5
    wind_capacity_kw = 30.0
    v_in, v_rated, v_out = 3.0, 11.0, 25.0
    wind_generation_kw = np.zeros(steps)
    for i, v in enumerate(wind_speed):
        if v < v_in or v > v_out:
            wind_generation_kw[i] = 0.0
        elif v_in <= v < v_rated:
            wind_generation_kw[i] = wind_capacity_kw * ((v - v_in) / (v_rated - v_in))**3
        else:
            wind_generation_kw[i] = wind_capacity_kw
    wind_generation_kw = np.round(wind_generation_kw, 3)
    
    # 5. Stochastic Occupancy (People in a 3-zone commercial/academic building, max 100 people)
    occupancy_count = np.zeros(steps, dtype=int)
    for i in range(steps):
        h = hour_of_day[i]
        wknd = is_weekend[i]
        if wknd:
            if 9 <= h <= 17:
                occupancy_count[i] = int(np.random.poisson(12))
            else:
                occupancy_count[i] = int(np.random.poisson(2))
        else:
            if 8 <= h <= 18:
                occupancy_count[i] = int(np.random.normal(75, 10))
            elif 7 <= h < 8 or 18 < h <= 20:
                occupancy_count[i] = int(np.random.normal(25, 8))
            else:
                occupancy_count[i] = int(np.random.poisson(4))
    occupancy_count = np.clip(occupancy_count, 0, 100)
    
    # 6. Physical Building Envelope & Thermodynamic Integration
    # Parameters for a medium commercial building:
    # C_z: Thermal capacitance = 40.0 kWh/°C
    # R_env: Envelope thermal resistance = 2.5 °C/kW (U*A = 0.4 kW/°C)
    # Heat internal: 0.1 kW / occupant
    # Window area: 80 m^2, SHGC = 0.35
    C_z = 40.0       # kWh/°C
    R_env = 2.5     # °C/kW
    SHGC = 0.35
    A_win = 80.0    # m^2
    cop_cooling = 3.2
    cop_heating = 2.8
    target_temp = 22.0
    
    indoor_temp = np.zeros(steps)
    hvac_power_kw = np.zeros(steps)
    lighting_power_kw = np.zeros(steps)
    base_plug_load_kw = np.zeros(steps)
    
    # Initialize indoor temp
    current_T_in = 21.5
    
    for i in range(steps):
        # Base plug loads (computers, servers, kitchen)
        base_plug = 5.0 + 0.15 * occupancy_count[i] + np.random.normal(0, 0.4)
        base_plug_load_kw[i] = max(3.0, round(base_plug, 3))
        
        # Smart Daylight-Harvested Lighting
        # Maximum lighting = 15 kW. If daylight is abundant (ghi > 200), lighting is dimmed.
        daylight_lux_equiv = ghi[i] * 0.1
        if occupancy_count[i] > 5:
            dimming_ratio = np.clip(1.0 - (daylight_lux_equiv / 60.0), 0.2, 1.0)
            lighting = 12.0 * dimming_ratio + np.random.normal(0, 0.2)
        else:
            lighting = 1.5  # Emergency night lighting
        lighting_power_kw[i] = max(1.0, round(lighting, 3))
        
        # Internal Heat Gain (kW)
        Q_internal = (occupancy_count[i] * 0.1) + (0.8 * base_plug_load_kw[i]) + (0.85 * lighting_power_kw[i])
        
        # Solar Thermal Gain (kW)
        Q_solar = (ghi[i] / 1000.0) * A_win * SHGC
        
        # Envelope Conduction Heat Gain (kW)
        Q_envelope = (outdoor_temp[i] - current_T_in) / R_env
        
        # HVAC Control Simulation (Thermostatic Deadband Control around 22°C with Occupancy Awareness)
        # If occupied, deadband [21.0, 23.0]; if unoccupied, wider deadband [17.0, 27.0]
        t_min = 21.0 if occupancy_count[i] > 10 else 17.0
        t_max = 23.0 if occupancy_count[i] > 10 else 27.0
        
        # Predicted free-floating indoor temp without HVAC over dt
        free_float_dT = ((Q_envelope + Q_internal + Q_solar) / C_z) * dt_hours
        predicted_T = current_T_in + free_float_dT
        
        Q_hvac = 0.0
        hvac_p_kw = 0.0
        if predicted_T > t_max:
            # Cooling needed
            needed_cooling_kw = ((predicted_T - target_temp) * C_z) / dt_hours
            Q_hvac = -min(needed_cooling_kw, 35.0)  # Max HVAC cooling thermal cap = 35 kW
            hvac_p_kw = abs(Q_hvac) / cop_cooling
        elif predicted_T < t_min:
            # Heating needed
            needed_heating_kw = ((target_temp - predicted_T) * C_z) / dt_hours
            Q_hvac = min(needed_heating_kw, 30.0)   # Max HVAC heating thermal cap = 30 kW
            hvac_p_kw = abs(Q_hvac) / cop_heating
        
        hvac_power_kw[i] = round(hvac_p_kw, 3)
        
        # Numerical Euler update of indoor temperature governed by the exact 1R-1C equation
        actual_dT = ((Q_envelope + Q_hvac + Q_internal + Q_solar) / C_z) * dt_hours
        current_T_in = current_T_in + actual_dT + np.random.normal(0, 0.05)
        indoor_temp[i] = round(current_T_in, 2)
    
    # 7. Total Building Demand (kW)
    total_demand_kw = np.round(hvac_power_kw + lighting_power_kw + base_plug_load_kw, 3)
    
    # 8. Time of Use (ToU) Electricity Price ($/kWh)
    # Peak hours: 14:00 - 20:00 ($0.28/kWh); Mid-peak: 07:00 - 14:00 ($0.16/kWh); Off-peak: otherwise ($0.09/kWh)
    electricity_price = np.zeros(steps)
    for i in range(steps):
        h = hour_of_day[i]
        if 14 <= h < 20 and not is_weekend[i]:
            electricity_price[i] = 0.28
        elif ((7 <= h < 14) or (20 <= h < 22)) and not is_weekend[i]:
            electricity_price[i] = 0.16
        else:
            electricity_price[i] = 0.09
            
    # 9. Battery Energy Storage System (BESS) Simulation
    # BESS Capacity: 60 kWh, Max Power: 20 kW, Eff: 92%
    bess_capacity_kwh = 60.0
    soc = np.zeros(steps)
    battery_power_kw = np.zeros(steps) # positive = discharging, negative = charging
    grid_power_kw = np.zeros(steps)
    current_soc = 0.50
    
    for i in range(steps):
        renewable_gen = solar_generation_kw[i] + wind_generation_kw[i]
        net_load = total_demand_kw[i] - renewable_gen
        
        p_bat = 0.0
        # If surplus renewable generation, charge battery
        if net_load < 0 and current_soc < 0.90:
            charge_power = min(abs(net_load), 15.0, (0.90 - current_soc) * bess_capacity_kwh / dt_hours)
            current_soc += (charge_power * 0.92 * dt_hours) / bess_capacity_kwh
            p_bat = -charge_power
        # If high price / peak deficit, discharge battery
        elif net_load > 0 and electricity_price[i] >= 0.16 and current_soc > 0.20:
            discharge_power = min(net_load, 15.0, (current_soc - 0.20) * bess_capacity_kwh / dt_hours)
            current_soc -= (discharge_power / 0.92 * dt_hours) / bess_capacity_kwh
            p_bat = discharge_power
            
        battery_power_kw[i] = round(p_bat, 3)
        soc[i] = round(current_soc * 100.0, 2)
        grid_power_kw[i] = round(max(0.0, net_load - p_bat), 3)
        
    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "outdoor_temperature": outdoor_temp,
        "indoor_temperature": indoor_temp,
        "indoor_humidity": indoor_humidity,
        "solar_ghi_wm2": np.round(ghi, 2),
        "solar_generation_kw": solar_generation_kw,
        "wind_speed_ms": np.round(wind_speed, 2),
        "wind_generation_kw": wind_generation_kw,
        "occupancy_count": occupancy_count,
        "hvac_power_kw": hvac_power_kw,
        "lighting_power_kw": lighting_power_kw,
        "base_plug_load_kw": base_plug_load_kw,
        "total_demand_kw": total_demand_kw,
        "battery_soc_pct": soc,
        "battery_power_kw": battery_power_kw,
        "grid_power_kw": grid_power_kw,
        "electricity_price": electricity_price,
        "is_synthetic": 1
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "calibrated_smart_building_microgrid.csv")
    print(f"Generating 8760 hours of calibrated smart building microgrid data...")
    df = generate_calibrated_microgrid_data(hours=8760, time_step_mins=60)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")
