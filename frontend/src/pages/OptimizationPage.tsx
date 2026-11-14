import React, { useEffect, useState } from 'react';
import { EnergyAPI } from '../services/api';
import { Sliders, DollarSign, Leaf, Zap, ShieldCheck, Play, ArrowDownRight } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

export const OptimizationPage: React.FC = () => {
  const [horizon, setHorizon] = useState<number>(24);
  const [initialT, setInitialT] = useState<number>(22.0);
  const [initialSoc, setInitialSoc] = useState<number>(50.0);
  const [optData, setOptData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const executeOptimization = async () => {
    setLoading(true);
    try {
      const res = await EnergyAPI.runOptimization(horizon, initialT, initialSoc);
      // Format schedule arrays into chart objects
      const schedule = res.schedule || {};
      const chartPoints = (res.timestamps || []).map((ts: string, idx: number) => ({
        timeLabel: ts ? ts.split(' ')[1]?.slice(0, 5) || `T+${idx}h` : `T+${idx}h`,
        base_indoor_temp: schedule.base_indoor_temp?.[idx] ?? 22.0,
        opt_indoor_temp: schedule.opt_indoor_temp?.[idx] ?? 22.0,
        base_hvac_kw: schedule.base_hvac_kw?.[idx] ?? 0.0,
        opt_hvac_kw: schedule.opt_hvac_kw?.[idx] ?? 0.0,
        opt_lighting_kw: schedule.opt_lighting_kw?.[idx] ?? 0.0,
        opt_battery_kw: schedule.opt_battery_kw?.[idx] ?? 0.0,
        opt_soc_pct: schedule.opt_soc_pct?.[idx] ?? 50.0,
        base_grid_kw: schedule.base_grid_kw?.[idx] ?? 0.0,
        opt_grid_kw: schedule.opt_grid_kw?.[idx] ?? 0.0
      }));
      setOptData({ ...res, chartPoints });
    } catch (err) {
      console.error('Optimization run failed:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    executeOptimization();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Sliders className="w-6 h-6 text-cyan-400" />
            <span>Occupancy-Aware Model Predictive Control (MPC)</span>
          </h1>
          <p className="text-xs text-slate-400">
            Multi-objective optimization balancing thermal comfort, daylight-harvested lighting, and BESS arbitrage
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5">
            <span className="text-xs text-slate-400">Initial Temp:</span>
            <input
              type="number"
              step="0.5"
              value={initialT}
              onChange={(e) => setInitialT(Number(e.target.value))}
              className="w-14 bg-transparent text-xs font-semibold text-cyan-300 focus:outline-none"
            />
            <span className="text-xs text-slate-400">°C</span>
          </div>

          <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5">
            <span className="text-xs text-slate-400">Initial SoC:</span>
            <input
              type="number"
              step="5"
              value={initialSoc}
              onChange={(e) => setInitialSoc(Number(e.target.value))}
              className="w-14 bg-transparent text-xs font-semibold text-emerald-400 focus:outline-none"
            />
            <span className="text-xs text-slate-400">%</span>
          </div>

          <button
            onClick={executeOptimization}
            disabled={loading}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 font-bold text-xs hover:brightness-110 transition-all shadow-md shadow-cyan-500/20"
          >
            <Play className={`w-3.5 h-3.5 fill-current ${loading ? 'animate-spin' : ''}`} />
            <span>Run MPC Optimization</span>
          </button>
        </div>
      </div>

      {/* Savings & Performance KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1.5">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Electricity Cost Savings</span>
            <DollarSign className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">
            {optData?.energy_savings_pct ?? '--'}%
          </div>
          <div className="text-[11px] text-slate-400">
            ${optData?.cost_optimized_usd} vs ${optData?.cost_baseline_usd} baseline
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1.5">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Peak Demand Shaved</span>
            <ArrowDownRight className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-cyan-400">
            {optData?.peak_demand_reduction_pct ?? '--'}%
          </div>
          <div className="text-[11px] text-slate-400">
            {optData?.peak_grid_optimized_kw} kW vs {optData?.peak_grid_baseline_kw} kW
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1.5">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>CO₂ Emissions Avoided</span>
            <Leaf className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-2xl font-bold text-teal-300">
            {optData?.co2_reduction_kg ?? '--'} <span className="text-xs font-normal text-slate-400">kg</span>
          </div>
          <div className="text-[11px] text-slate-400">
            Factor: 0.72 kg CO₂/kWh grid mix
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1.5">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Thermal Comfort Violations</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">
            {optData?.comfort_violation_hours ?? 0.0} <span className="text-xs font-normal text-slate-400">hrs</span>
          </div>
          <div className="text-[11px] text-slate-400">
            Strict ASHRAE-55 adherence
          </div>
        </div>
      </div>

      {/* Grid Import Shave Schedule Comparison */}
      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white">Grid Electricity Import: Baseline vs Optimized MPC (kW)</h2>
            <p className="text-xs text-slate-400">Shifting load away from expensive ToU peak periods using battery dispatch</p>
          </div>
          <div className="flex items-center space-x-3 text-xs">
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
              <span className="text-slate-300">Optimized Grid Import</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-red-400"></span>
              <span className="text-slate-400">Uncontrolled Baseline</span>
            </span>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={optData?.chartPoints || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              <Area type="monotone" dataKey="base_grid_kw" name="Baseline Grid Power (kW)" stroke="#f87171" fill="#f87171" fillOpacity={0.15} />
              <Area type="monotone" dataKey="opt_grid_kw" name="Optimized Grid Power (kW)" stroke="#34d399" fill="#34d399" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* BESS State of Charge & Thermal Setpoint Tracking */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-white">BESS State of Charge (% SoC) Trajectory</h2>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={optData?.chartPoints || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 100]} stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Line type="monotone" dataKey="opt_soc_pct" name="Battery SoC (%)" stroke="#a78bfa" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-white">Indoor Temperature Tracking Under MPC (°C)</h2>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={optData?.chartPoints || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis domain={[18, 26]} stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Line type="monotone" dataKey="opt_indoor_temp" name="Optimized Temp (°C)" stroke="#38bdf8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="base_indoor_temp" name="Baseline Temp (°C)" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
