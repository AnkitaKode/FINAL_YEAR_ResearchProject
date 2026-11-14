import React, { useEffect, useState } from 'react';
import { EnergyAPI } from '../services/api';
import { TrendingUp, BarChart2, CheckCircle2, SlidersHorizontal, AlertCircle } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

export const ForecastingPage: React.FC = () => {
  const [modelType, setModelType] = useState<string>('Proposed_DualHorizon_PINN');
  const [horizon, setHorizon] = useState<number>(24);
  const [models, setModels] = useState<any[]>([]);
  const [forecastData, setForecastData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    EnergyAPI.getModels().then(setModels).catch(console.error);
  }, []);

  const runForecast = async () => {
    setLoading(true);
    try {
      const data = await EnergyAPI.getForecast(modelType, horizon);
      // Format timestamps
      const formatted = (data.forecast || []).map((d: any) => ({
        ...d,
        timeLabel: d.timestamp ? d.timestamp.split(' ')[1]?.slice(0, 5) || d.timestamp : ''
      }));
      setForecastData({ ...data, forecast: formatted });
    } catch (err) {
      console.error('Forecast error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runForecast();
  }, [modelType, horizon]);

  const metrics = forecastData?.metrics || {};

  return (
    <div className="space-y-6">
      {/* Top Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
            <TrendingUp className="w-6 h-6 text-cyan-400" />
            <span>Multi-Horizon Microgrid Load & Generation Forecasting</span>
          </h1>
          <p className="text-xs text-slate-400">
            Compare physics-regularized neural predictions against empirical ground truth
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5">
            <span className="text-xs text-slate-400 font-medium">Model:</span>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="bg-transparent text-xs font-semibold text-cyan-300 focus:outline-none cursor-pointer"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} className="bg-slate-900 text-slate-200">
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5">
            <span className="text-xs text-slate-400 font-medium">Horizon:</span>
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="bg-transparent text-xs font-semibold text-emerald-400 focus:outline-none cursor-pointer"
            >
              <option value={12} className="bg-slate-900">12 Hours</option>
              <option value={24} className="bg-slate-900">24 Hours (Diurnal)</option>
              <option value={48} className="bg-slate-900">48 Hours (Multi-Day)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Model Accuracy Metrics Banner */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-[11px] text-slate-400 block font-medium">Test MAE</span>
          <span className="text-xl font-bold text-cyan-400">{metrics.test_mae ?? '--'}</span>
          <span className="text-[10px] text-slate-400 block">Lower is better</span>
        </div>
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-[11px] text-slate-400 block font-medium">Test RMSE</span>
          <span className="text-xl font-bold text-white">{metrics.test_rmse ?? '--'}</span>
          <span className="text-[10px] text-slate-400 block">Root mean square error</span>
        </div>
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-[11px] text-slate-400 block font-medium">R² Goodness of Fit</span>
          <span className="text-xl font-bold text-emerald-400">{metrics.test_r2 ?? '--'}</span>
          <span className="text-[10px] text-slate-400 block">Variance explained</span>
        </div>
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-[11px] text-slate-400 block font-medium">Physics Residual Error</span>
          <span className="text-xl font-bold text-amber-400">{metrics.physics_residual_error ?? '--'}</span>
          <span className="text-[10px] text-slate-400 block">ODE conservation residual</span>
        </div>
      </div>

      {/* Primary Forecast Chart: Actual vs Predicted Demand */}
      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white">Total Building Electricity Demand: Predicted vs Actual (kW)</h2>
            <p className="text-xs text-slate-400">Comparing active model trajectory with chronological holdout ground truth</p>
          </div>
          <div className="flex items-center space-x-3 text-xs">
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
              <span className="text-slate-300">Predicted (kW)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
              <span className="text-slate-400">Actual (kW)</span>
            </span>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecastData?.forecast || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              <Line type="monotone" dataKey="predicted_demand_kw" name="Predicted Demand (kW)" stroke="#22d3ee" strokeWidth={2.5} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="actual_demand_kw" name="Actual Demand (kW)" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Secondary Chart: Net Microgrid Load Forecast */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-white">Net Microgrid Load Forecast (Demand - Renewables)</h2>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={forecastData?.forecast || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Line type="monotone" dataKey="predicted_net_load_kw" name="Net Load (kW)" stroke="#fbbf24" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-white">Indoor Temperature Thermal Forecast (°C)</h2>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={forecastData?.forecast || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={['dataMin - 1', 'dataMax + 1']} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Line type="monotone" dataKey="predicted_indoor_temp" name="Predicted Temp (°C)" stroke="#38bdf8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="actual_indoor_temp" name="Actual Temp (°C)" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
