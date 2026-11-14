import React, { useEffect, useState } from 'react';
import { EnergyAPI } from '../services/api';
import {
  Activity,
  Sun,
  Wind,
  BatteryCharging,
  Thermometer,
  Users,
  Zap,
  TrendingDown,
  RefreshCw
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

export const DashboardPage: React.FC = () => {
  const [kpis, setKpis] = useState<any>(null);
  const [telemetry, setTelemetry] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [kpiRes, telemRes] = await Promise.all([
        EnergyAPI.getKPIs(),
        EnergyAPI.getTelemetry(48)
      ]);
      setKpis(kpiRes);
      // Format timestamps for chart readability
      const formatted = telemRes.map((d: any) => ({
        ...d,
        timeLabel: d.timestamp ? d.timestamp.split(' ')[1]?.slice(0, 5) || d.timestamp : ''
      }));
      setTelemetry(formatted);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading && !kpis) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center space-y-3">
          <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="text-sm text-slate-400">Loading Live Telemetry & PINN Inference...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Activity className="w-6 h-6 text-cyan-400" />
            <span>Smart Microgrid Telemetry & Live KPIs</span>
          </h1>
          <p className="text-xs text-slate-400">
            Real-time multi-channel sensor telemetry with coupled thermal-electrical state estimation
          </p>
        </div>
        <button
          onClick={fetchData}
          className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition-all self-start sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Telemetry</span>
        </button>
      </div>

      {/* KPI Stat Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Building Demand</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {kpis?.current_demand_kw ?? '--'} <span className="text-xs font-normal text-slate-400">kW</span>
          </div>
          <div className="text-[11px] text-slate-400 flex items-center space-x-1">
            <span>Daily Consumption:</span>
            <span className="font-semibold text-cyan-300">{kpis?.daily_total_demand_kwh ?? '--'} kWh</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Solar & Wind Gen</span>
            <Sun className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">
            {((kpis?.current_solar_kw || 0) + (kpis?.current_wind_kw || 0)).toFixed(1)} <span className="text-xs font-normal text-slate-400">kW</span>
          </div>
          <div className="text-[11px] text-slate-400 flex items-center space-x-1">
            <span>Solar: {kpis?.current_solar_kw ?? 0} kW</span>
            <span>• Wind: {kpis?.current_wind_kw ?? 0} kW</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Indoor Thermal State</span>
            <Thermometer className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {kpis?.current_indoor_temp ?? '--'} <span className="text-xs font-normal text-slate-400">°C</span>
          </div>
          <div className="text-[11px] text-slate-400 flex items-center space-x-1">
            <span>Outdoor: {kpis?.current_outdoor_temp ?? '--'}°C</span>
            <span>• Occ: {kpis?.current_occupancy ?? 0} heads</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Battery Storage (BESS)</span>
            <BatteryCharging className="w-4 h-4 text-violet-400" />
          </div>
          <div className="text-2xl font-bold text-violet-300">
            {kpis?.current_battery_soc ?? '--'} <span className="text-xs font-normal text-slate-400">% SoC</span>
          </div>
          <div className="text-[11px] text-slate-400 flex items-center space-x-1">
            <span>Renewable Share:</span>
            <span className="font-semibold text-emerald-400">{kpis?.renewable_fraction_pct ?? 0}%</span>
          </div>
        </div>
      </div>

      {/* Main Microgrid Power Balance Chart */}
      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-white">Microgrid Power Flow & Net Load Balance (kW)</h2>
            <p className="text-xs text-slate-400">Total Building Demand vs Combined Renewable Generation</p>
          </div>
          <div className="flex items-center space-x-4 text-xs">
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
              <span className="text-slate-300">Demand (kW)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
              <span className="text-slate-300">Renewables (kW)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
              <span className="text-slate-300">Net Load (kW)</span>
            </span>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={telemetry}>
              <defs>
                <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorRen" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#34d399" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              <Area type="monotone" dataKey="total_demand_kw" name="Total Demand (kW)" stroke="#22d3ee" strokeWidth={2} fillOpacity={1} fill="url(#colorDemand)" />
              <Area type="monotone" dataKey="renewable_gen_kw" name="Renewable Gen (kW)" stroke="#34d399" strokeWidth={2} fillOpacity={1} fill="url(#colorRen)" />
              <Line type="monotone" dataKey="net_load_kw" name="Net Load (kW)" stroke="#fbbf24" strokeWidth={1.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Secondary Chart: Thermodynamic Thermal State Tracking */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-white">Indoor vs Outdoor Thermal Trajectory (°C)</h2>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetry}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={['dataMin - 2', 'dataMax + 2']} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Line type="monotone" dataKey="indoor_temperature" name="Indoor Temp (°C)" stroke="#38bdf8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="outdoor_temperature" name="Outdoor Temp (°C)" stroke="#f87171" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-white">HVAC Power & Occupancy Load Correlation</h2>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={telemetry}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="left" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Area yAxisId="left" type="monotone" dataKey="hvac_power_kw" name="HVAC Power (kW)" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.2} />
                <Line yAxisId="right" type="stepAfter" dataKey="occupancy_count" name="Occupancy (heads)" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
