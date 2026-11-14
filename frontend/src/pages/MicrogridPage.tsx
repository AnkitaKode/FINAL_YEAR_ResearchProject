import React, { useEffect, useState } from 'react';
import { EnergyAPI } from '../services/api';
import { Zap, Sun, Wind, Battery, ArrowRightLeft, ShieldCheck } from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';

export const MicrogridPage: React.FC = () => {
  const [telemetry, setTelemetry] = useState<any[]>([]);
  const [kpis, setKpis] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([EnergyAPI.getKPIs(), EnergyAPI.getTelemetry(48)])
      .then(([kpiRes, telemRes]) => {
        setKpis(kpiRes);
        const formatted = telemRes.map((d: any) => ({
          ...d,
          timeLabel: d.timestamp ? d.timestamp.split(' ')[1]?.slice(0, 5) || d.timestamp : ''
        }));
        setTelemetry(formatted);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Zap className="w-6 h-6 text-cyan-400" />
          <span>Microgrid Power Balancing & Energy Storage (BESS)</span>
        </h1>
        <p className="text-xs text-slate-400">
          Governing law: Net Load = Building Demand - (Solar PV + Wind Generation)
        </p>
      </div>

      {/* Real-Time Microgrid Status Flow Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900/90 to-cyan-950/30 border border-slate-800">
        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">
          Instantaneous Microgrid Node Dispatch
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50">
            <Sun className="w-5 h-5 text-amber-400 mx-auto mb-1" />
            <span className="text-xs text-slate-400 block">Solar PV</span>
            <span className="text-lg font-bold text-white">{kpis?.current_solar_kw ?? 0} kW</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50">
            <Wind className="w-5 h-5 text-emerald-400 mx-auto mb-1" />
            <span className="text-xs text-slate-400 block">Wind Turbine</span>
            <span className="text-lg font-bold text-white">{kpis?.current_wind_kw ?? 0} kW</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50">
            <Battery className="w-5 h-5 text-violet-400 mx-auto mb-1" />
            <span className="text-xs text-slate-400 block">Battery (BESS)</span>
            <span className="text-lg font-bold text-violet-300">{kpis?.current_battery_soc ?? 0}%</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50">
            <Zap className="w-5 h-5 text-cyan-400 mx-auto mb-1" />
            <span className="text-xs text-slate-400 block">Building Demand</span>
            <span className="text-lg font-bold text-cyan-300">{kpis?.current_demand_kw ?? 0} kW</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 col-span-2 md:col-span-1">
            <ArrowRightLeft className="w-5 h-5 text-teal-400 mx-auto mb-1" />
            <span className="text-xs text-slate-400 block">Grid Dependency</span>
            <span className="text-lg font-bold text-teal-300">{kpis?.grid_dependency_pct ?? 0}%</span>
          </div>
        </div>
      </div>

      {/* Main Net Load Flow Chart */}
      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <h2 className="text-base font-bold text-white">48-Hour Microgrid Power Routing & Net Load Trajectory</h2>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={telemetry}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              <Area type="monotone" dataKey="total_demand_kw" name="Building Load (kW)" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.2} />
              <Area type="monotone" dataKey="solar_generation_kw" name="Solar Gen (kW)" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
              <Area type="monotone" dataKey="wind_generation_kw" name="Wind Gen (kW)" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
              <Line type="monotone" dataKey="net_load_kw" name="Net Load (kW)" stroke="#ec4899" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
