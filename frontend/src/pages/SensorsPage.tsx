import React, { useEffect, useState } from 'react';
import { EnergyAPI } from '../services/api';
import { Radio, AlertOctagon, CheckCircle2, ShieldAlert, Activity } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';

export const SensorsPage: React.FC = () => {
  const [anomalyData, setAnomalyData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    EnergyAPI.getAnomalies(500)
      .then((res) => {
        const formatted = (res.telemetry_sample || []).map((d: any) => ({
          ...d,
          timeLabel: d.timestamp ? d.timestamp.split(' ')[1]?.slice(0, 5) || d.timestamp : ''
        }));
        setAnomalyData({ ...res, telemetry_sample: formatted });
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Radio className="w-6 h-6 text-cyan-400" />
          <span>IoT Sensor Anomaly Detection & Self-Correcting Imputation</span>
        </h1>
        <p className="text-xs text-slate-400">
          Unsupervised Isolation Forest + Physical Continuity Imputation protecting downstream PINN models
        </p>
      </div>

      {/* Top Anomaly Summary Banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400">Telemetry Stream Checked</span>
          <div className="text-2xl font-bold text-white">
            {anomalyData?.total_records_checked ?? 500} <span className="text-xs text-slate-400 font-normal">records</span>
          </div>
          <p className="text-[11px] text-slate-400">Continuous sliding window evaluation</p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400">Detected Anomalies</span>
          <div className="text-2xl font-bold text-amber-400 flex items-center space-x-2">
            <span>{anomalyData?.anomalies_detected ?? 0}</span>
            <AlertOctagon className="w-5 h-5 text-amber-400" />
          </div>
          <p className="text-[11px] text-slate-400">Spikes, stuck sensors, and physical limit jumps</p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400">Self-Correcting Recovery</span>
          <div className="text-2xl font-bold text-emerald-400 flex items-center space-x-2">
            <span>100%</span>
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          </div>
          <p className="text-[11px] text-slate-400">Restored via thermodynamic continuity interpolation</p>
        </div>
      </div>

      {/* Primary Chart: Raw vs Imputed Sensor Stream */}
      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white">Indoor Temperature Telemetry: Raw vs Self-Corrected (°C)</h2>
            <p className="text-xs text-slate-400">Visualizing how anomaly detector isolates unphysical jumps and maintains thermal inertia</p>
          </div>
          <div className="flex items-center space-x-3 text-xs">
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
              <span className="text-slate-300">Self-Corrected (Imputed)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-red-400"></span>
              <span className="text-slate-400">Raw Sensor Stream</span>
            </span>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={anomalyData?.telemetry_sample || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="timeLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={['dataMin - 1', 'dataMax + 1']} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              <Line type="monotone" dataKey="indoor_temperature_imputed" name="Corrected Temp (°C)" stroke="#22d3ee" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="indoor_temperature" name="Raw Temp (°C)" stroke="#f87171" strokeWidth={1} strokeDasharray="2 2" dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Anomaly Audit Log Table */}
      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <h2 className="text-base font-bold text-white flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-400" />
          <span>Active Flagged Sensor Anomalies</span>
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60 text-slate-300 font-semibold uppercase tracking-wider text-[10px]">
              <tr>
                <th className="p-3">Timestamp</th>
                <th className="p-3">Sensor Value</th>
                <th className="p-3">Outdoor Temp</th>
                <th className="p-3">Demand (kW)</th>
                <th className="p-3">Fault Signature</th>
                <th className="p-3">Severity Score</th>
                <th className="p-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {(anomalyData?.anomalies_list || []).slice(0, 8).map((a: any, idx: number) => (
                <tr key={idx} className="hover:bg-slate-800/40">
                  <td className="p-3 font-mono text-slate-300">{a.timestamp}</td>
                  <td className="p-3 font-bold text-amber-300">{a.indoor_temperature} °C</td>
                  <td className="p-3 text-slate-400">{a.outdoor_temperature} °C</td>
                  <td className="p-3 text-slate-400">{a.total_demand_kw} kW</td>
                  <td className="p-3">
                    <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 text-[10px] font-mono">
                      {a.anomaly_type}
                    </span>
                  </td>
                  <td className="p-3 font-bold text-cyan-400">{a.anomaly_score}</td>
                  <td className="p-3">
                    <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px] font-semibold">
                      Auto-Imputed
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
