import React, { useEffect, useState } from 'react';
import { EnergyAPI } from '../services/api';
import { BarChart3, CheckCircle, Award, Layers, AlertCircle, ArrowUpRight } from 'lucide-react';

export const PerformancePage: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    EnergyAPI.getBenchmarks()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const benchmarks = data?.benchmarks || {};
  const ablations = data?.ablations || {};

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <BarChart3 className="w-6 h-6 text-cyan-400" />
          <span>Quantitative Model Benchmarking & Empirical Ablation Studies</span>
        </h1>
        <p className="text-xs text-slate-400">
          Evaluated strictly on held-out chronological test split (1,267 unseen time steps) with reproducible metrics
        </p>
      </div>

      {/* Main Comparative Benchmark Table */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <Award className="w-5 h-5 text-amber-400" />
            <span>Multi-Model Comparative Performance Matrix</span>
          </h2>
          <span className="text-xs text-slate-400 font-mono">Test Set: 1,267 sequences</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60 text-slate-300 font-semibold uppercase tracking-wider text-[10px]">
              <tr>
                <th className="p-3">Model Architecture</th>
                <th className="p-3">Test MAE</th>
                <th className="p-3">Test RMSE</th>
                <th className="p-3">R² Score</th>
                <th className="p-3">Physical Residual Error</th>
                <th className="p-3">Train Latency</th>
                <th className="p-3">Inference (ms)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {Object.entries(benchmarks).map(([key, m]: [string, any]) => {
                const isProposed = key === 'Proposed_DualHorizon_PINN';
                const isPINN = key === 'PINN_Thermal';
                return (
                  <tr
                    key={key}
                    className={`hover:bg-slate-800/40 transition-all ${
                      isProposed ? 'bg-cyan-500/10 font-bold border-l-4 border-cyan-400' : ''
                    }`}
                  >
                    <td className="p-3 flex items-center space-x-2">
                      {isProposed && <Award className="w-4 h-4 text-cyan-400 flex-shrink-0" />}
                      <span className={isProposed ? 'text-cyan-300' : isPINN ? 'text-emerald-300' : 'text-slate-200'}>
                        {m.model_name}
                      </span>
                    </td>
                    <td className={`p-3 font-mono ${isProposed ? 'text-cyan-400 font-bold' : 'text-slate-300'}`}>
                      {m.test_mae}
                    </td>
                    <td className="p-3 font-mono text-slate-300">{m.test_rmse}</td>
                    <td className={`p-3 font-mono ${m.test_r2 > 0.35 ? 'text-emerald-400' : 'text-slate-400'}`}>
                      {m.test_r2}
                    </td>
                    <td className={`p-3 font-mono font-bold ${m.physics_residual_error < 0.15 ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {m.physics_residual_error}
                    </td>
                    <td className="p-3 text-slate-400">{m.training_time_sec} s</td>
                    <td className="p-3 text-slate-400">{m.inference_latency_ms} ms</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Ablation Studies Cards */}
      <div className="space-y-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <span>Ablation Experiments: Isolating Individual Component Contributions</span>
          </h2>
          <p className="text-xs text-slate-400">
            Systematic removal of physical constraints, dual attention, and sensor pre-filtering
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(ablations).map(([key, ab]: [string, any]) => (
            <div key={key} className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="flex items-start justify-between">
                <h3 className="text-sm font-bold text-white">{key.replace(/_/g, ' ')}</h3>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  key.includes('Full') ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                }`}>
                  {key.includes('Full') ? 'Optimal' : 'Degraded Variant'}
                </span>
              </div>
              <p className="text-xs text-slate-400">{ab.configuration}</p>
              
              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800 text-[11px]">
                <div>
                  <span className="text-slate-400 block">MAE:</span>
                  <span className="font-mono font-bold text-slate-200">{ab.test_mae}</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Physics Error:</span>
                  <span className="font-mono font-bold text-amber-300">{ab.physics_residual_error}</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Empirical Impact:</span>
                  <span className="font-semibold text-cyan-300">{ab.performance_delta}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
