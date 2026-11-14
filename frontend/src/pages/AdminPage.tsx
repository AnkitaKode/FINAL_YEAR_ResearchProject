import React from 'react';
import { Shield, Server, Database, Cpu, CheckCircle2, AlertCircle } from 'lucide-react';

export const AdminPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Shield className="w-6 h-6 text-cyan-400" />
          <span>System Administration & Service Health</span>
        </h1>
        <p className="text-xs text-slate-400">
          Database connection pools, PyTorch model registry checkpoints, and background daemon states
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Database Engine</span>
            <Database className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-lg font-bold text-white">PostgreSQL 18 & SQLite</div>
          <span className="inline-flex items-center space-x-1 text-emerald-400 text-xs font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Connection Pool Active</span>
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Neural Acceleration</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-lg font-bold text-white">PyTorch 2.10.0+cpu</div>
          <span className="inline-flex items-center space-x-1 text-cyan-400 text-xs font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Finite-Difference Vectorized</span>
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">FastAPI Gateway</span>
            <Server className="w-4 h-4 text-violet-400" />
          </div>
          <div className="text-lg font-bold text-white">Port 8000 (Uvicorn)</div>
          <span className="inline-flex items-center space-x-1 text-emerald-400 text-xs font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>OpenAPI v3.1 Generated</span>
          </span>
        </div>
      </div>

      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <h2 className="text-base font-bold text-white">Artifacts & Model Registry Status</h2>
        <div className="divide-y divide-slate-800/80 text-xs">
          <div className="py-2.5 flex items-center justify-between">
            <span className="text-slate-300 font-mono">pinn_model.pt</span>
            <span className="text-emerald-400 font-semibold">Verified Weight Checkpoint</span>
          </div>
          <div className="py-2.5 flex items-center justify-between">
            <span className="text-slate-300 font-mono">proposed_dual_transformer.pt</span>
            <span className="text-emerald-400 font-semibold">Verified Weight Checkpoint</span>
          </div>
          <div className="py-2.5 flex items-center justify-between">
            <span className="text-slate-300 font-mono">lstm_model.pt</span>
            <span className="text-emerald-400 font-semibold">Verified Weight Checkpoint</span>
          </div>
          <div className="py-2.5 flex items-center justify-between">
            <span className="text-slate-300 font-mono">scaler_X.pkl / scaler_Y.pkl</span>
            <span className="text-cyan-400 font-semibold">Leakage-Free Standard Scalers</span>
          </div>
        </div>
      </div>
    </div>
  );
};
