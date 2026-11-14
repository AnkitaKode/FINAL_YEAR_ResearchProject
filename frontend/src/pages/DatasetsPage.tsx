import React, { useEffect, useState } from 'react';
import { EnergyAPI } from '../services/api';
import { Database, CheckCircle, Upload, Table, AlertTriangle, FileCode } from 'lucide-react';

export const DatasetsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    EnergyAPI.getDatasets()
      .then(setDatasets)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Database className="w-6 h-6 text-cyan-400" />
          <span>Dataset Management & Preprocessing Pipeline</span>
        </h1>
        <p className="text-xs text-slate-400">
          Validation, chronological splitting, feature engineering, and data leakage prevention audits
        </p>
      </div>

      {/* Upload Zone Card */}
      <div className="p-6 rounded-2xl bg-slate-900/40 border-2 border-dashed border-slate-700 hover:border-cyan-500/50 transition-all text-center space-y-3">
        <div className="w-12 h-12 rounded-xl bg-cyan-500/10 text-cyan-400 mx-auto flex items-center justify-center">
          <Upload className="w-6 h-6" />
        </div>
        <div>
          <h2 className="text-sm font-bold text-white">Upload Smart Building Meter or Weather CSV</h2>
          <p className="text-xs text-slate-400 mt-1">Supports ASHRAE Great Energy Predictor III, BDG2, UCI Household, or NREL NSRDB files</p>
        </div>
        <button className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold text-white transition-all">
          Browse CSV Files
        </button>
      </div>

      {/* Registered Datasets List */}
      <div className="space-y-4">
        <h2 className="text-base font-bold text-white flex items-center space-x-2">
          <Table className="w-4 h-4 text-emerald-400" />
          <span>Active Ingested Datasets</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {datasets.map((ds) => (
            <div key={ds.id} className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                    ds.source_type === 'calibrated_synthetic' ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  }`}>
                    {ds.source_type.replace('_', ' ')}
                  </span>
                  <h3 className="text-sm font-bold text-white mt-1.5">{ds.name}</h3>
                </div>
                <div className="flex items-center text-emerald-400 text-xs font-semibold space-x-1">
                  <CheckCircle className="w-4 h-4" />
                  <span>Validated</span>
                </div>
              </div>

              <p className="text-xs text-slate-400 leading-relaxed">{ds.description}</p>

              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800/80 text-[11px]">
                <div>
                  <span className="text-slate-400 block">Total Records:</span>
                  <span className="font-semibold text-slate-200">{ds.total_records.toLocaleString()}</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Resolution:</span>
                  <span className="font-semibold text-slate-200">{ds.sampling_rate}</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Leakage Prevention:</span>
                  <span className="font-semibold text-cyan-300">Chronological 70/15/15</span>
                </div>
              </div>

              {ds.columns && (
                <div className="pt-2">
                  <span className="text-[11px] text-slate-400 block mb-1.5 font-medium">Mapped Features ({ds.columns.length}):</span>
                  <div className="flex flex-wrap gap-1">
                    {ds.columns.slice(0, 8).map((col: string) => (
                      <span key={col} className="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-slate-300 font-mono">
                        {col}
                      </span>
                    ))}
                    {ds.columns.length > 8 && (
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-mono">
                        +{ds.columns.length - 8} more
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
