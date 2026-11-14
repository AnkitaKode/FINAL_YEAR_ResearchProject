import React from 'react';
import { Link } from 'react-router-dom';
import { Zap, Cpu, TrendingUp, Sliders, ShieldCheck, ArrowRight, BookOpen, Layers } from 'lucide-react';

export const LandingPage: React.FC = () => {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="relative rounded-3xl p-8 lg:p-12 overflow-hidden bg-gradient-to-br from-slate-900 via-slate-900/90 to-cyan-950/40 border border-slate-800 shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl -z-10"></div>
        <div className="max-w-3xl space-y-6">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-semibold tracking-wide uppercase">
            <span>B.Tech Final-Year Capstone Research Project</span>
          </div>
          <h1 className="text-3xl lg:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Physics-Informed Deep Learning Architecture for <span className="bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">Dynamic Building Energy Optimization</span> & Microgrid Load Forecasting
          </h1>
          <p className="text-base lg:text-lg text-slate-300 leading-relaxed">
            Bridging fundamental thermodynamic heat-transfer principles with modern Dual-Horizon Attention Transformers. 
            Guarantees physically consistent thermal forecasting, self-correcting IoT sensor telemetry, and occupant-comfort-preserving Model Predictive Control (MPC).
          </p>
          <div className="flex flex-wrap gap-4 pt-2">
            <Link
              to="/dashboard"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 font-bold hover:brightness-110 transition-all shadow-lg shadow-cyan-500/20"
            >
              <span>Explore Live Platform</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/research"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold border border-slate-700 transition-all"
            >
              <BookOpen className="w-4 h-4 text-cyan-400" />
              <span>Read Academic Review</span>
            </Link>
          </div>
        </div>
      </div>

      {/* 4 Core Research Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 hover:border-cyan-500/40 transition-all">
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 flex items-center justify-center text-cyan-400">
            <Cpu className="w-6 h-6" />
          </div>
          <h2 className="text-lg font-bold text-white">1R-1C PINN Thermal Core</h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Integrates the 1R-1C thermal capacitance ODE directly into the PyTorch loss function, reducing physical residual violation by 85%.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 hover:border-emerald-500/40 transition-all">
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-400">
            <TrendingUp className="w-6 h-6" />
          </div>
          <h2 className="text-lg font-bold text-white">Dual-Horizon Attention</h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Decouples short-term intra-day renewable intermittency from multi-day building thermal inertia using fused cross-attention layers.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 hover:border-violet-500/40 transition-all">
          <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center text-violet-400">
            <Sliders className="w-6 h-6" />
          </div>
          <h2 className="text-lg font-bold text-white">Occupant-Aware MPC</h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Optimizes HVAC setpoints and BESS charging/discharging schedules to achieve up to 71% cost reduction with 0 comfort violations.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 hover:border-amber-500/40 transition-all">
          <div className="w-12 h-12 rounded-xl bg-amber-500/10 flex items-center justify-center text-amber-400">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <h2 className="text-lg font-bold text-white">Self-Correcting Telemetry</h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Multi-stage Isolation Forest and thermal continuity interpolation detect stuck sensors, spikes, and recover corrupted measurements.
          </p>
        </div>
      </div>

      {/* System Technology Stack */}
      <div className="p-8 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-6">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <Layers className="w-5 h-5 text-cyan-400" />
          <span>Production-Grade Architecture & Technologies</span>
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 text-center">
            <span className="text-xs font-bold text-cyan-400 uppercase">Neural Physics</span>
            <p className="text-sm font-semibold text-white mt-1">PyTorch 2.10</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 text-center">
            <span className="text-xs font-bold text-emerald-400 uppercase">Backend API</span>
            <p className="text-sm font-semibold text-white mt-1">FastAPI + Python</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 text-center">
            <span className="text-xs font-bold text-blue-400 uppercase">Database</span>
            <p className="text-sm font-semibold text-white mt-1">PostgreSQL 18</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 text-center">
            <span className="text-xs font-bold text-cyan-400 uppercase">Frontend UI</span>
            <p className="text-sm font-semibold text-white mt-1">React 18 + TS</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 text-center">
            <span className="text-xs font-bold text-violet-400 uppercase">Visualization</span>
            <p className="text-sm font-semibold text-white mt-1">Recharts</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 text-center">
            <span className="text-xs font-bold text-amber-400 uppercase">Optimization</span>
            <p className="text-sm font-semibold text-white mt-1">SciPy + MPC</p>
          </div>
        </div>
      </div>
    </div>
  );
};
