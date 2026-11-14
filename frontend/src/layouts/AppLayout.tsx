import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Activity,
  Cpu,
  TrendingUp,
  Sliders,
  Radio,
  Zap,
  BarChart3,
  BookOpen,
  FileText,
  Shield,
  Database,
  Layers,
  Menu,
  X
} from 'lucide-react';

interface AppLayoutProps {
  children: React.ReactNode;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { label: 'Overview', path: '/', icon: Layers },
    { label: 'Live Dashboard', path: '/dashboard', icon: Activity },
    { label: 'Datasets', path: '/datasets', icon: Database },
    { label: 'PINN Thermal', path: '/pinn', icon: Cpu },
    { label: 'Forecasting', path: '/forecasting', icon: TrendingUp },
    { label: 'MPC Optimizer', path: '/optimization', icon: Sliders },
    { label: 'IoT Sensors', path: '/sensors', icon: Radio },
    { label: 'Microgrid', path: '/microgrid', icon: Zap },
    { label: 'Benchmark', path: '/performance', icon: BarChart3 },
    { label: 'Research', path: '/research', icon: BookOpen },
    { label: 'Documentation', path: '/documentation', icon: FileText },
    { label: 'Admin', path: '/admin', icon: Shield },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 px-4 lg:px-8 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Zap className="w-6 h-6 text-slate-950 font-bold" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 bg-clip-text text-transparent">
              PINN Energy & Microgrid
            </h1>
            <p className="text-xs text-slate-400 hidden sm:block">
              Physics-Informed Deep Learning • Final-Year B.Tech Research Platform
            </p>
          </div>
        </div>

        {/* Desktop Quick Status */}
        <div className="hidden md:flex items-center space-x-4">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>PyTorch 1R-1C Engine: Operational</span>
          </div>
          <Link
            to="/auth"
            className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold text-white transition-all shadow-md shadow-cyan-600/20"
          >
            Researcher Login
          </Link>
        </div>

        {/* Mobile menu button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 rounded-lg bg-slate-800 text-slate-300"
        >
          {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </header>

      <div className="flex flex-1">
        {/* Desktop Sidebar Navigation */}
        <aside className="hidden md:flex flex-col w-64 bg-slate-900/50 border-r border-slate-800 p-4 space-y-1">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 px-3 py-2">
            Navigation Menu
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/20 to-emerald-500/10 text-cyan-300 border-l-4 border-cyan-400'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </Link>
            );
          })}

          <div className="mt-auto pt-4 border-t border-slate-800/80">
            <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-700/50 text-xs text-slate-400 space-y-1.5">
              <div className="font-semibold text-slate-200 flex items-center justify-between">
                <span>Model Target</span>
                <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 text-[10px]">Dual-Horizon</span>
              </div>
              <p className="text-[11px] text-slate-400">
                1R-1C Thermal PDE with ASHRAE-55 Comfort MPC
              </p>
            </div>
          </div>
        </aside>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div className="md:hidden fixed inset-0 z-40 bg-slate-950/90 backdrop-blur-md pt-20 px-6 space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`flex items-center space-x-3 px-4 py-3 rounded-xl text-base font-medium ${
                    isActive ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400'
                  }`}
                >
                  <Icon className="w-5 h-5 text-cyan-400" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        )}

        {/* Main Content Area */}
        <main className="flex-1 p-4 lg:p-8 overflow-y-auto max-w-7xl mx-auto w-full">
          {children}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-900/60 px-6 py-4 text-center text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div>
          Physics-Informed Deep Learning Architecture • Final-Year Research Project
        </div>
        <div className="text-slate-400">
          Backend API: <span className="text-cyan-400 font-mono">http://127.0.0.1:8000</span>
        </div>
      </footer>
    </div>
  );
};
