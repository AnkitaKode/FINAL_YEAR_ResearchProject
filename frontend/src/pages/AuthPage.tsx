import React, { useState } from 'react';
import { EnergyAPI } from '../services/api';
import { Shield, Lock, Mail, ArrowRight, CheckCircle2 } from 'lucide-react';

export const AuthPage: React.FC = () => {
  const [email, setEmail] = useState('researcher@smartgrid.edu');
  const [password, setPassword] = useState('pinn2026');
  const [loggedIn, setLoggedIn] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (email && password) {
      setLoggedIn(true);
      setError('');
    } else {
      setError('Please provide valid academic credentials.');
    }
  };

  return (
    <div className="max-w-md mx-auto my-12 p-8 rounded-3xl bg-slate-900/60 border border-slate-800 shadow-2xl space-y-6">
      <div className="text-center space-y-2">
        <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 text-cyan-400 mx-auto flex items-center justify-center">
          <Shield className="w-6 h-6" />
        </div>
        <h1 className="text-xl font-bold text-white">Researcher Authentication</h1>
        <p className="text-xs text-slate-400">Access protected training checkpoints and administrative tools</p>
      </div>

      {loggedIn ? (
        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center space-y-2">
          <CheckCircle2 className="w-6 h-6 text-emerald-400 mx-auto" />
          <p className="text-xs font-bold text-emerald-300">Successfully Authenticated as Lead Researcher</p>
          <span className="text-[11px] text-slate-400 font-mono">Role: Academic Researcher (Admin)</span>
        </div>
      ) : (
        <form onSubmit={handleLogin} className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-300">Academic Email</label>
            <div className="flex items-center space-x-2 px-3 py-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <Mail className="w-4 h-4 text-slate-400" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="bg-transparent text-xs text-white w-full focus:outline-none"
              />
            </div>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-300">Password</label>
            <div className="flex items-center space-x-2 px-3 py-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <Lock className="w-4 h-4 text-slate-400" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-transparent text-xs text-white w-full focus:outline-none"
              />
            </div>
          </div>

          {error && <p className="text-xs text-red-400">{error}</p>}

          <button
            type="submit"
            className="w-full py-3 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs transition-all shadow-lg shadow-cyan-600/20"
          >
            Authenticate Session
          </button>
        </form>
      )}
    </div>
  );
};
