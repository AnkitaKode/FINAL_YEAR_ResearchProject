import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from './layouts/AppLayout';
import { LandingPage } from './pages/LandingPage';
import { DashboardPage } from './pages/DashboardPage';
import { DatasetsPage } from './pages/DatasetsPage';
import { PINNPage } from './pages/PINNPage';
import { ForecastingPage } from './pages/ForecastingPage';
import { OptimizationPage } from './pages/OptimizationPage';
import { SensorsPage } from './pages/SensorsPage';
import { MicrogridPage } from './pages/MicrogridPage';
import { PerformancePage } from './pages/PerformancePage';
import { ResearchPage } from './pages/ResearchPage';
import { DocumentationPage } from './pages/DocumentationPage';
import { AuthPage } from './pages/AuthPage';
import { AdminPage } from './pages/AdminPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/pinn" element={<PINNPage />} />
          <Route path="/forecasting" element={<ForecastingPage />} />
          <Route path="/optimization" element={<OptimizationPage />} />
          <Route path="/sensors" element={<SensorsPage />} />
          <Route path="/microgrid" element={<MicrogridPage />} />
          <Route path="/performance" element={<PerformancePage />} />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/documentation" element={<DocumentationPage />} />
          <Route path="/auth" element={<AuthPage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  );
};

export default App;
