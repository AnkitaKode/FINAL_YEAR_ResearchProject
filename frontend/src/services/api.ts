import axios from 'axios';

const API_BASE = '/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const EnergyAPI = {
  getKPIs: async () => {
    const res = await apiClient.get('/analytics/kpis');
    return res.data;
  },
  getTelemetry: async (limit = 48) => {
    const res = await apiClient.get(`/sensors/telemetry?limit=${limit}`);
    return res.data;
  },
  getDatasets: async () => {
    const res = await apiClient.get('/datasets');
    return res.data;
  },
  getForecast: async (modelType = 'Proposed_DualHorizon_PINN', horizon = 24) => {
    const res = await apiClient.post('/forecast/generate', {
      model_type: modelType,
      horizon_hours: horizon,
      start_hour_offset: 0
    });
    return res.data;
  },
  getModels: async () => {
    const res = await apiClient.get('/forecast/models');
    return res.data;
  },
  runOptimization: async (horizon = 24, initialT = 22.0, initialSoc = 50.0) => {
    const res = await apiClient.post('/optimization/run-mpc', {
      horizon_hours: horizon,
      initial_indoor_temp: initialT,
      initial_soc_pct: initialSoc
    });
    return res.data;
  },
  getAnomalies: async (maxRows = 500) => {
    const res = await apiClient.get(`/sensors/anomalies?max_rows=${maxRows}`);
    return res.data;
  },
  getBenchmarks: async () => {
    const res = await apiClient.get('/analytics/benchmarks');
    return res.data;
  },
  getResearchOverview: async () => {
    const res = await apiClient.get('/research/overview');
    return res.data;
  },
  getLiterature: async () => {
    const res = await apiClient.get('/research/papers');
    return res.data;
  }
};
