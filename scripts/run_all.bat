@echo off
echo ==============================================================================
echo Launching Physics-Informed Microgrid Full-Stack Research Platform
echo Backend:  http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/api/v1/docs)
echo Frontend: http://localhost:5173
echo ==============================================================================

start "FastAPI Backend" cmd /k "cd /d %~dp0\.. && .venv\Scripts\activate.bat && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"
start "React Frontend" cmd /k "cd /d %~dp0\..\frontend && npm run dev"

echo Both services launched in separate terminal windows.
pause
