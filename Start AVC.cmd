@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" setup\check_runtime.py
  if errorlevel 1 goto failed
  ".venv\Scripts\python.exe" -m src.app --app-window %*
) else (
  echo Run Setup AVC.cmd first.
  pause
  exit /b 1
)
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo AVC could not start. Run Setup AVC.cmd to repair its Python environment.
pause
exit /b 1
