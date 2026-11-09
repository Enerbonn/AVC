@echo off
setlocal
cd /d "%~dp0"
python setup\install.py
if errorlevel 1 goto failed
echo.
echo Setup complete. Open Start AVC.cmd.
pause
exit /b 0
:failed
echo Setup failed. Python 3.12 is recommended. Check the error above and your internet connection.
pause
exit /b 1
