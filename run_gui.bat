@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app_gui.py
) else (
  python app_gui.py
)
pause
