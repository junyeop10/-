@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" app_gui.py
) else if exist "C:\Windows\pyw.exe" (
  start "" "C:\Windows\pyw.exe" app_gui.py
) else (
  start "" pythonw app_gui.py
)
