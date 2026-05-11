@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHON_EXE=C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
set "PYTHONHOME="
"%PYTHON_EXE%" build_csv.py "./exp/pareto_probe_4"
