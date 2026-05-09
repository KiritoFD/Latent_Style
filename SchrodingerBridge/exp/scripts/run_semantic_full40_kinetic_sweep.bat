@echo off
setlocal
cd /d %~dp0
for %%N in (kin40_10 kin40_02 kin40_00) do (
  echo ==================================================
  echo Training %%N
  python run.py --config "experiments\semantic_full40_kinetic_sweep\%%N.json"
  if errorlevel 1 exit /b %errorlevel%
)
endlocal
