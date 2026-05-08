@echo off
setlocal
cd /d %~dp0
for %%N in (kin40_10 kin40_02 kin40_00) do (
  if exist "experiments\semantic_full40_kinetic_sweep\%%N\epoch_0040.pt" (
    echo ==================================================
    echo Evaluating %%N
    python run_evaluation.py "experiments\semantic_full40_kinetic_sweep\%%N" --output "experiments\semantic_full40_kinetic_sweep\full_eval\%%N" --batch_size 2 --force
    if errorlevel 1 exit /b %errorlevel%
  )
)
endlocal
