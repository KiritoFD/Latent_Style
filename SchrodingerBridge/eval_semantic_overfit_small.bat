@echo off
setlocal
setlocal EnableDelayedExpansion

cd /d "%~dp0"

for %%N in (A_baseline B_no_kinetic C_no_low_freq) do (
  if exist "experiments\semantic_overfit_small\%%N\epoch_0003.pt" (
    echo [%%N] eval
    python run_evaluation.py "experiments\semantic_overfit_small\%%N" --output "experiments\semantic_overfit_small\full_eval\%%N" --batch_size 1
    echo.
  ) else (
    echo [%%N] missing epoch_0003.pt, skip eval
    echo.
  )
)

exit /b 0
