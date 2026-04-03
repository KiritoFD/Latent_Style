@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/2] Running C01_HF_Tyrant...
uv run run.py --config config_C01_HF_Tyrant.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/2] Running C02_HF_Leakage...
uv run run.py --config config_C02_HF_Leakage.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "ROOT_DIR=%SRC_DIR%\.."
set "TARGET_DIR=%ROOT_DIR%\Color120"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%ROOT_DIR%\Color120_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Move finished. Running batch distill eval...
cd /d "%ROOT_DIR%"
uv run src/batch_distill_full_eval.py --exp_dir Color120
if %errorlevel% neq 0 exit /b %errorlevel%

echo Distill/eval finished. Exporting CSV summary...
python import_summary_history_to_csv.py -i Color120 -o Color120.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo All done.
