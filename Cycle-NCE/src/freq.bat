@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/4] Running 01_golden_funnel...
uv run run.py --config config_01_golden_funnel.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/4] Running 02_naked_fusion...
uv run run.py --config config_02_naked_fusion.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/4] Running 03_macro_dictator...
uv run run.py --config config_03_macro_dictator.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/4] Running 04_micro_rebel...
uv run run.py --config config_04_micro_rebel.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "ROOT_DIR=%SRC_DIR%\.."
set "TARGET_DIR=%ROOT_DIR%\freq"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%ROOT_DIR%\freq_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Move finished. Running batch distill eval...
cd /d "%ROOT_DIR%"
uv run src/batch_distill_full_eval.py --exp_dir freq
if %errorlevel% neq 0 exit /b %errorlevel%

echo Distill/eval finished. Exporting CSV summary...
python import_summary_history_to_csv.py -i freq -o freq.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe per experiment...
set "FINAL_EPOCH=0030"
for /d %%D in ("%TARGET_DIR%\freq_*") do (
  if exist "%%~fD\epoch_%FINAL_EPOCH%.pt" (
    echo Probing %%~nxD with %%~fD\epoch_%FINAL_EPOCH%.pt...
    uv run src/probe_ma.py --checkpoint "%%~fD\epoch_%FINAL_EPOCH%.pt" --num-samples 8 --json-out "%%~fD\ma_probe_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
    uv run src/probe_ma_sweep.py --input-glob "%%~fD\ma_probe*.json" --output-dir "%%~fD" --output-prefix ma_probe_view
    if %errorlevel% neq 0 exit /b %errorlevel%
  ) else (
    echo WARNING: checkpoint not found for %%~nxD at %%~fD\epoch_%FINAL_EPOCH%.pt
  )
)

echo Building cross-experiment MA summary...
uv run src/probe_ma_sweep.py --input-glob "%TARGET_DIR%\freq_*\ma_probe*.json" --output-dir "%TARGET_DIR%" --output-prefix ma_probe_all_pairs
if %errorlevel% neq 0 exit /b %errorlevel%

for /d %%D in ("%TARGET_DIR%\freq_*") do (
  if exist "%TARGET_DIR%\ma_probe_all_pairs.html" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.html" "%%~fD\ma_probe_all_pairs.html" >nul
  if exist "%TARGET_DIR%\ma_probe_all_pairs.csv" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.csv" "%%~fD\ma_probe_all_pairs.csv" >nul
  if exist "%TARGET_DIR%\ma_probe_all_pairs.json" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.json" "%%~fD\ma_probe_all_pairs.json" >nul
)

echo All done.
