@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/2] Running F01_Patch135_Gain1.5_LR2e4...
uv run run.py --config config_F01_Patch135_Gain1.5_LR2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/2] Running F02_Patch357_Gain1.5_LR2e4...
uv run run.py --config config_F02_Patch357_Gain1.5_LR2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "ROOT_DIR=%SRC_DIR%\.."
set "TARGET_DIR=%ROOT_DIR%\FinalMicro_2"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%ROOT_DIR%\FinalMicro_2_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Move finished. Running batch distill eval...
cd /d "%ROOT_DIR%"
uv run src/batch_distill_full_eval.py --exp_dir FinalMicro_2
if %errorlevel% neq 0 exit /b %errorlevel%

echo Distill/eval finished. Exporting CSV summary...
python import_summary_history_to_csv.py -i FinalMicro_2 -o FinalMicro_2.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo All done.
