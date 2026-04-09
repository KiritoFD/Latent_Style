@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/6] Running 01_baseline_burn...
uv run .\run.py --config .\config_01_baseline_burn.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/6] Running 02_baseline_shielded...
uv run .\run.py --config .\config_02_baseline_shielded.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/6] Running 03_shielded_relax_idt...
uv run .\run.py --config .\config_03_shielded_relax_idt.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/6] Running 04_shielded_high_swd...
uv run .\run.py --config .\config_04_shielded_high_swd.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/6] Running 05_shielded_bipolar_patch...
uv run .\run.py --config .\config_05_shielded_bipolar_patch.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/6] Running 06_shielded_attn_unleashed...
uv run .\run.py --config .\config_06_shielded_attn_unleashed.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "TARGET_DIR=%SRC_DIR%\..\NumFix"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%SRC_DIR%\..\NumFix_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Running batch distill + full eval...
uv run .\batch_distill_full_eval.py --exp_dir ..\NumFix\
if %errorlevel% neq 0 exit /b %errorlevel%

echo Exporting summary CSV...
python .\import_summary_history_to_csv.py -i ..\NumFix -o .\NumFix.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe sweep...
set "FINAL_EPOCH=0060"
for /d %%D in ("%TARGET_DIR%\NumFix_*") do (
  if exist "%%~fD\epoch_%FINAL_EPOCH%.pt" (
    uv run .\probe_ma.py --checkpoint "%%~fD\epoch_%FINAL_EPOCH%.pt" --num-samples 8 --json-out "%%~fD\ma_probe_base_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
  )
)

echo All done.
