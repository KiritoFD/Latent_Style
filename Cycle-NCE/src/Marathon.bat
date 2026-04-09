@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/8] Running 01_marathon_baseline...
uv run .\run.py --config .\config_01_marathon_baseline.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/8] Running 02_heavy_style...
uv run .\run.py --config .\config_02_heavy_style.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/8] Running 03_relax_idt...
uv run .\run.py --config .\config_03_relax_idt.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/8] Running 04_mode_seeking_repulse...
uv run .\run.py --config .\config_04_mode_seeking_repulse.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/8] Running 05_extreme_bipolar...
uv run .\run.py --config .\config_05_extreme_bipolar.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/8] Running 06_color_starvation...
uv run .\run.py --config .\config_06_color_starvation.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/8] Running 07_with_smooth_ablation...
uv run .\run.py --config .\config_07_with_smooth_ablation.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/8] Running 08_the_hail_mary...
uv run .\run.py --config .\config_08_the_hail_mary.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "TARGET_DIR=%SRC_DIR%\..\Marathon"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%SRC_DIR%\..\Marathon_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Running batch distill + full eval...
uv run .\batch_distill_full_eval.py --exp_dir ..\Marathon\
if %errorlevel% neq 0 exit /b %errorlevel%

echo Exporting summary CSV...
python .\import_summary_history_to_csv.py -i ..\Marathon -o .\Marathon.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe sweep...
set "FINAL_EPOCH=0180"
for /d %%D in ("%TARGET_DIR%\Marathon_*") do (
  if exist "%%~fD\epoch_%FINAL_EPOCH%.pt" (
    uv run .\probe_ma.py --checkpoint "%%~fD\epoch_%FINAL_EPOCH%.pt" --num-samples 8 --json-out "%%~fD\ma_probe_base_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
  )
)

echo All done.
