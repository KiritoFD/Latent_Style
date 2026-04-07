@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/10] Running 01_baseline...
uv run .\run.py --config .\config_01_baseline.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/10] Running 02_no_pos_emb...
uv run .\run.py --config .\config_02_no_pos_emb.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/10] Running 03_high_temp_attn...
uv run .\run.py --config .\config_03_high_temp_attn.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/10] Running 04_no_color_highway...
uv run .\run.py --config .\config_04_no_color_highway.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/10] Running 05_no_skip...
uv run .\run.py --config .\config_05_no_skip.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/10] Running 06_patch_micro_only...
uv run .\run.py --config .\config_06_patch_micro_only.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/10] Running 07_patch_macro_only...
uv run .\run.py --config .\config_07_patch_macro_only.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/10] Running 08_swd_sort_mode...
uv run .\run.py --config .\config_08_swd_sort_mode.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/10] Running 09_no_pos_high_temp...
uv run .\run.py --config .\config_09_no_pos_high_temp.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [10/10] Running 10_macro_no_skip...
uv run .\run.py --config .\config_10_macro_no_skip.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "TARGET_DIR=%SRC_DIR%\..\chess"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%SRC_DIR%\..\chess_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Running batch distill + full eval...
uv run .\batch_distill_full_eval.py --exp_dir ..\chess\
if %errorlevel% neq 0 exit /b %errorlevel%

echo Exporting summary CSV...
python .\import_summary_history_to_csv.py -i ..\chess -o .\chess.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe sweep...
set "FINAL_EPOCH=0060"
for /d %%D in ("%TARGET_DIR%\chess_*") do (
  if exist "%%~fD\epoch_%FINAL_EPOCH%.pt" (
    uv run .\probe_ma.py --checkpoint "%%~fD\epoch_%FINAL_EPOCH%.pt" --num-samples 8 --json-out "%%~fD\ma_probe_base_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
  )
)

echo All done.
