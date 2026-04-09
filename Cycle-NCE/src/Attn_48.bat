@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/17] Running 01_macro_base...
uv run .\run.py --config .\config_01_macro_base.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/17] Running 02_macro_idt_10...
uv run .\run.py --config .\config_02_macro_idt_10.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/17] Running 03_macro_idt_15...
uv run .\run.py --config .\config_03_macro_idt_15.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/17] Running 04_color_down_20...
uv run .\run.py --config .\config_04_color_down_20.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/17] Running 05_color_down_05...
uv run .\run.py --config .\config_05_color_down_05.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/17] Running 06_swd_up_80...
uv run .\run.py --config .\config_06_swd_up_80.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/17] Running 07_attn_direct_qk...
uv run .\run.py --config .\config_07_attn_direct_qk.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/17] Running 08_attn_raw_v...
uv run .\run.py --config .\config_08_attn_raw_v.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/17] Running 09_attn_no_smooth...
uv run .\run.py --config .\config_09_attn_no_smooth.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [10/17] Running 10_attn_fully_unleashed...
uv run .\run.py --config .\config_10_attn_fully_unleashed.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [11/17] Running 11_patch_bipolar...
uv run .\run.py --config .\config_11_patch_bipolar.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [12/17] Running 12_patch_bipolar_unleashed...
uv run .\run.py --config .\config_12_patch_bipolar_unleashed.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [13/17] Running 13_sota_push_macro...
uv run .\run.py --config .\config_13_sota_push_macro.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [14/17] Running 14_sota_push_bipolar...
uv run .\run.py --config .\config_14_sota_push_bipolar.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [15/17] Running 15_sota_push_repulse...
uv run .\run.py --config .\config_15_sota_push_repulse.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [16/17] Running 16_the_hail_mary...
uv run .\run.py --config .\config_16_the_hail_mary.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [17/17] Running 17_db_tsw_test...
uv run .\run.py --config .\config_17_db_tsw_test.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "TARGET_DIR=%SRC_DIR%\..\Attn_48"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%SRC_DIR%\..\Attn_48_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Running batch distill + full eval...
uv run .\batch_distill_full_eval.py --exp_dir ..\Attn_48\
if %errorlevel% neq 0 exit /b %errorlevel%

echo Exporting summary CSV...
python .\import_summary_history_to_csv.py -i ..\Attn_48 -o .\Attn_48.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe sweep...
set "FINAL_EPOCH=0060"
for /d %%D in ("%TARGET_DIR%\Attn_48_*") do (
  if exist "%%~fD\epoch_%FINAL_EPOCH%.pt" (
    uv run .\probe_ma.py --checkpoint "%%~fD\epoch_%FINAL_EPOCH%.pt" --num-samples 8 --json-out "%%~fD\ma_probe_base_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
  )
)

echo All done.
