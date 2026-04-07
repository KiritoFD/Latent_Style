@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/16] Running 01_idt_08...
uv run .\run.py --config .\config_01_idt_08.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/16] Running 02_idt_12...
uv run .\run.py --config .\config_02_idt_12.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/16] Running 03_idt_16...
uv run .\run.py --config .\config_03_idt_16.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/16] Running 04_color_down_25...
uv run .\run.py --config .\config_04_color_down_25.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/16] Running 05_swd_up_60...
uv run .\run.py --config .\config_05_swd_up_60.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/16] Running 06_swd_dominate...
uv run .\run.py --config .\config_06_swd_dominate.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/16] Running 07_repulse_off...
uv run .\run.py --config .\config_07_repulse_off.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/16] Running 08_repulse_strong...
uv run .\run.py --config .\config_08_repulse_strong.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/16] Running 09_repulse_margin_2...
uv run .\run.py --config .\config_09_repulse_margin_2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [10/16] Running 10_patch_micro_only...
uv run .\run.py --config .\config_10_patch_micro_only.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [11/16] Running 11_patch_macro_only...
uv run .\run.py --config .\config_11_patch_macro_only.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [12/16] Running 12_patch_bipolar...
uv run .\run.py --config .\config_12_patch_bipolar.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [13/16] Running 13_patch_bipolar_extreme...
uv run .\run.py --config .\config_13_patch_bipolar_extreme.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [14/16] Running 14_sweetspot_texture...
uv run .\run.py --config .\config_14_sweetspot_texture.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [15/16] Running 15_sweetspot_color...
uv run .\run.py --config .\config_15_sweetspot_color.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [16/16] Running 16_sweetspot_balanced...
uv run .\run.py --config .\config_16_sweetspot_balanced.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "TARGET_DIR=%SRC_DIR%\..\base"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%SRC_DIR%\..\base_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Running batch distill + full eval...
uv run .\batch_distill_full_eval.py --exp_dir ..\base\
if %errorlevel% neq 0 exit /b %errorlevel%

echo Exporting summary CSV...
python .\import_summary_history_to_csv.py -i ..\base -o .\base.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe sweep...
set "FINAL_EPOCH=0100"
for /d %%D in ("%TARGET_DIR%\base_*") do (
  if exist "%%~fD\epoch_%FINAL_EPOCH%.pt" (
    uv run .\probe_ma.py --checkpoint "%%~fD\epoch_%FINAL_EPOCH%.pt" --num-samples 8 --json-out "%%~fD\ma_probe_base_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
  )
)

echo All done.
