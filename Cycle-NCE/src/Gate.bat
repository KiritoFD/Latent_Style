@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/12] Running 01_baseline...
uv run .\run.py --config .\config_01_baseline.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/12] Running 02_weak_decoder...
uv run .\run.py --config .\config_02_weak_decoder.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/12] Running 03_restore_skip_shortcut...
uv run .\run.py --config .\config_03_restore_skip_shortcut.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/12] Running 04_attn_gate_fixed...
uv run .\run.py --config .\config_04_attn_gate_fixed.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/12] Running 05_attn_gate_learned...
uv run .\run.py --config .\config_05_attn_gate_learned.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/12] Running 06_gate_learned_idt_energy...
uv run .\run.py --config .\config_06_gate_learned_idt_energy.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/12] Running 07_aux_loss_weak...
uv run .\run.py --config .\config_07_aux_loss_weak.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/12] Running 08_aux_loss_strong...
uv run .\run.py --config .\config_08_aux_loss_strong.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/12] Running 09_gate_and_bipolar...
uv run .\run.py --config .\config_09_gate_and_bipolar.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [10/12] Running 10_gate_and_low_color...
uv run .\run.py --config .\config_10_gate_and_low_color.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [11/12] Running 11_gate_bipolar_low_color...
uv run .\run.py --config .\config_11_gate_bipolar_low_color.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [12/12] Running 12_gate_energy_bipolar_low_color...
uv run .\run.py --config .\config_12_gate_energy_bipolar_low_color.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "TARGET_DIR=%SRC_DIR%\..\Gate"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%SRC_DIR%\..\Gate_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Running batch distill + full eval...
uv run .\batch_distill_full_eval.py --exp_dir ..\Gate\
if %errorlevel% neq 0 exit /b %errorlevel%

echo Exporting summary CSV...
python .\import_summary_history_to_csv.py -i ..\Gate -o .\Gate.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe sweep...
set "FINAL_EPOCH=0060"
for /d %%D in ("%TARGET_DIR%\Gate_*") do (
  if exist "%%~fD\epoch_%FINAL_EPOCH%.pt" (
    uv run .\probe_ma.py --checkpoint "%%~fD\epoch_%FINAL_EPOCH%.pt" --num-samples 8 --json-out "%%~fD\ma_probe_base_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
  )
)

echo All done.
