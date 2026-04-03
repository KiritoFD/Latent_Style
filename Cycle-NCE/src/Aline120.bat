@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/5] Running aline_01_oracle...
uv run run.py --config config_aline_01_oracle.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/5] Running aline_02_texture_maniac...
uv run run.py --config config_aline_02_texture_maniac.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/5] Running aline_03_ghost_wireframe...
uv run run.py --config config_aline_03_ghost_wireframe.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/5] Running aline_04_macro_trap...
uv run run.py --config config_aline_04_macro_trap.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/5] Running aline_05_idt_poison...
uv run run.py --config config_aline_05_idt_poison.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "ROOT_DIR=%SRC_DIR%\.."
set "TARGET_DIR=%ROOT_DIR%\Aline120"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%ROOT_DIR%\Aline120_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Move finished. Running batch distill eval...
cd /d "%ROOT_DIR%"
uv run src/batch_distill_full_eval.py --exp_dir Aline120
if %errorlevel% neq 0 exit /b %errorlevel%

echo Distill/eval finished. Exporting CSV summary...
python import_summary_history_to_csv.py -i Aline120 -o Aline120.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo All done.
