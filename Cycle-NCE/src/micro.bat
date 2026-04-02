@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/4] Running E01_Patch3_Gain4_LR2e4...
uv run run.py --config config_E01_Patch3_Gain4_LR2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/4] Running E02_Patch3_5_Gain4_LR2e4...
uv run run.py --config config_E02_Patch3_5_Gain4_LR2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/4] Running E03_Patch3_5_7_Gain4_LR2e4...
uv run run.py --config config_E03_Patch3_5_7_Gain4_LR2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/4] Running E04_Patch1_3_5_Gain4_LR2e4...
uv run run.py --config config_E04_Patch1_3_5_Gain4_LR2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "ROOT_DIR=%SRC_DIR%\.."
set "TARGET_DIR=%ROOT_DIR%\micro"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%ROOT_DIR%\micro_*") do (
  echo Copying %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
)

echo Copy finished. Running batch distill eval...
cd /d "%ROOT_DIR%"
uv run src/batch_distill_full_eval.py --exp_dir micro
if %errorlevel% neq 0 exit /b %errorlevel%

echo All done.
