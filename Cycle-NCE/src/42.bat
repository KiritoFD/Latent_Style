@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/10] Running A01_Macro_Only_LR3e4...
uv run run.py --config config_A01_Macro_Only_LR3e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/10] Running A02_Micro_Only_LR3e4...
uv run run.py --config config_A02_Micro_Only_LR3e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/10] Running A03_Bipolar_Extreme...
uv run run.py --config config_A03_Bipolar_Extreme.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/10] Running A04_FullSpec_Conv2_LR4e4...
uv run run.py --config config_A04_FullSpec_Conv2_LR4e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/10] Running A05_FullSpec_Conv3_LR2e4...
uv run run.py --config config_A05_FullSpec_Conv3_LR2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/10] Running A06_FullSpec_Conv3_LR3e4...
uv run run.py --config config_A06_FullSpec_Conv3_LR3e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/10] Running A07_NoSkip_Conv2_LR3e4...
uv run run.py --config config_A07_NoSkip_Conv2_LR3e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/10] Running A08_Noise01_Conv2_LR3e4...
uv run run.py --config config_A08_Noise01_Conv2_LR3e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/10] Running A09_Gain4_Conv2_LR2e4...
uv run run.py --config config_A09_Gain4_Conv2_LR2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [10/10] Running A10_Color200_Conv2_LR3e4...
uv run run.py --config config_A10_Color200_Conv2_LR3e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "ROOT_DIR=%SRC_DIR%\.."
set "TARGET_DIR=%ROOT_DIR%\42"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%ROOT_DIR%\42_*") do (
  echo Copying %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
)

echo Copy finished. Running batch distill eval...
cd /d "%ROOT_DIR%"
uv run src/batch_distill_full_eval.py --exp_dir 42
if %errorlevel% neq 0 exit /b %errorlevel%

echo All done.
