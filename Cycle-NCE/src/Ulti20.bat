@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/20] Running L16_01_Sw100_Hf1_PMic_Co10_Id5...
uv run run.py --config config_L16_01_Sw100_Hf1_PMic_Co10_Id5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/20] Running L16_02_Sw100_Hf1_PMic_Co80_Id30...
uv run run.py --config config_L16_02_Sw100_Hf1_PMic_Co80_Id30.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/20] Running L16_03_Sw100_Hf1_PMac_Co10_Id30...
uv run run.py --config config_L16_03_Sw100_Hf1_PMac_Co10_Id30.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/20] Running L16_04_Sw100_Hf1_PMac_Co80_Id5...
uv run run.py --config config_L16_04_Sw100_Hf1_PMac_Co80_Id5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/20] Running L16_05_Sw100_Hf4_PMic_Co10_Id30...
uv run run.py --config config_L16_05_Sw100_Hf4_PMic_Co10_Id30.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/20] Running L16_06_Sw100_Hf4_PMic_Co80_Id5...
uv run run.py --config config_L16_06_Sw100_Hf4_PMic_Co80_Id5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/20] Running L16_07_Sw100_Hf4_PMac_Co10_Id5...
uv run run.py --config config_L16_07_Sw100_Hf4_PMac_Co10_Id5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/20] Running L16_08_Sw100_Hf4_PMac_Co80_Id30...
uv run run.py --config config_L16_08_Sw100_Hf4_PMac_Co80_Id30.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/20] Running L16_09_Sw250_Hf1_PMic_Co10_Id5...
uv run run.py --config config_L16_09_Sw250_Hf1_PMic_Co10_Id5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [10/20] Running L16_10_Sw250_Hf1_PMic_Co80_Id30...
uv run run.py --config config_L16_10_Sw250_Hf1_PMic_Co80_Id30.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [11/20] Running L16_11_Sw250_Hf1_PMac_Co10_Id30...
uv run run.py --config config_L16_11_Sw250_Hf1_PMac_Co10_Id30.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [12/20] Running L16_12_Sw250_Hf1_PMac_Co80_Id5...
uv run run.py --config config_L16_12_Sw250_Hf1_PMac_Co80_Id5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [13/20] Running L16_13_Sw250_Hf4_PMic_Co10_Id30...
uv run run.py --config config_L16_13_Sw250_Hf4_PMic_Co10_Id30.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [14/20] Running L16_14_Sw250_Hf4_PMic_Co80_Id5...
uv run run.py --config config_L16_14_Sw250_Hf4_PMic_Co80_Id5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [15/20] Running L16_15_Sw250_Hf4_PMac_Co10_Id5...
uv run run.py --config config_L16_15_Sw250_Hf4_PMac_Co10_Id5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [16/20] Running L16_16_Sw250_Hf4_PMac_Co80_Id30...
uv run run.py --config config_L16_16_Sw250_Hf4_PMac_Co80_Id30.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [17/20] Running Probe_17_LR_High...
uv run run.py --config config_Probe_17_LR_High.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [18/20] Running Probe_18_LR_Low...
uv run run.py --config config_Probe_18_LR_Low.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [19/20] Running Probe_19_LR_OneCycle_Aggressive...
uv run run.py --config config_Probe_19_LR_OneCycle_Aggressive.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [20/20] Running Probe_20_Baseline_Anchor...
uv run run.py --config config_Probe_20_Baseline_Anchor.json
if %errorlevel% neq 0 exit /b %errorlevel%

