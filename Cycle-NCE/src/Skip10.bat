@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/10] Running S01_None...
uv run run.py --config config_S01_None.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/10] Running S02_Naive_G1p0...
uv run run.py --config config_S02_Naive_G1p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/10] Running S03_Naive_G0p5...
uv run run.py --config config_S03_Naive_G0p5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/10] Running S04_Naive_G1p5...
uv run run.py --config config_S04_Naive_G1p5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/10] Running S05_Adaptive...
uv run run.py --config config_S05_Adaptive.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/10] Running S06_Normalized...
uv run run.py --config config_S06_Normalized.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/10] Running S07_Stress_Naive...
uv run run.py --config config_S07_Stress_Naive.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/10] Running S08_Stress_Norm...
uv run run.py --config config_S08_Stress_Norm.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/10] Running S09_Fusion_Add...
uv run run.py --config config_S09_Fusion_Add.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [10/10] Running S10_Fusion_Cat...
uv run run.py --config config_S10_Fusion_Cat.json
if %errorlevel% neq 0 exit /b %errorlevel%

