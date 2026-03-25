@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
echo ==========================================
echo Running SWD Patch Size Ablation (4 exps)
echo ==========================================
echo.
echo ------------------------------------------
echo Running Experiment 1: patch_size_ablation_1_ps3-5-7-11
echo ------------------------------------------
uv run run.py --config config_patch_size_ablation_1_ps3-5-7-11.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 2: patch_size_ablation_2_ps5-9-15
echo ------------------------------------------
uv run run.py --config config_patch_size_ablation_2_ps5-9-15.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 3: patch_size_ablation_3_ps7-11-19
echo ------------------------------------------
uv run run.py --config config_patch_size_ablation_3_ps7-11-19.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 4: patch_size_ablation_4_ps11-15-23
echo ------------------------------------------
uv run run.py --config config_patch_size_ablation_4_ps11-15-23.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo SWD patch size ablation finished.
