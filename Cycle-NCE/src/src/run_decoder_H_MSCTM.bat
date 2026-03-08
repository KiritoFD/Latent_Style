@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\decoder-H-MSCTM-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting decoder-H-MSCTM (120 Epochs)
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: decoder-H-MSCTM
echo ------------------------------------------
uv run run.py --config config_decoder-H-MSCTM.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\decoder-H-MSCTM\full_eval" "%AGG_ROOT%\decoder-H-MSCTM\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating summary_history metrics for Epoch 120 ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0120
if %errorlevel% neq 0 exit /b %errorlevel%
