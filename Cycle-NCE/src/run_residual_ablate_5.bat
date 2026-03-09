@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\residual-ablation-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 5 Residual Ablations
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: residual-base-gain05
echo ------------------------------------------
uv run run.py --config config_residual-base-gain05.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\residual-base-gain05\full_eval" "%AGG_ROOT%\residual-base-gain05\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: residual-relax-id
echo ------------------------------------------
uv run run.py --config config_residual-relax-id.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\residual-relax-id\full_eval" "%AGG_ROOT%\residual-relax-id\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: residual-lower-tv
echo ------------------------------------------
uv run run.py --config config_residual-lower-tv.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\residual-lower-tv\full_eval" "%AGG_ROOT%\residual-lower-tv\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: residual-soft-hf
echo ------------------------------------------
uv run run.py --config config_residual-soft-hf.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\residual-soft-hf\full_eval" "%AGG_ROOT%\residual-soft-hf\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: residual-macro-patch
echo ------------------------------------------
uv run run.py --config config_residual-macro-patch.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\residual-macro-patch\full_eval" "%AGG_ROOT%\residual-macro-patch\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating summary_history metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0120
if %errorlevel% neq 0 exit /b %errorlevel%
