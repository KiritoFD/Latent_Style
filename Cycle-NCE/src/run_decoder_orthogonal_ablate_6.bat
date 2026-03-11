@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\decoder-orthogonal-ablation-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 6 Orthogonal Decoder Ablations
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: exp0-baseline
echo ------------------------------------------
uv run run.py --config config_exp0-baseline.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp0-baseline\full_eval" "%AGG_ROOT%\exp0-baseline\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: exp1-hf-ratio-4p0
echo ------------------------------------------
uv run run.py --config config_exp1-hf-ratio-4p0.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp1-hf-ratio-4p0\full_eval" "%AGG_ROOT%\exp1-hf-ratio-4p0\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: exp2-large-patches
echo ------------------------------------------
uv run run.py --config config_exp2-large-patches.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp2-large-patches\full_eval" "%AGG_ROOT%\exp2-large-patches\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: exp3-hard-cdf
echo ------------------------------------------
uv run run.py --config config_exp3-hard-cdf.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp3-hard-cdf\full_eval" "%AGG_ROOT%\exp3-hard-cdf\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: exp4-skip-retain-plus20
echo ------------------------------------------
uv run run.py --config config_exp4-skip-retain-plus20.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp4-skip-retain-plus20\full_eval" "%AGG_ROOT%\exp4-skip-retain-plus20\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: exp5-proj-1280
echo ------------------------------------------
uv run run.py --config config_exp5-proj-1280.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp5-proj-1280\full_eval" "%AGG_ROOT%\exp5-proj-1280\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating summary_history metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0120
if %errorlevel% neq 0 exit /b %errorlevel%
