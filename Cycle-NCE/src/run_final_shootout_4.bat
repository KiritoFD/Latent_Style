@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\final-shootout-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 4-Way Final Shootout (120 Epochs)
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: M1-Aggressive-Fine
echo ------------------------------------------
uv run run.py --config config_M1-Aggressive-Fine.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_M1-Aggressive-Fine\full_eval" "%AGG_ROOT%\ablate_M1-Aggressive-Fine\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: M2-Smooth-Impasto
echo ------------------------------------------
uv run run.py --config config_M2-Smooth-Impasto.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_M2-Smooth-Impasto\full_eval" "%AGG_ROOT%\ablate_M2-Smooth-Impasto\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: M3-Macro-Flowing
echo ------------------------------------------
uv run run.py --config config_M3-Macro-Flowing.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_M3-Macro-Flowing\full_eval" "%AGG_ROOT%\ablate_M3-Macro-Flowing\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: M4-Gentle-Balanced
echo ------------------------------------------
uv run run.py --config config_M4-Gentle-Balanced.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_M4-Gentle-Balanced\full_eval" "%AGG_ROOT%\ablate_M4-Gentle-Balanced\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating summary_history metrics for Epoch 120 ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0120
if %errorlevel% neq 0 exit /b %errorlevel%
