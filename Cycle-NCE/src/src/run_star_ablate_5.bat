@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\star-ablation-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 5 Star Pattern Ablations (80 Epochs)
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: G0-Base-Gain0.5
echo ------------------------------------------
uv run run.py --config config_G0-Base-Gain0.5.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G0-Base-Gain0.5\full_eval" "%AGG_ROOT%\G0-Base-Gain0.5\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: G1-Relax-ID
echo ------------------------------------------
uv run run.py --config config_G1-Relax-ID.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G1-Relax-ID\full_eval" "%AGG_ROOT%\G1-Relax-ID\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: G2-Lower-TV
echo ------------------------------------------
uv run run.py --config config_G2-Lower-TV.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G2-Lower-TV\full_eval" "%AGG_ROOT%\G2-Lower-TV\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: G3-Soft-HF
echo ------------------------------------------
uv run run.py --config config_G3-Soft-HF.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G3-Soft-HF\full_eval" "%AGG_ROOT%\G3-Soft-HF\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: G4-Macro-Patch
echo ------------------------------------------
uv run run.py --config config_G4-Macro-Patch.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G4-Macro-Patch\full_eval" "%AGG_ROOT%\G4-Macro-Patch\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating summary_history metrics for Epoch 80 ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_080
if %errorlevel% neq 0 exit /b %errorlevel%
