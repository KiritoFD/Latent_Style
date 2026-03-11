@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\nce-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 6-way orthogonal ablation (80 Epochs, eval@40/80)
echo Sweep tag=nce, base batch_size=256
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: A1_Deep_Only
echo ------------------------------------------
uv run run.py --config config_nce_A1_Deep_Only.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\nce_A1_Deep_Only\full_eval" "%AGG_ROOT%\nce_A1_Deep_Only\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: A2_Shallow_Only
echo ------------------------------------------
uv run run.py --config config_nce_A2_Shallow_Only.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\nce_A2_Shallow_Only\full_eval" "%AGG_ROOT%\nce_A2_Shallow_Only\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: A3_Patch_Coarse
echo ------------------------------------------
uv run run.py --config config_nce_A3_Patch_Coarse.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\nce_A3_Patch_Coarse\full_eval" "%AGG_ROOT%\nce_A3_Patch_Coarse\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: A4_Patch_Fine
echo ------------------------------------------
uv run run.py --config config_nce_A4_Patch_Fine.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\nce_A4_Patch_Fine\full_eval" "%AGG_ROOT%\nce_A4_Patch_Fine\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: A5_High_TV
echo ------------------------------------------
uv run run.py --config config_nce_A5_High_TV.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\nce_A5_High_TV\full_eval" "%AGG_ROOT%\nce_A5_High_TV\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: A6_Strong_ID
echo ------------------------------------------
uv run run.py --config config_nce_A6_Strong_ID.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\nce_A6_Strong_ID\full_eval" "%AGG_ROOT%\nce_A6_Strong_ID\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating epoch_0040 metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0040 --summary-csv summary_history_metrics_e040.csv
if %errorlevel% neq 0 exit /b %errorlevel%
echo Aggregating epoch_0080 metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0080 --summary-csv summary_history_metrics_e080.csv
if %errorlevel% neq 0 exit /b %errorlevel%
