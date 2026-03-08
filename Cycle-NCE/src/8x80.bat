@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\ablate-8x80-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 8-way ablation (80 Epochs, eval@40/80)
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: ablate_E1_Macro19_Rigid_LR14e4
echo ------------------------------------------
uv run run.py --config config_ablate_E1_Macro19_Rigid_LR14e4.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_E1_Macro19_Rigid_LR14e4\full_eval" "%AGG_ROOT%\ablate_E1_Macro19_Rigid_LR14e4\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: ablate_E2_15Series_Rigid_LR14e4
echo ------------------------------------------
uv run run.py --config config_ablate_E2_15Series_Rigid_LR14e4.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_E2_15Series_Rigid_LR14e4\full_eval" "%AGG_ROOT%\ablate_E2_15Series_Rigid_LR14e4\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: ablate_E3_15Series_Soft_LR14e4
echo ------------------------------------------
uv run run.py --config config_ablate_E3_15Series_Soft_LR14e4.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_E3_15Series_Soft_LR14e4\full_eval" "%AGG_ROOT%\ablate_E3_15Series_Soft_LR14e4\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: ablate_E4_9Series_Rigid_LR14e4
echo ------------------------------------------
uv run run.py --config config_ablate_E4_9Series_Rigid_LR14e4.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_E4_9Series_Rigid_LR14e4\full_eval" "%AGG_ROOT%\ablate_E4_9Series_Rigid_LR14e4\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: ablate_E5_9Series_Soft_LR14e4
echo ------------------------------------------
uv run run.py --config config_ablate_E5_9Series_Soft_LR14e4.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_E5_9Series_Soft_LR14e4\full_eval" "%AGG_ROOT%\ablate_E5_9Series_Soft_LR14e4\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: ablate_E6_9Series_Free_LR30e4
echo ------------------------------------------
uv run run.py --config config_ablate_E6_9Series_Free_LR30e4.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_E6_9Series_Free_LR30e4\full_eval" "%AGG_ROOT%\ablate_E6_9Series_Free_LR30e4\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: ablate_E7_15Series_Free_LR14e4_wNCE1
echo ------------------------------------------
uv run run.py --config config_ablate_E7_15Series_Free_LR14e4_wNCE1.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_E7_15Series_Free_LR14e4_wNCE1\full_eval" "%AGG_ROOT%\ablate_E7_15Series_Free_LR14e4_wNCE1\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: ablate_E8_MicroExtreme_Soft_LR14e4
echo ------------------------------------------
uv run run.py --config config_ablate_E8_MicroExtreme_Soft_LR14e4.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\ablate_E8_MicroExtreme_Soft_LR14e4\full_eval" "%AGG_ROOT%\ablate_E8_MicroExtreme_Soft_LR14e4\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating epoch_0040 metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0040 --summary-csv summary_history_metrics_e040.csv
if %errorlevel% neq 0 exit /b %errorlevel%
echo Aggregating epoch_0080 metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0080 --summary-csv summary_history_metrics_e080.csv
if %errorlevel% neq 0 exit /b %errorlevel%
