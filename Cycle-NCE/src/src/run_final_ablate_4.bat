@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\final-ablation-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 4 Orthogonal Ablations (120 Epochs, Eval@40/80/120)
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: G0_Balanced_Base
echo ------------------------------------------
uv run run.py --config config_G0_Balanced_Base.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G0_Balanced_Base\full_eval" "%AGG_ROOT%\G0_Balanced_Base\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: G1_High_HF_Test
echo ------------------------------------------
uv run run.py --config config_G1_High_HF_Test.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G1_High_HF_Test\full_eval" "%AGG_ROOT%\G1_High_HF_Test\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: G2_High_TV_Test
echo ------------------------------------------
uv run run.py --config config_G2_High_TV_Test.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G2_High_TV_Test\full_eval" "%AGG_ROOT%\G2_High_TV_Test\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: G3_Relaxed_ID_Test
echo ------------------------------------------
uv run run.py --config config_G3_Relaxed_ID_Test.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\G3_Relaxed_ID_Test\full_eval" "%AGG_ROOT%\G3_Relaxed_ID_Test\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating summary_history metrics for Epoch 120 ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0120
if %errorlevel% neq 0 exit /b %errorlevel%
