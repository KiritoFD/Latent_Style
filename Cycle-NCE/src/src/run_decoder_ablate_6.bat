@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\decoder-ablation-aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 6 Decoder Ablations
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: decoder-A-anchor-nohf
echo ------------------------------------------
uv run run.py --config config_decoder-A-anchor-nohf.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\decoder-A-anchor-nohf\full_eval" "%AGG_ROOT%\decoder-A-anchor-nohf\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: decoder-B-hf-strict-id
echo ------------------------------------------
uv run run.py --config config_decoder-B-hf-strict-id.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\decoder-B-hf-strict-id\full_eval" "%AGG_ROOT%\decoder-B-hf-strict-id\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: decoder-C-relaxed-id-nohf
echo ------------------------------------------
uv run run.py --config config_decoder-C-relaxed-id-nohf.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\decoder-C-relaxed-id-nohf\full_eval" "%AGG_ROOT%\decoder-C-relaxed-id-nohf\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: decoder-D-sweetspot
echo ------------------------------------------
uv run run.py --config config_decoder-D-sweetspot.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\decoder-D-sweetspot\full_eval" "%AGG_ROOT%\decoder-D-sweetspot\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: decoder-E-extreme-brush
echo ------------------------------------------
uv run run.py --config config_decoder-E-extreme-brush.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\decoder-E-extreme-brush\full_eval" "%AGG_ROOT%\decoder-E-extreme-brush\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: decoder-F-tv-off
echo ------------------------------------------
uv run run.py --config config_decoder-F-tv-off.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\decoder-F-tv-off\full_eval" "%AGG_ROOT%\decoder-F-tv-off\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%

echo.
echo Aggregating summary_history metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0080
if %errorlevel% neq 0 exit /b %errorlevel%
