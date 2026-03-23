@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\decoder-dynamic-epoch-aggregate"
set "COOLDOWN_SEC=8"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 7 Decoder-D Ablations (Custom Matrix)
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: exp01_tv_0p02
echo ------------------------------------------
uv run run.py --config config_exp01_tv_0p02.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp01_tv_0p02\full_eval" "%AGG_ROOT%\exp01_tv_0p02\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: exp02_color_1p5
echo ------------------------------------------
uv run run.py --config config_exp02_color_1p5.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp02_color_1p5\full_eval" "%AGG_ROOT%\exp02_color_1p5\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: exp03_hf_4p0
echo ------------------------------------------
uv run run.py --config config_exp03_hf_4p0.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp03_hf_4p0\full_eval" "%AGG_ROOT%\exp03_hf_4p0\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: exp04_gate_0p8
echo ------------------------------------------
uv run run.py --config config_exp04_gate_0p8.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp04_gate_0p8\full_eval" "%AGG_ROOT%\exp04_gate_0p8\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: exp05_rank_8
echo ------------------------------------------
uv run run.py --config config_exp05_rank_8.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp05_rank_8\full_eval" "%AGG_ROOT%\exp05_rank_8\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: exp06_god_combo
echo ------------------------------------------
uv run run.py --config config_exp06_god_combo.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp06_god_combo\full_eval" "%AGG_ROOT%\exp06_god_combo\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: exp07_user_proj720
echo ------------------------------------------
uv run run.py --config config_exp07_user_proj720.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\exp07_user_proj720\full_eval" "%AGG_ROOT%\exp07_user_proj720\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul

echo.
echo Aggregating summary_history metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0040
if %errorlevel% neq 0 exit /b %errorlevel%
