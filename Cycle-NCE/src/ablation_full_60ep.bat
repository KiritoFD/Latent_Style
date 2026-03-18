@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
set "AGG_ROOT=..\ablation-full-aggregate"
set "COOLDOWN_SEC=8"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"
echo ==========================================
echo Starting 12 Full-Stack Ablation Experiments
echo ==========================================

echo.
echo ------------------------------------------
echo Running Experiment: abl_heavy_decoder
echo ------------------------------------------
uv run run.py --config config_abl_heavy_decoder.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_heavy_decoder\full_eval" "%AGG_ROOT%\abl_heavy_decoder\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: abl_no_residual
echo ------------------------------------------
uv run run.py --config config_abl_no_residual.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_no_residual\full_eval" "%AGG_ROOT%\abl_no_residual\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: abl_vanilla_gn
echo ------------------------------------------
uv run run.py --config config_abl_vanilla_gn.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_vanilla_gn\full_eval" "%AGG_ROOT%\abl_vanilla_gn\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: abl_no_skip_filter
echo ------------------------------------------
uv run run.py --config config_abl_no_skip_filter.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_no_skip_filter\full_eval" "%AGG_ROOT%\abl_no_skip_filter\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: abl_no_id
echo ------------------------------------------
uv run run.py --config config_abl_no_id.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_no_id\full_eval" "%AGG_ROOT%\abl_no_id\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: abl_hard_sort
echo ------------------------------------------
uv run run.py --config config_abl_hard_sort.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_hard_sort\full_eval" "%AGG_ROOT%\abl_hard_sort\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: abl_no_hf_swd
echo ------------------------------------------
uv run run.py --config config_abl_no_hf_swd.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_no_hf_swd\full_eval" "%AGG_ROOT%\abl_no_hf_swd\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: abl_no_color
echo ------------------------------------------
uv run run.py --config config_abl_no_color.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_no_color\full_eval" "%AGG_ROOT%\abl_no_color\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: abl_no_tv
echo ------------------------------------------
uv run run.py --config config_abl_no_tv.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\abl_no_tv\full_eval" "%AGG_ROOT%\abl_no_tv\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: scale_c64
echo ------------------------------------------
uv run run.py --config config_scale_c64.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\scale_c64\full_eval" "%AGG_ROOT%\scale_c64\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: scale_c256
echo ------------------------------------------
uv run run.py --config config_scale_c256.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\scale_c256\full_eval" "%AGG_ROOT%\scale_c256\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul
echo.
echo ------------------------------------------
echo Running Experiment: baseline
echo ------------------------------------------
uv run run.py --config config_baseline.json
if %errorlevel% neq 0 exit /b %errorlevel%
robocopy "..\baseline\full_eval" "%AGG_ROOT%\baseline\full_eval" /E /R:1 /W:1 /XD images
if %errorlevel% geq 8 exit /b %errorlevel%
echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...
timeout /t %COOLDOWN_SEC% /nobreak >nul

echo.
echo Aggregating summary_history metrics ...
uv run python ..\scripts\collect_ablation_results.py --root "%AGG_ROOT%" --output-dir "%AGG_ROOT%" --epoch-dir epoch_0060
if %errorlevel% neq 0 exit /b %errorlevel%
