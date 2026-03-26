@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
echo ==========================================
echo Running strong ablation (6 exps)
echo ==========================================
echo.
echo ------------------------------------------
echo Running Experiment 1: strong_idt30_swd100_color50
echo ------------------------------------------
uv run run.py --config config_strong_idt30_swd100_color50.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 2: strong_idt30_swd100_color80
echo ------------------------------------------
uv run run.py --config config_strong_idt30_swd100_color80.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 3: strong_idt30_swd100_color20
echo ------------------------------------------
uv run run.py --config config_strong_idt30_swd100_color20.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 4: strong_idt30_swd150_color50
echo ------------------------------------------
uv run run.py --config config_strong_idt30_swd150_color50.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 5: strong_idt30_swd150_color80
echo ------------------------------------------
uv run run.py --config config_strong_idt30_swd150_color80.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 6: strong_idt30_swd150_color20
echo ------------------------------------------
uv run run.py --config config_strong_idt30_swd150_color20.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo strong ablation finished.
