@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
echo ==========================================
echo Running Color Orthogonal Sweep (rank16, e60)
echo ==========================================
echo.
echo ------------------------------------------
echo Running Experiment: color_01_adain_wc2_tv05_r16_e60
echo ------------------------------------------
uv run run.py --config config_color_01_adain_wc2_tv05_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: color_02_adain_wc2_tv20_r16_e60
echo ------------------------------------------
uv run run.py --config config_color_02_adain_wc2_tv20_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: color_03_adain_wc5_tv05_r16_e60
echo ------------------------------------------
uv run run.py --config config_color_03_adain_wc5_tv05_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: color_04_adain_wc5_tv20_r16_e60
echo ------------------------------------------
uv run run.py --config config_color_04_adain_wc5_tv20_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: color_05_hist_wc2_tv05_r16_e60
echo ------------------------------------------
uv run run.py --config config_color_05_hist_wc2_tv05_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: color_06_hist_wc2_tv20_r16_e60
echo ------------------------------------------
uv run run.py --config config_color_06_hist_wc2_tv20_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: color_07_hist_wc5_tv05_r16_e60
echo ------------------------------------------
uv run run.py --config config_color_07_hist_wc5_tv05_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment: color_08_hist_wc5_tv20_r16_e60
echo ------------------------------------------
uv run run.py --config config_color_08_hist_wc5_tv20_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo Color orthogonal sweep finished.
