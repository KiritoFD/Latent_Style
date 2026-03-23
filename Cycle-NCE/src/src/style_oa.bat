@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
echo ==========================================
echo Running Style OA Orthogonal Ablation (8 exps)
echo ==========================================
echo.
echo ------------------------------------------
echo Running Experiment 1: style_oa_1_lr2e4_wc2_swd60_id15_e120
echo ------------------------------------------
uv run run.py --config config_style_oa_1_lr2e4_wc2_swd60_id15_e120.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 2: style_oa_2_lr2e4_wc2_swd90_id30_e120
echo ------------------------------------------
uv run run.py --config config_style_oa_2_lr2e4_wc2_swd90_id30_e120.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 3: style_oa_3_lr2e4_wc5_swd60_id30_e120
echo ------------------------------------------
uv run run.py --config config_style_oa_3_lr2e4_wc5_swd60_id30_e120.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 4: style_oa_4_lr2e4_wc5_swd90_id15_e120
echo ------------------------------------------
uv run run.py --config config_style_oa_4_lr2e4_wc5_swd90_id15_e120.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 5: style_oa_5_lr5e4_wc2_swd60_id30_e120
echo ------------------------------------------
uv run run.py --config config_style_oa_5_lr5e4_wc2_swd60_id30_e120.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 6: style_oa_6_lr5e4_wc2_swd90_id15_e120
echo ------------------------------------------
uv run run.py --config config_style_oa_6_lr5e4_wc2_swd90_id15_e120.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 7: style_oa_7_lr5e4_wc5_swd60_id15_e120
echo ------------------------------------------
uv run run.py --config config_style_oa_7_lr5e4_wc5_swd60_id15_e120.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 8: style_oa_8_lr5e4_wc5_swd90_id30_e120
echo ------------------------------------------
uv run run.py --config config_style_oa_8_lr5e4_wc5_swd90_id30_e120.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo Style OA ablation finished.
