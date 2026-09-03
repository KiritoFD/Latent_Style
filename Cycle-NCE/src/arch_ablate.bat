@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting E1_wgw_light_h2_g1_d2...
echo ==================================================
uv run run.py --config config_E1_wgw_light_h2_g1_d2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting E2_wgw_heavy_h3_g2_d3...
echo ==================================================
uv run run.py --config config_E2_wgw_heavy_h3_g2_d3.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting E3_wgw_heavy_w16_h3_g2_d3...
echo ==================================================
uv run run.py --config config_E3_wgw_heavy_w16_h3_g2_d3.json
if %errorlevel% neq 0 exit /b %errorlevel%

