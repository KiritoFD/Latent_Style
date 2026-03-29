@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting A1_swin_h2_g1_d2...
echo ==================================================
uv run run.py --config config_A1_swin_h2_g1_d2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting A2_swin_h2_g2_d2...
echo ==================================================
uv run run.py --config config_A2_swin_h2_g2_d2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting A3_swin_h3_g2_d2...
echo ==================================================
uv run run.py --config config_A3_swin_h3_g2_d2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting B1_weaver_h3_g2_d1...
echo ==================================================
uv run run.py --config config_B1_weaver_h3_g2_d1.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting B2_weaver_h2_g2_d1...
echo ==================================================
uv run run.py --config config_B2_weaver_h2_g2_d1.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting B3_weaver_h3_g2_d2...
echo ==================================================
uv run run.py --config config_B3_weaver_h3_g2_d2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting C1_asym_h1_g2_d2...
echo ==================================================
uv run run.py --config config_C1_asym_h1_g2_d2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting C2_asym_h2_g2_d3...
echo ==================================================
uv run run.py --config config_C2_asym_h2_g2_d3.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting D1_cgw_h2_g2_d3_impasto_s3_r12...
echo ==================================================
uv run run.py --config config_D1_cgw_h2_g2_d3_impasto_s3_r12.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting D2_cgw_h2_g2_d4_impasto_s3_r12...
echo ==================================================
uv run run.py --config config_D2_cgw_h2_g2_d4_impasto_s3_r12.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting D3_cgw_h2_g2_d3_impasto_s4_r15...
echo ==================================================
uv run run.py --config config_D3_cgw_h2_g2_d3_impasto_s4_r15.json
if %errorlevel% neq 0 exit /b %errorlevel%

