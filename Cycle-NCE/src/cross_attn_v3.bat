@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Running final_12_base_ref...
echo ==================================================
uv run run.py --config config_final_12_base_ref.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_1_lr4_id35_swd60_c5...
echo ==================================================
uv run run.py --config config_final_1_lr4_id35_swd60_c5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_2_lr5_id30_swd80_c2...
echo ==================================================
uv run run.py --config config_final_2_lr5_id30_swd80_c2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_3_lr6_id25_swd60_c5...
echo ==================================================
uv run run.py --config config_final_3_lr6_id25_swd60_c5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_4_lr8_id30_swd80_c2...
echo ==================================================
uv run run.py --config config_final_4_lr8_id30_swd80_c2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_5_lr5_id15_swd120_c5...
echo ==================================================
uv run run.py --config config_final_5_lr5_id15_swd120_c5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_6_lr5_id20_swd150_c10...
echo ==================================================
uv run run.py --config config_final_6_lr5_id20_swd150_c10.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_7_lr8_id15_swd120_c5...
echo ==================================================
uv run run.py --config config_final_7_lr8_id15_swd120_c5.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_8_lr8_id20_swd150_c10...
echo ==================================================
uv run run.py --config config_final_8_lr8_id20_swd150_c10.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_9_dim128_tok64...
echo ==================================================
uv run run.py --config config_final_9_dim128_tok64.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_10_dim96_tok128...
echo ==================================================
uv run run.py --config config_final_10_dim96_tok128.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting final_11_dim128_tok128...
echo ==================================================
uv run run.py --config config_final_11_dim128_tok128.json
if %errorlevel% neq 0 exit /b %errorlevel%

