@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Running Run_0_Baseline...
echo ==================================================
uv run run.py --config config_Run_0_Baseline.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Run_1_lr_high_8e4...
echo ==================================================
uv run run.py --config config_Run_1_lr_high_8e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Run_2_lr_low_2e4...
echo ==================================================
uv run run.py --config config_Run_2_lr_low_2e4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Run_3_id_loose_15...
echo ==================================================
uv run run.py --config config_Run_3_id_loose_15.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Run_4_id_tight_45...
echo ==================================================
uv run run.py --config config_Run_4_id_tight_45.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Run_5_swd_max_200...
echo ==================================================
uv run run.py --config config_Run_5_swd_max_200.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Run_6_color_bold_100...
echo ==================================================
uv run run.py --config config_Run_6_color_bold_100.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Run_7_lum_strict_10...
echo ==================================================
uv run run.py --config config_Run_7_lum_strict_10.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Run_8_arch_old_dict...
echo ==================================================
uv run run.py --config config_Run_8_arch_old_dict.json
if %errorlevel% neq 0 exit /b %errorlevel%

