@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting arch_1_pM_sC_dH...
echo ==================================================
uv run run.py --config config_arch_1_pM_sC_dH.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting arch_2_pM_sA_dL...
echo ==================================================
uv run run.py --config config_arch_2_pM_sA_dL.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting arch_3_pM_sC_dL...
echo ==================================================
uv run run.py --config config_arch_3_pM_sC_dL.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting arch_4_pM_sA_dH...
echo ==================================================
uv run run.py --config config_arch_4_pM_sA_dH.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting arch_5_pMW_sA_dH...
echo ==================================================
uv run run.py --config config_arch_5_pMW_sA_dH.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting arch_6_pMW_sC_dL...
echo ==================================================
uv run run.py --config config_arch_6_pMW_sC_dL.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting arch_7_pMW_sA_dL...
echo ==================================================
uv run run.py --config config_arch_7_pMW_sA_dL.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting arch_8_pMW_sC_dH...
echo ==================================================
uv run run.py --config config_arch_8_pMW_sC_dH.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting weight_0_base...
echo ==================================================
uv run run.py --config config_weight_0_base.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting weight_1_swd_low...
echo ==================================================
uv run run.py --config config_weight_1_swd_low.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting weight_2_swd_high...
echo ==================================================
uv run run.py --config config_weight_2_swd_high.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting weight_3_color_low...
echo ==================================================
uv run run.py --config config_weight_3_color_low.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting weight_4_color_high...
echo ==================================================
uv run run.py --config config_weight_4_color_high.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting weight_5_id_loose...
echo ==================================================
uv run run.py --config config_weight_5_id_loose.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting weight_6_id_tight...
echo ==================================================
uv run run.py --config config_weight_6_id_tight.json
if %errorlevel% neq 0 exit /b %errorlevel%

