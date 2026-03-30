@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_base_hf_off...
echo ==================================================
uv run run.py --config config_p_base_hf_off.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_base_hf_1p0...
echo ==================================================
uv run run.py --config config_p_base_hf_1p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_base_hf_3p0...
echo ==================================================
uv run run.py --config config_p_base_hf_3p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_1_5_9_15_hf_off...
echo ==================================================
uv run run.py --config config_p_1_5_9_15_hf_off.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_1_5_9_15_hf_1p0...
echo ==================================================
uv run run.py --config config_p_1_5_9_15_hf_1p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_1_5_9_15_hf_3p0...
echo ==================================================
uv run run.py --config config_p_1_5_9_15_hf_3p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_5_9_15_25_hf_off...
echo ==================================================
uv run run.py --config config_p_5_9_15_25_hf_off.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_5_9_15_25_hf_1p0...
echo ==================================================
uv run run.py --config config_p_5_9_15_25_hf_1p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_5_9_15_25_hf_3p0...
echo ==================================================
uv run run.py --config config_p_5_9_15_25_hf_3p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_5_9_15_hf_off...
echo ==================================================
uv run run.py --config config_p_5_9_15_hf_off.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_5_9_15_hf_1p0...
echo ==================================================
uv run run.py --config config_p_5_9_15_hf_1p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting p_5_9_15_hf_3p0...
echo ==================================================
uv run run.py --config config_p_5_9_15_hf_3p0.json
if %errorlevel% neq 0 exit /b %errorlevel%

