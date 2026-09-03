@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Running Exp00_Baseline...
echo ==================================================
uv run run.py --config config.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp01_Patch_Micro...
echo ==================================================
uv run run.py --config config_Exp01_Patch_Micro.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp02_Patch_Macro...
echo ==================================================
uv run run.py --config config_Exp02_Patch_Macro.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp03_Color_Mid...
echo ==================================================
uv run run.py --config config_Exp03_Color_Mid.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp04_Color_Strong...
echo ==================================================
uv run run.py --config config_Exp04_Color_Strong.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp05_LumaCh_Off...
echo ==================================================
uv run run.py --config config_Exp05_LumaCh_Off.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp06_LumaCh_Strong...
echo ==================================================
uv run run.py --config config_Exp06_LumaCh_Strong.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp07_LumaRange_Off...
echo ==================================================
uv run run.py --config config_Exp07_LumaRange_Off.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp08_LumaRange_Strong...
echo ==================================================
uv run run.py --config config_Exp08_LumaRange_Strong.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp09_LumaRange_Wide...
echo ==================================================
uv run run.py --config config_Exp09_LumaRange_Wide.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==================================================
echo Starting Exp10_StyleMod_LegacyDict...
echo ==================================================
uv run run.py --config config_Exp10_StyleMod_LegacyDict.json
if %errorlevel% neq 0 exit /b %errorlevel%

