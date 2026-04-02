@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/3] Running Z01_ResOff_Adapt_ZeroIDT...
uv run run.py --config config_Z01_ResOff_Adapt_ZeroIDT.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/3] Running Z02_ResOff_Adapt_ZeroIDT_HFSWD...
uv run run.py --config config_Z02_ResOff_Adapt_ZeroIDT_HFSWD.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/3] Running Z03_ResOff_None_ZeroIDT...
uv run run.py --config config_Z03_ResOff_None_ZeroIDT.json
if %errorlevel% neq 0 exit /b %errorlevel%

