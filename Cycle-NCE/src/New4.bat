@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/4] Running N01_Stylized_Naive...
uv run run.py --config config_N01_Stylized_Naive.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/4] Running N02_Stylized_Adaptive...
uv run run.py --config config_N02_Stylized_Adaptive.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/4] Running N03_Stylized_Adaptive_Retain0p2...
uv run run.py --config config_N03_Stylized_Adaptive_Retain0p2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/4] Running N04_Stylized_Norm...
uv run run.py --config config_N04_Stylized_Norm.json
if %errorlevel% neq 0 exit /b %errorlevel%

