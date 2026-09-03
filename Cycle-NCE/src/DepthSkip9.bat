@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/9] Running A01_ResOn_Naive_Conv1...
uv run run.py --config history_configs\config_A01_ResOn_Naive_Conv1.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/9] Running A02_ResOn_None_Swin4...
uv run run.py --config history_configs\config_A02_ResOn_None_Swin4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/9] Running E03_ResOff_None_Conv1...
uv run run.py --config history_configs\config_E03_ResOff_None_Conv1.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/9] Running E04_ResOff_None_Swin2...
uv run run.py --config history_configs\config_E04_ResOff_None_Swin2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/9] Running E05_ResOff_None_Swin4...
uv run run.py --config history_configs\config_E05_ResOff_None_Swin4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/9] Running E06_ResOff_Adapt_Conv1...
uv run run.py --config history_configs\config_E06_ResOff_Adapt_Conv1.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/9] Running E07_ResOff_Adapt_Swin2...
uv run run.py --config history_configs\config_E07_ResOff_Adapt_Swin2.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/9] Running E08_ResOff_Adapt_Swin4...
uv run run.py --config history_configs\config_E08_ResOff_Adapt_Swin4.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/9] Running E09_ResOff_Naive_Conv1...
uv run run.py --config history_configs\config_E09_ResOff_Naive_Conv1.json
if %errorlevel% neq 0 exit /b %errorlevel%

