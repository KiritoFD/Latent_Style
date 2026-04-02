@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/3] Running T01_ResOn_None_Swin4_Noise...
uv run run.py --config history_configs\config_T01_ResOn_None_Swin4_Noise.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/3] Running T02_ResOff_Adapt_Conv1_LowIDT...
uv run run.py --config history_configs\config_T02_ResOff_Adapt_Conv1_LowIDT.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/3] Running T03_ResOff_Adapt_Conv1_HFSWD...
uv run run.py --config history_configs\config_T03_ResOff_Adapt_Conv1_HFSWD.json
if %errorlevel% neq 0 exit /b %errorlevel%

