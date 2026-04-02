@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/10] Running G1_NoRes_NoSkip...
uv run run.py --config config_G1_NoRes_NoSkip.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/10] Running G1_NoRes_Naive_Clean...
uv run run.py --config config_G1_NoRes_Naive_Clean.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/10] Running G1_NoRes_Naive_Style...
uv run run.py --config config_G1_NoRes_Naive_Style.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/10] Running G1_NoRes_Adapt_Clean...
uv run run.py --config config_G1_NoRes_Adapt_Clean.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/10] Running G1_NoRes_Adapt_Style...
uv run run.py --config config_G1_NoRes_Adapt_Style.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/10] Running G2_Res_NoSkip...
uv run run.py --config config_G2_Res_NoSkip.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/10] Running G2_Res_Naive_Clean...
uv run run.py --config config_G2_Res_Naive_Clean.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/10] Running G2_Res_Naive_Style...
uv run run.py --config config_G2_Res_Naive_Style.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/10] Running G2_Res_Adapt_Clean...
uv run run.py --config config_G2_Res_Adapt_Clean.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [10/10] Running G2_Res_Adapt_Style...
uv run run.py --config config_G2_Res_Adapt_Style.json
if %errorlevel% neq 0 exit /b %errorlevel%

