@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo   R8 Extreme Performance Experiments
echo   Target: LPIPS ^< 0.45, Style ^> 0.70
echo ==========================================
echo.

cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

set "AGG_ROOT=..\R8_Extreme_Aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"

echo [Exp 1/4] R8_Base - Strongest Rank=8 Baseline
echo ==========================================
uv run run.py --config config_R8_Base.json
if %errorlevel% neq 0 (
    echo [ERROR] R8_Base failed!
    exit /b %errorlevel%
)
echo [DONE] R8_Base completed.
echo.

timeout /t 10 /nobreak >nul

echo [Exp 2/4] R8_Small_Patch - Remove Large Receptive Field
echo ==========================================
uv run run.py --config config_R8_Small_Patch.json
if %errorlevel% neq 0 (
    echo [ERROR] R8_Small_Patch failed!
    exit /b %errorlevel%
)
echo [DONE] R8_Small_Patch completed.
echo.

timeout /t 10 /nobreak >nul

echo [Exp 3/4] R8_NCE - Activate Semantic Anchors
echo ==========================================
uv run run.py --config config_R8_NCE.json
if %errorlevel% neq 0 (
    echo [ERROR] R8_NCE failed!
    exit /b %errorlevel%
)
echo [DONE] R8_NCE completed.
echo.

timeout /t 10 /nobreak >nul

echo [Exp 4/4] R8_Combo - Ultimate Form (Small Patch + NCE)
echo ==========================================
uv run run.py --config config_R8_Combo.json
if %errorlevel% neq 0 (
    echo [ERROR] R8_Combo failed!
    exit /b %errorlevel%
)
echo [DONE] R8_Combo completed.
echo.

echo ==========================================
echo   All 4 Experiments Completed!
echo   Check eval_cache for results
echo ==========================================
echo.
echo Key metrics to check (Epoch 80):
echo   - LPIPS ^< 0.45 (content preservation)
echo   - CLIP Style ^> 0.70 (style transfer quality)
echo.

