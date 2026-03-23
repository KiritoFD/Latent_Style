@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo   8组正交消融实验 (Orthogonal Grid Search)
echo   目标: LPIPS ^< 0.45, Style ^> 0.70
echo   预计总时间: ~8小时 (每组约1小时)
echo ================================================================================
echo.

cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

set "AGG_ROOT=..\Orthogonal_Ablation_Aggregate"
if not exist "%AGG_ROOT%" mkdir "%AGG_ROOT%"

:: Exp_01: R=8, 大Patch, NCE关 (基线)
echo.
echo ================================================================================
echo [Exp 01/08] R=8, 大Patch, NCE关 - 基线验证
echo ================================================================================
uv run run.py --config config_Exp_01.json
if %errorlevel% neq 0 (
    echo [ERROR] Exp_01 failed!
    exit /b %errorlevel%
)
echo [DONE] Exp_01 completed.
echo Cooling down...
timeout /t 30 /nobreak >nul

:: Exp_02: R=8, 小Patch, NCE关
echo.
echo ================================================================================
echo [Exp 02/08] R=8, 小Patch, NCE关 - 小感受野验证
echo ================================================================================
uv run run.py --config config_Exp_02.json
if %errorlevel% neq 0 (
    echo [ERROR] Exp_02 failed!
    exit /b %errorlevel%
)
echo [DONE] Exp_02 completed.
echo Cooling down...
timeout /t 30 /nobreak >nul

:: Exp_03: R=8, 大Patch, NCE开
echo.
echo ================================================================================
echo [Exp 03/08] R=8, 大Patch, NCE开 - NCE压制大Patch验证
echo ================================================================================
uv run run.py --config config_Exp_03.json
if %errorlevel% neq 0 (
    echo [ERROR] Exp_03 failed!
    exit /b %errorlevel%
)
echo [DONE] Exp_03 completed.
echo Cooling down...
timeout /t 30 /nobreak >nul

:: Exp_04: R=8, 小Patch, NCE开 (潜在最优解A)
echo.
echo ================================================================================
echo [Exp 04/08] R=8, 小Patch, NCE开 - 潜在最优解A
echo ================================================================================
uv run run.py --config config_Exp_04.json
if %errorlevel% neq 0 (
    echo [ERROR] Exp_04 failed!
    exit /b %errorlevel%
)
echo [DONE] Exp_04 completed.
echo Cooling down...
timeout /t 30 /nobreak >nul

:: Exp_05: R=16, 大Patch, NCE关
echo.
echo ================================================================================
echo [Exp 05/08] R=16, 大Patch, NCE关 - 中等容量验证
echo ================================================================================
uv run run.py --config config_Exp_05.json
if %errorlevel% neq 0 (
    echo [ERROR] Exp_05 failed!
    exit /b %errorlevel%
)
echo [DONE] Exp_05 completed.
echo Cooling down...
timeout /t 30 /nobreak >nul

:: Exp_06: R=16, 小Patch, NCE关
echo.
echo ================================================================================
echo [Exp 06/08] R=16, 小Patch, NCE关 - 小Patch驯服中等容量
echo ================================================================================
uv run run.py --config config_Exp_06.json
if %errorlevel% neq 0 (
    echo [ERROR] Exp_06 failed!
    exit /b %errorlevel%
)
echo [DONE] Exp_06 completed.
echo Cooling down...
timeout /t 30 /nobreak >nul

:: Exp_07: R=16, 大Patch, NCE开
echo.
echo ================================================================================
echo [Exp 07/08] R=16, 大Patch, NCE开 - NCE驾驭大容量
echo ================================================================================
uv run run.py --config config_Exp_07.json
if %errorlevel% neq 0 (
    echo [ERROR] Exp_07 failed!
    exit /b %errorlevel%
)
echo [DONE] Exp_07 completed.
echo Cooling down...
timeout /t 30 /nobreak >nul

:: Exp_08: R=16, 小Patch, NCE开 (潜在最优解B)
echo.
echo ================================================================================
echo [Exp 08/08] R=16, 小Patch, NCE开 - 潜在最优解B
echo ================================================================================
uv run run.py --config config_Exp_08.json
if %errorlevel% neq 0 (
    echo [ERROR] Exp_08 failed!
    exit /b %errorlevel%
)
echo [DONE] Exp_08 completed.

echo.
echo ================================================================================
echo   所有8组实验已完成！
echo   请检查各实验的 full_eval 目录查看评估结果
echo ================================================================================
