@echo off
REM ============================================================
REM Latent Color Loss Experiment - First Batch (6 experiments)
REM ============================================================
REM This script runs the 6 key experiments from the experiment plan:
REM   - Exp 0: TV Anchor (baseline without latent color)
REM   - Exp 1: Stats mode
REM   - Exp 2: Hist mode
REM   - Exp 3: Wasserstein mode
REM   - Exp 4: Stats+Wasserstein mode
REM   - Exp 5: Spectrum mode
REM ============================================================

setlocal enabledelayedexpansion

cd /d "I:\Github\Latent_Style\Cycle-NCE\src"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ============================================================
echo Latent Color Loss Experiment Batch
echo Start Time: %date% %time%
echo ============================================================
echo.

REM ------------------------------------------------------------
REM Experiment 0: TV Anchor (w_latent_color=0)
REM ------------------------------------------------------------
echo [Exp 0] TV Anchor - Starting...
uv run python run.py --config config_LCE0_TV_Anchor.json
if errorlevel 1 (
    echo [Exp 0] TV Anchor - FAILED with error code !errorlevel!
    goto :error_handler
) else (
    echo [Exp 0] TV Anchor - COMPLETED successfully
)
echo.

REM ------------------------------------------------------------
REM Experiment 1: Stats mode
REM ------------------------------------------------------------
echo [Exp 1] Stats - Starting...
uv run python run.py --config config_LCE1_Stats.json
if errorlevel 1 (
    echo [Exp 1] Stats - FAILED with error code !errorlevel!
    goto :error_handler
) else (
    echo [Exp 1] Stats - COMPLETED successfully
)
echo.

REM ------------------------------------------------------------
REM Experiment 2: Hist mode
REM ------------------------------------------------------------
echo [Exp 2] Hist - Starting...
uv run python run.py --config config_LCE2_Hist.json
if errorlevel 1 (
    echo [Exp 2] Hist - FAILED with error code !errorlevel!
    goto :error_handler
) else (
    echo [Exp 2] Hist - COMPLETED successfully
)
echo.

REM ------------------------------------------------------------
REM Experiment 3: Wasserstein mode
REM ------------------------------------------------------------
echo [Exp 3] Wasserstein - Starting...
uv run python run.py --config config_LCE3_Wasserstein.json
if errorlevel 1 (
    echo [Exp 3] Wasserstein - FAILED with error code !errorlevel!
    goto :error_handler
) else (
    echo [Exp 3] Wasserstein - COMPLETED successfully
)
echo.

REM ------------------------------------------------------------
REM Experiment 4: Stats+Wasserstein mode
REM ------------------------------------------------------------
echo [Exp 4] Stats+Wasserstein - Starting...
uv run python run.py --config config_LCE4_Stats_Wass.json
if errorlevel 1 (
    echo [Exp 4] Stats+Wasserstein - FAILED with error code !errorlevel!
    goto :error_handler
) else (
    echo [Exp 4] Stats+Wasserstein - COMPLETED successfully
)
echo.

REM ------------------------------------------------------------
REM Experiment 5: Spectrum mode
REM ------------------------------------------------------------
echo [Exp 5] Spectrum - Starting...
uv run python run.py --config config_LCE5_Spectrum.json
if errorlevel 1 (
    echo [Exp 5] Spectrum - FAILED with error code !errorlevel!
    goto :error_handler
) else (
    echo [Exp 5] Spectrum - COMPLETED successfully
)
echo.

echo ============================================================
echo ALL EXPERIMENTS COMPLETED!
echo End Time: %date% %time%
echo ============================================================
goto :eof

:error_handler
echo ============================================================
echo EXPERIMENT FAILED - stopping batch execution
echo Failed at experiment - check logs above
echo Time: %date% %time%
echo ============================================================
exit /b 1
