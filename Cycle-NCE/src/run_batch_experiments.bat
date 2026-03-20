@echo off
cd /d "%~dp0"
echo ============================================================
echo Latent Color Loss Experiment Batch
echo Started: %date% %time%
echo ============================================================


echo.
echo [1/6] Running config_Exp_00_TV_Anchor.json
echo ============================================================
uv run run.py --config config_Exp_00_TV_Anchor.json
if errorlevel 1 (
    echo ERROR: config_Exp_00_TV_Anchor.json failed!
    goto :end
)
echo config_Exp_00_TV_Anchor.json completed at %date% %time%

echo.
echo [2/6] Running config_Exp_01_Stats.json
echo ============================================================
uv run run.py --config config_Exp_01_Stats.json
if errorlevel 1 (
    echo ERROR: config_Exp_01_Stats.json failed!
    goto :end
)
echo config_Exp_01_Stats.json completed at %date% %time%

echo.
echo [3/6] Running config_Exp_02_Hist.json
echo ============================================================
uv run run.py --config config_Exp_02_Hist.json
if errorlevel 1 (
    echo ERROR: config_Exp_02_Hist.json failed!
    goto :end
)
echo config_Exp_02_Hist.json completed at %date% %time%

echo.
echo [4/6] Running config_Exp_03_Wasserstein.json
echo ============================================================
uv run run.py --config config_Exp_03_Wasserstein.json
if errorlevel 1 (
    echo ERROR: config_Exp_03_Wasserstein.json failed!
    goto :end
)
echo config_Exp_03_Wasserstein.json completed at %date% %time%

echo.
echo [5/6] Running config_Exp_04_StatsWass.json
echo ============================================================
uv run run.py --config config_Exp_04_StatsWass.json
if errorlevel 1 (
    echo ERROR: config_Exp_04_StatsWass.json failed!
    goto :end
)
echo config_Exp_04_StatsWass.json completed at %date% %time%

echo.
echo [6/6] Running config_Exp_05_Spectrum.json
echo ============================================================
uv run run.py --config config_Exp_05_Spectrum.json
if errorlevel 1 (
    echo ERROR: config_Exp_05_Spectrum.json failed!
    goto :end
)
echo config_Exp_05_Spectrum.json completed at %date% %time%

echo.
echo ============================================================
echo All experiments completed!
echo End time: %date% %time%
echo ============================================================
:end
pause
