@echo off
echo ========================================================
echo   LatentAdaCUT Ablation Study (3x 120 Epochs)
echo   Target Device: RTX 3060 12G / 4070
echo ========================================================

echo.
echo [1/3] Launching Experiment A: Texture Overdrive (Patch=7, Low Identity)...
uv run run.py --config config_expA_texture.json
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Experiment A crashed or OOM. Proceeding to B...
) else (
    echo [SUCCESS] Experiment A completed.
)

echo.
echo [2/3] Launching Experiment B: Depth Unleashed (8 ResBlocks, Gain=1.0)...
uv run run.py --config config_expB_depth.json
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Experiment B crashed or OOM. Proceeding to C...
) else (
    echo [SUCCESS] Experiment B completed.
)

echo.
echo [3/3] Launching Experiment C: Optimization Shock (Weight Decay, Semigroup)...
uv run run.py --config config_expC_reg.json
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Experiment C crashed.
) else (
    echo [SUCCESS] Experiment C completed.
)

echo.
echo ========================================================
echo   ALL EXPERIMENTS CONCLUDED.
echo   Check outputs in:
echo   - ../spatial-adagn-expA-texture
echo   - ../spatial-adagn-expB-depth
echo   - ../spatial-adagn-expC-reg
echo ========================================================
pause