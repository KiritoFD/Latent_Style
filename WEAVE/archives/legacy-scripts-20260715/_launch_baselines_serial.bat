@echo off
:: Serial launcher: StyTR-2 (3 datasets) then StyleID R5-WikiArt
:: Run via schtasks to survive SSH disconnects
setlocal

echo === BASELINES SERIAL START: %DATE% %TIME% ===

:: Step 1: StyTR-2 on all 3 datasets
echo --- Step 1: StyTR-2 inference ---
call "I:\StyTR2\launch_stytr2.bat"

:: Step 2: StyleID R5-WikiArt
echo --- Step 2: StyleID R5-WikiArt ---
call "I:\launch_styleid_r5.bat"

echo === BASELINES SERIAL END: %DATE% %TIME% ===
endlocal
