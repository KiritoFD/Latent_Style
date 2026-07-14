@echo off
:: Serial launcher: StyTR-2 (3 datasets) then AesPA-Net (3 datasets)
:: Run via schtasks after StyleID R5 completes
setlocal

echo === STYTR2+AESPA SERIAL START: %DATE% %TIME% ===

:: Step 1: StyTR-2 on all 3 datasets (misc.py already patched)
echo --- Step 1: StyTR-2 inference ---
call "I:\StyTR2\launch_stytr2.bat"

:: Step 2: AesPA-Net on all 3 datasets
echo --- Step 2: AesPA-Net inference ---
call "I:\AesPA-Net\launch_aespa.bat"

echo === STYTR2+AESPA SERIAL END: %DATE% %TIME% ===
endlocal
