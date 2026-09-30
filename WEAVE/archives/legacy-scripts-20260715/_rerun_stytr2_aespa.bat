@echo off
:: Re-run StyTR-2 + AesPA-Net inference (both fixed) then eval
:: StyTR-2: torch._six patched in ViT_helper.py
:: AesPA-Net: skvideo import patched in utils.py
setlocal

echo === RERUN STYTR2+AESPA START: %DATE% %TIME% ===

:: Step 1: StyTR-2 on all 3 datasets (ViT_helper.py patched)
echo --- Step 1: StyTR-2 inference (fixed) ---
call "I:\StyTR2\launch_stytr2.bat"

:: Step 2: AesPA-Net on all 3 datasets (utils.py patched)
echo --- Step 2: AesPA-Net inference (fixed) ---
call "I:\AesPA-Net\launch_aespa.bat"

:: Step 3: Run evals for StyTR-2 and AesPA-Net
echo --- Step 3: Run evals ---
powershell -ExecutionPolicy Bypass -File "I:\run_stytr2_aespa_evals.ps1"

echo === RERUN STYTR2+AESPA END: %DATE% %TIME% ===
endlocal
