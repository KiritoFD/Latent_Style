@echo off
:: Wait for StyleID R5 python process to finish, then run StyTR-2 + AesPA-Net serially
:: Deployed to I:\wait_and_chain_baselines.bat on remote
setlocal

echo === WAIT_AND_CHAIN START: %DATE% %TIME% ===

:WAIT_LOOP
:: Check if any python.exe is still running (StyleID R5)
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I "python.exe" >NUL
if %ERRORLEVEL%==0 (
    :: Python still running, wait 60s and check again
    timeout /T 60 /NOBREAK >NUL
    goto WAIT_LOOP
)

echo === Python finished, starting serial launcher: %DATE% %TIME% ===

:: Run StyTR-2 + AesPA-Net serially
call "I:\launch_stytr2_aespa_serial.bat"

echo === WAIT_AND_CHAIN END: %DATE% %TIME% ===
endlocal
