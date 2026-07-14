@echo off
:: Launch StyleID R5-WikiArt inference via schtasks
setlocal
set PYTHON=C:\Program Files\Python312\python.exe
set SCRIPT=I:\Github\Latent_Style\SchrodingerBridge\scripts\run_styleid_r5.py
set LOG=I:\exp_baselines\styleid\r5_wikiart\styleid_r5.log

if not exist "I:\exp_baselines\styleid\r5_wikiart" mkdir "I:\exp_baselines\styleid\r5_wikiart"

echo === StyleID R5 start: %DATE% %TIME% === > "%LOG%"
"%PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1
echo === StyleID R5 end: %DATE% %TIME% === >> "%LOG%"
endlocal
