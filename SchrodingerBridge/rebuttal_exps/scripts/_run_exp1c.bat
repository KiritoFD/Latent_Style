@echo off
cd /d I:\Github\Latent_Style\WEAVE
if not exist C:\Users\Administrator\logs mkdir C:\Users\Administrator\logs
echo === STARTING EXP2 V2 === > C:\Users\Administrator\logs\exp2_idt_variance_v2.log
python -u scripts\exp2_idt_variance.py >> C:\Users\Administrator\logs\exp2_idt_variance_v2.log 2>&1
echo EXP2_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\exp2_idt_variance_v2.log
echo === STARTING EXP1C ADAIN SWEEP === > C:\Users\Administrator\logs\exp1c_adain_sweep.log
python -u scripts\exp1c_adain_sweep_v2.py >> C:\Users\Administrator\logs\exp1c_adain_sweep.log 2>&1
echo EXP1C_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\exp1c_adain_sweep.log
