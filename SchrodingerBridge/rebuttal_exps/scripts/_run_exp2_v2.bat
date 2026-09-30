@echo off
cd /d I:\Github\Latent_Style\WEAVE
if not exist C:\Users\Administrator\logs mkdir C:\Users\Administrator\logs
del /Q exp\rebuttal\exp2_features.pt 2>nul
python -u scripts\exp2_idt_variance.py > C:\Users\Administrator\logs\exp2_idt_variance_v2.log 2>&1
echo EXIT_CODE=%ERRORLEVEL%
