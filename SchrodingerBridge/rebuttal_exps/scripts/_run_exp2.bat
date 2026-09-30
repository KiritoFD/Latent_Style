@echo off
cd /d I:\Github\Latent_Style\WEAVE
if not exist C:\Users\Administrator\logs mkdir C:\Users\Administrator\logs
python -u scripts\exp2_idt_variance.py > C:\Users\Administrator\logs\exp2_idt_variance.log 2>&1
echo EXIT_CODE=%ERRORLEVEL%
