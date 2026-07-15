@echo off
set CONFIG_NAME=%1
powershell -ExecutionPolicy Bypass -File C:\Users\Administrator\_run_s2_train_eval.ps1 -ConfigName %CONFIG_NAME% > C:\Users\Administrator\logs\s2_%CONFIG_NAME%_out.txt 2>&1
