@echo off
wmic process call create "powershell.exe -ExecutionPolicy Bypass -File I:\Github\Latent_Style\SchrodingerBridge\scripts\run_sty_stage4.ps1"
