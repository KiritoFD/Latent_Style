@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
start "abl_ll_fm_eval" /min powershell -NoProfile -ExecutionPolicy Bypass -File "I:\Github\Latent_Style\SchrodingerBridge\scripts\_abl_m1_resume.ps1" > "C:\Users\Administrator\logs\abl_ll_fm_resume.out" 2> "C:\Users\Administrator\logs\abl_ll_fm_resume.err"
echo launched
