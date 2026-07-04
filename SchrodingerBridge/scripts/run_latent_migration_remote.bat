@echo off
REM Run latent migration training on remote WSL via SSH (foreground, blocking)
REM SAMST-latent (15 epochs) + SaMam-latent (10000 iters)
setlocal
set HOST=administrator@100.115.18.62
set PORT=2222

echo [%date% %time%] Starting latent migration training on remote...
ssh -p %PORT% -o LogLevel=ERROR %HOST% "wsl -- bash -lc 'bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/run_latent_migration_train.sh'"
echo [%date% %time%] Remote training finished. Exit code: %ERRORLEVEL%
endlocal
