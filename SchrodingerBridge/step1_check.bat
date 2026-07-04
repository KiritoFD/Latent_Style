@echo off
REM Step 1: Check checkpoints for FC-SB Phase 2 experiments

echo Creating directory on remote server...
ssh -p 2222 administrator@100.115.18.62 "mkdir -p /home/xy"

echo.
echo Uploading check script...
scp -P 2222 "g:\GitHub\Latent_Style\SchrodingerBridge\check_checkpoint.sh" administrator@100.115.18.62:/home/xy/check_checkpoint.sh

echo.
echo Executing checkpoint check...
ssh -p 2222 administrator@100.115.18.62 "chmod +x /home/xy/check_checkpoint.sh && bash /home/xy/check_checkpoint.sh"

echo.
echo Downloading results...
scp -P 2222 administrator@100.115.18.62:/home/xy/checkpoint_check.txt "g:\GitHub\Latent_Style\SchrodingerBridge\checkpoint_check.txt"

echo.
echo Done! Results saved to checkpoint_check.txt
type "g:\GitHub\Latent_Style\SchrodingerBridge\checkpoint_check.txt"
