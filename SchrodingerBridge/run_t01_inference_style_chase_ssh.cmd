@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge

python tools\experiments\run_t01_inference_style_chase.py 1> logs\t01_inference_style_chase_ssh.out.log 2> logs\t01_inference_style_chase_ssh.err.log

endlocal
