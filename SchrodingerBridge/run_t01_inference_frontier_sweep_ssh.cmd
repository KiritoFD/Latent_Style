@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge

python tools\experiments\run_t01_inference_frontier_sweep.py 1> logs\t01_inference_comprehensive_sweep_ssh.out.log 2> logs\t01_inference_comprehensive_sweep_ssh.err.log

endlocal
