@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge

python tools\experiments\run_diffeomorphic_tangent_sweep.py --force-train 1> logs\diffeomorphic_tangent_sweep_ssh.out.log 2> logs\diffeomorphic_tangent_sweep_ssh.err.log

endlocal
