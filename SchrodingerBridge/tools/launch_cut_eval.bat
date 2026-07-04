@echo off
powershell -Command "Start-Process -FilePath 'I:\GitHub\Latent_Style\SchrodingerBridge\tools\run_cut_eval.bat' -WindowStyle Hidden -RedirectStandardOutput 'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\cut_eval.log' -RedirectStandardError 'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\cut_eval_err.log'"
echo ==LAUNCH_DONE==
