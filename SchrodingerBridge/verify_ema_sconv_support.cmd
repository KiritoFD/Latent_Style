@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" -B -c "import ast,pathlib; files=['src/config_schema.py','src/lancet_backbone.py','src/lancet_runtime.py','tools/experiments/run_vae_backend_256_probe.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8')) for f in files]; print('remote ast ok')"
