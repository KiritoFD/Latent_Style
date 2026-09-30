import subprocess, sys, os
os.environ["PYTHONPATH"] = r"I:\Github\Latent_Style\SchrodingerBridge\src"
config = sys.argv[1] if len(sys.argv) > 1 else r"I:\Github\Latent_Style\SchrodingerBridge\configs\exp_probe_tasm_ft6.json"
cmd = [r"C:\Program Files\Python312\python.exe", "-u", r"I:\Github\Latent_Style\SchrodingerBridge\src\run.py", "--config", config]
subprocess.run(cmd, check=True)