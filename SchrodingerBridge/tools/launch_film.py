import subprocess, os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="620_film_gate03_5ep")
args = parser.parse_args()

exp_name = args.name
exp_dir = f"/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/{exp_name}"
src_dir = "/mnt/i/Github/Latent_Style/SchrodingerBridge/src"

os.chdir(src_dir)
env = os.environ.copy()
env["PYTHONPATH"] = src_dir

log = open(f"{exp_dir}/train.log", "w")
p = subprocess.Popen(
    ["python3", "run.py", "--config", f"{exp_dir}/config.json"],
    stdout=log, stderr=subprocess.STDOUT, env=env,
    start_new_session=True,
    close_fds=True,
)
print(f"Launched {exp_name} PID={p.pid}")