"""Launch WCT VGG-19 eval as detached process on remote."""
import sys
import subprocess
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PYTHON = r'C:\Program Files\Python312\python.exe'
EVAL_SCRIPT = r'I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py'
EVAL_DIR = r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\wct_vgg19'
TEST_DIR = r'I:\wikiart_distinct5_samam_512_classview\test'
LOG_FILE = r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\wct_vgg19_eval.log'
ERR_FILE = r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\wct_vgg19_eval_err.log'

DETACHED_FLAGS = 0x00000008 | 0x00000200 | 0x08000000

cmd = [
    PYTHON, EVAL_SCRIPT, EVAL_DIR,
    '--reuse_generated',
    '--save_generated_images',
    '--style_subdirs', 'Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e',
    '--test_dir', TEST_DIR,
    '--eval_only_lpips_clip_style',
    '--clip_style_idt_baseline', '0.6399',
]

print(f"Launching WCT VGG-19 evaluation as detached process...")
print(f"CMD: {' '.join(cmd[:4])}...")

Path(LOG_FILE).write_text('')
Path(ERR_FILE).write_text('')

with open(LOG_FILE, 'w') as log_f, open(ERR_FILE, 'w') as err_f:
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=err_f,
        creationflags=DETACHED_FLAGS,
        close_fds=True,
        cwd=r'I:\GitHub\Latent_Style\SchrodingerBridge',
    )

print(f"Detached PID: {proc.pid}")

for i in range(6):
    time.sleep(5)
    ret = proc.poll()
    log_content = Path(LOG_FILE).read_text() if Path(LOG_FILE).exists() else ''
    err_content = Path(ERR_FILE).read_text() if Path(ERR_FILE).exists() else ''
    log_tail = log_content[-500:] if log_content else ''
    err_tail = err_content[-500:] if err_content else ''
    print(f"  [{(i+1)*5}s] alive={ret is None} log_tail={log_tail!r}")
    if err_tail:
        print(f"  [{(i+1)*5}s] ERR: {err_tail!r}")
    if ret is not None:
        print(f"  Process exited with code {ret}")
        break

print("==LAUNCH_BG_DONE==")
