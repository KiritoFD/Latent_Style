import os, subprocess
d = r'I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src\step_020000\images'
if os.path.exists(d):
    print('imgs:', len(os.listdir(d)))
else:
    print('dir missing')
# check python process
r = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], capture_output=True, text=True)
print(r.stdout[-200:] if r.stdout else 'no tasklist')
