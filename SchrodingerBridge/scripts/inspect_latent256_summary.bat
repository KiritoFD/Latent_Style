@echo off
set PYTHON=C:\Program Files\Python312\python.exe
"%PYTHON%" -c "import json; s=json.load(open(r'C:\Users\Administrator\exp\latent256_sfm\latent256_b16_e10\full_eval\epoch_0001\summary.json')); a=s.get('analysis',{}); print('ANALYSIS_KEYS=', list(a.keys())); [print(k, '=', a[k]) for k in a if isinstance(a[k], dict)]; print('---'); [print(k, '=', a[k]) for k in a if not isinstance(a[k], (dict, list))]"
echo === DONE ===
