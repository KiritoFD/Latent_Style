@echo off
set PYTHON=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PYTHON%" set PYTHON=C:\Program Files\Python312\python.exe
"%PYTHON%" -c "import json; s=json.load(open(r'C:\Users\Administrator\exp\pixel256_sfm\pixel256_b2_e10\full_eval\epoch_0003\summary.json')); print('TOP_KEYS=', list(s.keys())); [print(k, '=', s[k]) for k in s if not isinstance(s[k], (dict, list))]"
echo === DONE ===
