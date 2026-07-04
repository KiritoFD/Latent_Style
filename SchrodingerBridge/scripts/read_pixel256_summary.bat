@echo off
set PYTHON=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PYTHON%" set PYTHON=C:\Program Files\Python312\python.exe
"%PYTHON%" -c "import json; s=json.load(open(r'C:\Users\Administrator\exp\pixel256_sfm\pixel256_b2_e10\full_eval\epoch_0003\summary.json')); m=s.get('metrics',{}); mt=s.get('metrics_style_transfer',{}); mi=s.get('metrics_identity',{}); print('clip_style_all=',m.get('clip_style')); print('clip_style_transfer=',mt.get('clip_style')); print('clip_style_identity=',mi.get('clip_style')); print('lpips_all=',m.get('content_lpips')); print('lpips_transfer=',mt.get('content_lpips')); print('lpips_identity=',mi.get('content_lpips')); print('clip_t_all=',m.get('clip_t')); print('count=',m.get('count'))"
echo === DONE ===
