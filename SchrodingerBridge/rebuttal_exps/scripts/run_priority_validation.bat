@echo off
setlocal
set ROOT=I:\Github\Latent_Style\WEAVE
set PY=C:\Program Files\Python312\python.exe
set LOG=%ROOT%\exp\rebuttal\logs\priority_validation.log
cd /d %ROOT%

echo ==== priority validation start %date% %time% ==== > "%LOG%"
"%PY%" -u rebuttal_exps\scripts\expB_reference_margin.py --n_iters 1000 >> "%LOG%" 2>&1
if errorlevel 1 goto :fail

"%PY%" -u rebuttal_exps\scripts\expD_inference_ablation.py >> "%LOG%" 2>&1
if errorlevel 1 goto :fail

for %%C in (canonical_seed42_nostop canonical_seed123_nostop rebuttal_D5_hh_head_seed7) do (
  echo ==== train %%C %date% %time% ==== >> "%LOG%"
  "%PY%" -u run.py --config rebuttal_exps\configs\%%C.json >> "%LOG%" 2>&1
  if errorlevel 1 goto :fail
)

"%PY%" -u rebuttal_exps\scripts\expA_per_epoch_eval.py --run_dir runs\submission\rebuttal_seed42_nostop --seed 42 --tag canonical42 >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
"%PY%" -u rebuttal_exps\scripts\expA_per_epoch_eval.py --run_dir runs\submission\rebuttal_seed123_nostop --seed 123 --tag canonical123 >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
"%PY%" -u rebuttal_exps\scripts\expA_per_epoch_eval.py --run_dir runs\submission\rebuttal_D5_hh_head_seed7 --seed 7 --tag D5HH >> "%LOG%" 2>&1
if errorlevel 1 goto :fail

echo ==== priority validation complete %date% %time% ==== >> "%LOG%"
exit /b 0

:fail
echo ==== priority validation failed %date% %time% errorlevel=%errorlevel% ==== >> "%LOG%"
exit /b 1
