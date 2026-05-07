@echo off
setlocal
cd /d "%~dp0"

echo Running OMF ablation matrix...
echo.

echo [01_gold_no_skip] train
python run.py --config "experiments\omf_ablation_matrix\configs\01_gold_no_skip.json"
if errorlevel 1 goto :error
echo [01_gold_no_skip] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\01_gold_no_skip\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\01_gold_no_skip" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [02_gold_norm_skip] train
python run.py --config "experiments\omf_ablation_matrix\configs\02_gold_norm_skip.json"
if errorlevel 1 goto :error
echo [02_gold_norm_skip] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\02_gold_norm_skip\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\02_gold_norm_skip" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [03_no_repulsive] train
python run.py --config "experiments\omf_ablation_matrix\configs\03_no_repulsive.json"
if errorlevel 1 goto :error
echo [03_no_repulsive] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\03_no_repulsive\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\03_no_repulsive" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [04_repulsive_05] train
python run.py --config "experiments\omf_ablation_matrix\configs\04_repulsive_05.json"
if errorlevel 1 goto :error
echo [04_repulsive_05] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\04_repulsive_05\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\04_repulsive_05" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [05_color_05] train
python run.py --config "experiments\omf_ablation_matrix\configs\05_color_05.json"
if errorlevel 1 goto :error
echo [05_color_05] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\05_color_05\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\05_color_05" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [06_color_12] train
python run.py --config "experiments\omf_ablation_matrix\configs\06_color_12.json"
if errorlevel 1 goto :error
echo [06_color_12] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\06_color_12\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\06_color_12" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [07_swd_10] train
python run.py --config "experiments\omf_ablation_matrix\configs\07_swd_10.json"
if errorlevel 1 goto :error
echo [07_swd_10] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\07_swd_10\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\07_swd_10" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [08_swd_30] train
python run.py --config "experiments\omf_ablation_matrix\configs\08_swd_30.json"
if errorlevel 1 goto :error
echo [08_swd_30] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\08_swd_30\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\08_swd_30" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [09_kinetic_05] train
python run.py --config "experiments\omf_ablation_matrix\configs\09_kinetic_05.json"
if errorlevel 1 goto :error
echo [09_kinetic_05] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\09_kinetic_05\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\09_kinetic_05" --batch_size 4
if errorlevel 1 goto :error
echo.
echo [10_kinetic_20_norm_skip] train
python run.py --config "experiments\omf_ablation_matrix\configs\10_kinetic_20_norm_skip.json"
if errorlevel 1 goto :error
echo [10_kinetic_20_norm_skip] eval
python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\10_kinetic_20_norm_skip\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\10_kinetic_20_norm_skip" --batch_size 4
if errorlevel 1 goto :error
echo.
echo All OMF ablation runs finished.
exit /b 0

:error
echo.
echo OMF ablation matrix aborted due to an error.
exit /b 1
