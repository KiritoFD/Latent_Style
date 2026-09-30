@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
python src\run.py --config configs\exp_brk_ac_fft_loss.json > logs\brk_ac_fft_loss_train.log 2>&1
echo TRAIN_DONE_EXITCODE=%errorlevel% >> logs\brk_ac_fft_loss_train.log
