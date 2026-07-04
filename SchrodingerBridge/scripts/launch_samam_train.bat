@echo off
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "wsl -- bash -lc 'nohup bash /mnt/c/Users/Administrator/run_samam_latent_train_only.sh > /mnt/i/exp_samam_latent_train_nohup.log 2>&1 &'"
echo launched
