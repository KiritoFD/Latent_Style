@echo off
scp -P 2222 "g:\GitHub\Latent_Style\SchrodingerBridge\extract_metrics_remote.py" administrator@100.115.18.62:~/extract_metrics_remote.py
ssh -p 2222 administrator@100.115.18.62 "cd /home/xy/Latent_Style/SchrodingerBridge && python3 extract_metrics_remote.py > ~/metrics.txt 2>&1"
scp -P 2222 administrator@100.115.18.62:~/metrics.txt "g:\GitHub\Latent_Style\SchrodingerBridge\metrics.txt"
echo Done!
