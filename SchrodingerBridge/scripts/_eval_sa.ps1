# Launch StyleAligned evaluation on D5, P2A, R5
# StyleAligned inference is done, GPU is free for evaluation
& "C:\Program Files\Python312\python.exe" "C:\Users\Administrator\_eval_remote_baselines.py" --method stylealigned --datasets D5,P2A,R5 --batch_size 16 2>&1 | Out-File -FilePath "C:\Users\Administrator\logs\eval_stylealigned.log" -Encoding utf8
