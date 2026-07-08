tasklist | findstr /I python
echo ---
tasklist | findstr /I cmd.exe
echo ---
del /F C:\Users\Administrator\logs\stylealigned_run.log 2>&1
echo ---
C:\Users\Administrator\miniconda3\python.exe C:\Users\Administrator\_run_stylealigned_remote.py > C:\Users\Administrator\logs\sa_run3.log 2>&1
