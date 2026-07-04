"""Simple test: can we run Python and write to a file?"""
import sys, time
log_path = r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\test_python.log'
with open(log_path, 'w') as f:
    f.write(f'Python {sys.version}\n')
    f.write(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'PID: {__import__("os").getpid()}\n')
    f.write('Test successful!\n')
print('Test script completed')
