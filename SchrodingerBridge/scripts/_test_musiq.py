"""Test MUSIQ metric creation with correct name."""
import os
os.environ['TORCH_HOME'] = r'C:\Users\Administrator\.cache\torch'
import pyiqa

# List all available metrics
all_metrics = pyiqa.list_models() if hasattr(pyiqa, 'list_models') else []
print(f"Total metrics available: {len(all_metrics)}")
musiq_metrics = [m for m in all_metrics if 'musiq' in m.lower()]
print(f"MUSIQ metrics: {musiq_metrics}")

# Try creating musiq
for name in ['musiq', 'musiq-koniq', 'musiq_koniq', 'musiq-spaq']:
    try:
        m = pyiqa.create_metric(name, device='cpu')
        print(f"SUCCESS: {name} -> {type(m).__name__}")
        break
    except Exception as e:
        print(f"FAIL: {name} -> {e}")
