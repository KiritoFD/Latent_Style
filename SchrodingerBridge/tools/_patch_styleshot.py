"""Patch StyleShot ip_adapter.py to skip .to(device) that breaks CPU offload."""
import sys

filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic line that forces pipe to GPU float32
old = "self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
new = "# PATCHED: skip .to(device) to allow CPU offload for VRAM saving\n        # self.pipe = self.pipe.to(self.device, dtype=torch.float32)"

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Patched: removed pipe.to(device) in {filepath}")
else:
    print(f"Already patched or line not found in {filepath}")
