"""Patch StyleShot ip_adapter.py to skip ALL .to(device) calls that break CPU offload."""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Patch line 1: self.pipe = sd_pipe.to(self.device)
old1 = "self.pipe = sd_pipe.to(self.device)"
new1 = "# PATCHED: skip .to(device) to allow CPU offload\n        self.pipe = sd_pipe  # .to(self.device)"

if old1 in content:
    content = content.replace(old1, new1)
    print(f"Patched: {old1} -> kept on CPU")
else:
    print(f"Line 1 already patched or not found")

# Also ensure line 2 stays patched
old2 = "self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
if old2 in content:
    content = content.replace(old2, "# PATCHED: skip float32 conversion\n        # self.pipe = self.pipe.to(self.device, dtype=torch.float32)")
    print(f"Patched: line 2 also commented out")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Done.")
