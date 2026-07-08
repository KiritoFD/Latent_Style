"""Patch StyleShot ip_adapter.py: restore pipe.to(device) for full-GPU fp16 loading.
No CPU offload - all fp16 on GPU. pipe.to(device) moves pipe to GPU,
then StyleShot adds its own modules (style_aware_encoder, ImageProjModels) also on GPU.
"""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Restore pipe.to(device) - we want full GPU loading with fp16
content = content.replace(
    "self.pipe = sd_pipe  # .to(self.device) -- let model_cpu_offload manage pipe placement",
    "self.pipe = sd_pipe.to(self.device)"
)

# Keep pipe.to(device, dtype=torch.float32) COMMENTED OUT
if "self.pipe = self.pipe.to(self.device, dtype=torch.float32)" in content:
    content = content.replace(
        "self.pipe = self.pipe.to(self.device, dtype=torch.float32)",
        "# self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
    )

# Also fix double-commented line from previous patches
content = content.replace(
    "# # self.pipe = self.pipe.to(self.device, dtype=torch.float32)",
    "# self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

for i, line in enumerate(content.split('\n'), 1):
    if '.to(' in line and 'device' in line:
        print(f"  L{i}: {line.strip()}")
