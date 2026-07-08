"""Patch StyleShot ip_adapter.py for fp16 + sequential CPU offload:
- Do NOT call pipe.to(device) - let sequential CPU offload manage pipe
- Keep all extra modules (style_aware_encoder, ImageProjModels, IPAttnProcessor) on GPU in fp16
"""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove pipe.to(device) - sequential offload manages it
content = content.replace(
    "self.pipe = sd_pipe.to(self.device)",
    "self.pipe = sd_pipe  # .to(self.device) -- sequential CPU offload manages pipe placement"
)

# Keep pipe.to(device, dtype=torch.float32) commented
if "\nself.pipe = self.pipe.to(self.device, dtype=torch.float32)" in content:
    content = content.replace(
        "\nself.pipe = self.pipe.to(self.device, dtype=torch.float32)",
        "\n# self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
    )

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

for i, line in enumerate(content.split('\n'), 1):
    if 'pipe.to' in line and 'self.device' in line:
        print(f"  L{i}: {line.strip()}")
