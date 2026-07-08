"""Patch StyleShot ip_adapter.py for model_cpu_offload:
- pipe.to(device) -> do NOT move pipe to device (enable_model_cpu_offload manages it)
- All modules fp16 (already done by patch5)
- Keep style_aware_encoder and ImageProjModels on GPU (not managed by pipe offload)
"""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# The key change: StyleShot.__init__ calls self.pipe = sd_pipe.to(self.device)
# But with enable_model_cpu_offload, the pipe manages its own device placement.
# We need to NOT call pipe.to(device) - let model_cpu_offload handle it.
content = content.replace(
    "self.pipe = sd_pipe.to(self.device)",
    "self.pipe = sd_pipe  # .to(self.device) -- let model_cpu_offload manage pipe placement"
)

# Ensure the second pipe.to is still commented out
if "self.pipe = self.pipe.to(self.device, dtype=torch.float32)" in content:
    content = content.replace(
        "self.pipe = self.pipe.to(self.device, dtype=torch.float32)",
        "# self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
    )

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
for i, line in enumerate(content.split('\n'), 1):
    if '.to(' in line and 'device' in line:
        print(f"  L{i}: {line.strip()}")
