"""Patch StyleShot ip_adapter.py for fp16 + model_cpu_offload:
- Skip pipe.to(device) - model_cpu_offload manages pipe modules
- All StyleShot extra modules in fp16
- style_aware_encoder stays fp32 (separate, output cast to fp16)
"""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Skip pipe.to(device) - model_cpu_offload manages it
content = content.replace(
    "self.pipe = sd_pipe.to(self.device)",
    "self.pipe = sd_pipe  # .to(self.device) -- model_cpu_offload manages pipe"
)
content = content.replace(
    "self.pipe = sd_pipe  # .to(self.device) -- sequential CPU offload manages pipe",
    "self.pipe = sd_pipe  # .to(self.device) -- model_cpu_offload manages pipe"
)
content = content.replace(
    "self.pipe = sd_pipe  # .to(self.device) -- sequential CPU offload manages pipe placement",
    "self.pipe = sd_pipe  # .to(self.device) -- model_cpu_offload manages pipe"
)

# 2. Keep pipe.to(device, dtype=torch.float32) commented
if "\nself.pipe = self.pipe.to(self.device, dtype=torch.float32)" in content:
    content = content.replace(
        "\nself.pipe = self.pipe.to(self.device, dtype=torch.float32)",
        "\n# self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
    )

# 3. Change all fp32 -> fp16 for StyleShot modules (to match fp16 pipe)
content = content.replace(".to(self.device, dtype=torch.float32)", ".to(self.device, dtype=torch.float16)")
content = content.replace(".to(device, dtype=torch.float32)", ".to(device, dtype=torch.float16)")

# 4. style_aware_encoder: keep fp32 internally (it's a separate model),
#    but cast output to fp16 for fp16 UNet
#    Already done: style_embeds = ... .to(self.device, dtype=torch.float16)

# 5. Also fix any double-commented lines
content = content.replace("# # self.pipe = self.pipe.to(self.device, dtype=torch.float32)",
                          "# self.pipe = self.pipe.to(self.device, dtype=torch.float32)")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
n_fp16 = content.count("float16")
n_fp32 = content.count("float32")
print(f"float16 refs: {n_fp16}")
print(f"float32 refs: {n_fp32}")

for i, line in enumerate(content.split('\n'), 1):
    if '.to(' in line and 'device' in line:
        print(f"  L{i}: {line.strip()}")
