"""COMPREHENSIVE patch: ALL dtype -> fp32 for fp32+sequential_offload compatibility.
Also: pipe.to(device) is skipped (sequential offload manages it).
"""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Skip pipe.to(device) - sequential offload manages it
content = content.replace(
    "self.pipe = sd_pipe.to(self.device)",
    "self.pipe = sd_pipe  # .to(self.device) -- sequential CPU offload manages pipe"
)
content = content.replace(
    "self.pipe = sd_pipe  # .to(self.device) -- sequential CPU offload manages pipe placement",
    "self.pipe = sd_pipe  # .to(self.device) -- sequential CPU offload manages pipe"
)

# 2. Ensure pipe.to(device, fp32) is commented
if "\nself.pipe = self.pipe.to(self.device, dtype=torch.float32)" in content:
    content = content.replace(
        "\nself.pipe = self.pipe.to(self.device, dtype=torch.float32)",
        "\n# self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
    )
if "\n# self.pipe = self.pipe.to(self.device, dtype=torch.float32)" not in content:
    # Try other formats
    content = content.replace(
        "self.pipe = self.pipe.to(self.device, dtype=torch.float32)",
        "# self.pipe = self.pipe.to(self.device, dtype=torch.float32)"
    )

# 3. ALL .to(device, dtype=torch.float16) -> fp32
content = content.replace(".to(self.device, dtype=torch.float16)", ".to(self.device, dtype=torch.float32)")
content = content.replace(".to(device, dtype=torch.float16)", ".to(device, dtype=torch.float32)")

# 4. clip_image and clip_image_embeds -> fp32
content = content.replace("clip_image.to(self.device, dtype=torch.float32)", "clip_image.to(self.device, dtype=torch.float32)")  # already fp32
content = content.replace("clip_image_embeds.to(self.device, dtype=torch.float32)", "clip_image_embeds.to(self.device, dtype=torch.float32)")  # already fp32

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
