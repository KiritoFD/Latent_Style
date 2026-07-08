"""Patch StyleShot ip_adapter.py: change ALL .to(device, dtype=torch.float16) to .to(device, dtype=torch.float32)
for consistency with fp32 UNet when using CPU offload. Also ensure all internal modules use fp32."""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all float16 -> float32 in .to() calls (except the ones we already patched)
content = content.replace(".to(self.device, dtype=torch.float16)", ".to(self.device, dtype=torch.float32)")
content = content.replace(".to(device, dtype=torch.float16)", ".to(device, dtype=torch.float32)")
# Also fix clip_image.to(self.device, dtype=torch.float16) -> float32
content = content.replace("clip_image.to(self.device, dtype=torch.float16)", "clip_image.to(self.device, dtype=torch.float32)")
# Fix clip_image_embeds.to(self.device, dtype=torch.float16) -> float32
content = content.replace("clip_image_embeds.to(self.device, dtype=torch.float16)", "clip_image_embeds.to(self.device, dtype=torch.float32)")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify: count remaining float16 references
count_fp16 = content.count("float16")
count_fp32 = content.count("float32")
print(f"Remaining float16 refs: {count_fp16}")
print(f"float32 refs: {count_fp32}")

# Show all .to(device lines
for i, line in enumerate(content.split('\n'), 1):
    if '.to(' in line and 'device' in line:
        print(f"  L{i}: {line.strip()}")
