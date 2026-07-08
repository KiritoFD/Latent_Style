"""Patch StyleShot ip_adapter.py: fix dtype mismatch (fp16→fp32 in set_ip_adapter, and ensure get_image_embeds outputs fp32)."""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

patches = 0
for i, line in enumerate(lines):
    # Patch 1: line 536 area - .to(device, dtype=torch.float16) in IPAttnProcessor creation
    if ".to(device, dtype=torch.float16)" in line and "IPAttnProcessor" not in line:
        # Could be the .to() on the IPAttnProcessor object
        old = ".to(device, dtype=torch.float16)"
        new = ".to(device, dtype=torch.float32)"
        lines[i] = line.replace(old, new)
        print(f"Line {i+1}: patched fp16 -> fp32")
        patches += 1
    elif ".to(device, dtype=torch.float16)" in line:
        old = ".to(device, dtype=torch.float16)"
        new = ".to(device, dtype=torch.float32)"
        lines[i] = line.replace(old, new)
        print(f"Line {i+1}: patched fp16 -> fp32 (IPAttnProcessor)")
        patches += 1

    # Patch 2: ensure style_image_proj_modules uses fp32
    if "style_image_proj_modules.to(self.device, dtype=torch.float32)" in line:
        print(f"Line {i+1}: already fp32, OK")
    elif "style_image_proj_modules.to(self.device)" in line and "float32" not in line and "float16" not in line:
        lines[i] = line.replace("style_image_proj_modules.to(self.device)",
                                "style_image_proj_modules.to(self.device, dtype=torch.float32)")
        print(f"Line {i+1}: patched style_image_proj_modules to fp32")
        patches += 1

    # Patch 3: get_image_embeds output dtype (line 548 area)
    if "style_embeds = self.style_aware_encoder(style_image).to(self.device, dtype=torch.float32)" in line:
        print(f"Line {i+1}: get_image_embeds already fp32, OK")

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\nTotal patches applied: {patches}")

# Verify the changes
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
import re
# Show all lines with .to(device
for i, line in enumerate(content.split('\n'), 1):
    if '.to(' in line and 'device' in line:
        print(f"  L{i}: {line.strip()}")
