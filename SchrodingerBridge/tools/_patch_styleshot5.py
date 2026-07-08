"""Patch StyleShot ip_adapter.py for fp16 full-GPU loading:
1. Restore pipe.to(device) but NOT pipe.to(device, dtype=torch.float32) - let pipe keep its load-time dtype (fp16)
2. Change all dtype to float16 to match fp16 UNet
3. Keep style_aware_encoder in fp32 (separate module, outputs converted to fp16)
"""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Restore pipe.to(device) - remove the PATCHED comment
content = content.replace(
    "# PATCHED: skip .to(device) to allow CPU offload\n        self.pipe = sd_pipe  # .to(self.device)",
    "self.pipe = sd_pipe.to(self.device)"
)

# 2. Keep pipe.to(device, dtype=torch.float32) COMMENTED OUT - don't force fp32
# (already commented from patch2)

# 3. Change all float32 -> float16 (except style_aware_encoder which stays fp32)
# style_image_proj_modules: fp32 -> fp16 (output goes into fp16 UNet)
content = content.replace(
    "style_image_proj_modules.to(self.device, dtype=torch.float32)",
    "style_image_proj_modules.to(self.device, dtype=torch.float16)"
)

# ImageProjModel .to() calls: fp32 -> fp16
content = content.replace(
    ").to(self.device, dtype=torch.float32)",
    ").to(self.device, dtype=torch.float16)"
)

# clip_image_embeds: fp32 -> fp16
content = content.replace(
    "clip_image_embeds.to(self.device, dtype=torch.float32)",
    "clip_image_embeds.to(self.device, dtype=torch.float16)"
)

# clip_image input: fp32 -> fp16
content = content.replace(
    "clip_image.to(self.device, dtype=torch.float32)",
    "clip_image.to(self.device, dtype=torch.float16)"
)

# IPAttnProcessor: fp32 -> fp16
content = content.replace(
    ".to(device, dtype=torch.float32)",
    ".to(device, dtype=torch.float16)"
)

# style_embeds output: fp32 -> fp16 (input to fp16 UNet)
content = content.replace(
    "style_embeds = self.style_aware_encoder(style_image).to(self.device, dtype=torch.float32)",
    "style_embeds = self.style_aware_encoder(style_image).to(self.device, dtype=torch.float16)"
)

# style patch returns: fp32 -> fp16
content = content.replace(
    "high_style_patch.to(device, dtype=torch.float32), middle_style_patch.to(device, dtype=torch.float32), low_style_patch.to(device, dtype=torch.float32)",
    "high_style_patch.to(device, dtype=torch.float16), middle_style_patch.to(device, dtype=torch.float16), low_style_patch.to(device, dtype=torch.float16)"
)

# Keep style_aware_encoder itself as fp32 (separate module)
# Line: self.style_aware_encoder = Style_Aware_Encoder(...).to(self.device, dtype=torch.float32)
# This stays fp32 - its output is converted to fp16 above

# Remove old patch comments
content = content.replace("# PATCHED: skip .to(device) to allow CPU offload for VRAM saving\n        ", "")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
count_fp16 = content.count("float16")
count_fp32 = content.count("float32")
print(f"float16 refs: {count_fp16}")
print(f"float32 refs: {count_fp32}")

for i, line in enumerate(content.split('\n'), 1):
    if '.to(' in line and 'device' in line:
        print(f"  L{i}: {line.strip()}")
