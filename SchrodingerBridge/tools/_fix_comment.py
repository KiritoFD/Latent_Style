"""Fix double-commented line in ip_adapter.py"""
filepath = r"C:\Users\Administrator\StyleShot\ip_adapter\ip_adapter.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = content if False else f.read()

content = content.replace("# # self.pipe = self.pipe.to(self.device, dtype=torch.float32)",
                          "# self.pipe = self.pipe.to(self.device, dtype=torch.float32)")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# Check
for i, line in enumerate(content.split('\n'), 1):
    if 'pipe.to' in line and 'self.device' in line:
        print(f"  L{i}: {line.strip()}")
