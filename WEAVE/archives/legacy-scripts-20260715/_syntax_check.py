"""Syntax check for modified files."""
import ast, sys
files = [
    "src/spectral_bridge620.py",
    "src/spectral_losses620.py",
]
ok = True
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"  {f}: OK")
    except SyntaxError as e:
        print(f"  {f}: SYNTAX_ERROR {e}")
        ok = False
sys.exit(0 if ok else 1)
