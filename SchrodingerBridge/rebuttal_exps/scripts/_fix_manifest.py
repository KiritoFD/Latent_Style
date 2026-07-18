"""Fix manifest data_root to match str(Path(data_root)) on Windows."""
import json
from pathlib import Path

manifest_path = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\weave_gen\data\train_sdxl\.latent_cache\manifest.json")
d = json.loads(manifest_path.read_text(encoding="utf-8"))
# Match what str(Path("data/train_sdxl")) returns on Windows
correct = str(Path("data/train_sdxl"))
d["data_root"] = correct
manifest_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"data_root set to: {repr(d['data_root'])}")
