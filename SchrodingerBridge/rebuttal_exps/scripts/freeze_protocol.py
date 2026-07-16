"""Freeze protocol: compute SHA-256 hashes of all artifacts for reproducibility.

Following reviewer_audit_and_required_experiments.md Section 3 (P0 protocol freeze).
"""
import hashlib, json, os, sys
from pathlib import Path
from datetime import datetime

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

OUTPUT_DIR = WEAVE_ROOT / "experiments" / "rebuttal_20260716"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_commit():
    import subprocess
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(WEAVE_ROOT), capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def main():
    print("=" * 60)
    print("Protocol Freeze: Computing artifact hashes")
    print("=" * 60)

    commit = git_commit()
    print(f"Git commit: {commit}")

    artifacts = {
        "git_commit": commit,
        "timestamp": datetime.now().isoformat(),
        "files": {},
        "configs": {},
        "checkpoints": {},
    }

    # 1. Core config files
    core_files = [
        "config.json",
        "inference.json",
        "utils/run_evaluation.py",
        "utils/compute_dino_metrics.py",
        "utils/artfid_metric.py",
        "trainer.py",
        "flow.py",
        "model.py",
        "wavelet.py",
        "internal_dynamics.py",
        "config_schema.py",
    ]
    print("\n--- Core files ---")
    for rel in core_files:
        p = WEAVE_ROOT / rel
        if p.exists():
            h = sha256_file(p)
            artifacts["files"][rel] = {"sha256": h, "size": p.stat().st_size}
            print(f"  {rel}: {h[:16]}... ({p.stat().st_size} bytes)")

    # 2. Rebuttal sweep configs
    print("\n--- Rebuttal configs ---")
    config_dir = WEAVE_ROOT / "experiments" / "rebuttal"
    if config_dir.exists():
        for cf in sorted(config_dir.glob("*.json")):
            rel = str(cf.relative_to(WEAVE_ROOT)).replace("\\", "/")
            h = sha256_file(cf)
            content = cf.read_text(encoding="utf-8")
            artifacts["configs"][rel] = {
                "sha256": h,
                "content_hash": sha256_str(content),
            }
            print(f"  {rel}: {h[:16]}...")

    # 3. Production checkpoint
    print("\n--- Production checkpoint ---")
    prod_ckpt = WEAVE_ROOT / "runs" / "submission" / "hf_oriented_internal_early_stop" / "epoch_0004.pt"
    if prod_ckpt.exists():
        h = sha256_file(prod_ckpt)
        artifacts["checkpoints"]["production_epoch_0004"] = {
            "path": str(prod_ckpt.relative_to(WEAVE_ROOT)).replace("\\", "/"),
            "sha256": h,
            "size_mb": round(prod_ckpt.stat().st_size / 1024 / 1024, 1),
        }
        print(f"  epoch_0004.pt: {h[:16]}... ({prod_ckpt.stat().st_size / 1024 / 1024:.1f} MB)")

    # 4. Robustness checkpoints (seed7/123)
    print("\n--- Robustness checkpoints ---")
    for run_name in ["early_stop_seed7", "early_stop_seed123"]:
        run_dir = WEAVE_ROOT / "runs" / "submission" / "robustness" / run_name
        if not run_dir.exists():
            continue
        for ckpt in sorted(run_dir.glob("epoch_*.pt")):
            rel = str(ckpt.relative_to(WEAVE_ROOT)).replace("\\", "/")
            h = sha256_file(ckpt)
            key = f"{run_name}_{ckpt.stem}"
            artifacts["checkpoints"][key] = {
                "path": rel,
                "sha256": h,
                "size_mb": round(ckpt.stat().st_size / 1024 / 1024, 1),
            }
            print(f"  {run_name}/{ckpt.name}: {h[:16]}...")

    # 5. Config provenance diff
    print("\n--- Config provenance diff ---")
    base_config = (WEAVE_ROOT / "experiments" / "rebuttal" / "lambda_ll_0p1.json")
    if base_config.exists():
        import json as j
        base = j.loads(base_config.read_text(encoding="utf-8"))
        for cf in sorted((WEAVE_ROOT / "experiments" / "rebuttal").glob("*.json")):
            if cf.name == base_config.name:
                continue
            cur = j.loads(cf.read_text(encoding="utf-8"))
            diffs = []
            for key in ["bridge", "training", "checkpoint"]:
                if key in base and key in cur:
                    for subkey in set(list(base.get(key, {}).keys()) + list(cur.get(key, {}).keys())):
                        bv = base.get(key, {}).get(subkey)
                        cv = cur.get(key, {}).get(subkey)
                        if bv != cv:
                            diffs.append(f"{key}.{subkey}: {bv} -> {cv}")
            if diffs:
                print(f"  {cf.name}: {', '.join(diffs)}")
            else:
                print(f"  {cf.name}: identical (unexpected)")

    # 6. Dataset manifest
    print("\n--- Dataset manifest ---")
    test_dir = WEAVE_ROOT / "data" / "test"
    if test_dir.exists():
        families = {}
        total = 0
        for fam_dir in sorted(test_dir.iterdir()):
            if fam_dir.is_dir():
                count = len([f for f in fam_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])
                families[fam_dir.name] = count
                total += count
        artifacts["dataset"] = {
            "test_dir": str(test_dir.relative_to(WEAVE_ROOT)).replace("\\", "/"),
            "families": families,
            "total_images": total,
        }
        print(f"  Families: {families}")
        print(f"  Total: {total} images")

    # Save
    out_path = OUTPUT_DIR / "protocol_manifest.json"
    out_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nProtocol manifest saved: {out_path}")
    print(f"FREEZE_EXIT=0")


if __name__ == "__main__":
    main()
