import argparse
import copy
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def deep_set(d: Dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


@dataclass
class Exp:
    name: str
    group: str
    overrides: List[Tuple[str, Any]]
    epochs: int = 100  # 默认全部跑 200 epoch


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_cmd(cmd: List[str], log_path: Path, cwd: Optional[Path] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        return p.wait()


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def make_experiments(base_cfg: Dict[str, Any]) -> List[Exp]:
    # === 基础假设 ===
    # 1. model.py 已经加了 Tanh Soft-Clamp (解决了白化)。
    # 2. losses.py 已经加了 Patch去均值 + FP32 (解决了纹理学习)。
    # 3. 基础配置：Moment=10 (保色), SWD=100 (保纹理), Id=10 (松绑)
    
    exps: List[Exp] = []

    # =========================================================
    # Group A: 黄金基准 (The Golden Balance)
    # 目标：验证“防白化+去均值”后的标准效果。
    # 预期：颜色准（Moment），纹理有（SWD），不泛白（Tanh）。
    # =========================================================
    exps.append(Exp("A01_Balanced_Fix", "A", [
        ("loss.w_color_moment", 10.0),
        ("loss.w_swd", 100.0),
        ("loss.w_identity", 3.0),
        ("loss.swd_num_projections", 128),
        ("loss.swd_patch_sizes", [3, 5])
    ]))

    # =========================================================
    # Group B: 极限画质 (High Fidelity)
    # 目标：榨干 3060 算力。512 个投影能提供极其细腻的分布梯度。
    # 预期：如果你觉得 A01 纹理有点噪或不连贯，B01 应该会更顺滑、更像“画”出来的。
    # =========================================================
    exps.append(Exp("B01_HiFi_512Proj", "B", [
        ("loss.w_color_moment", 10.0),
        ("loss.w_swd", 100.0),
        ("loss.w_identity", 3.0),
        ("loss.swd_num_projections", 512), # 投影数翻4倍
        ("loss.swd_patch_sizes", [3, 5])
    ]))

    # =========================================================
    # Group C: 纯微观纹理 (Micro Texture)
    # 目标：解决“结构扭曲”。去掉 Patch=5，只看 3x3 (局部)。
    # 预期：这组的内容保持度应该是最好的。适合那种“只换笔触、不动形状”的风格。
    # =========================================================
    exps.append(Exp("C01_Patch3_Only", "C", [
        ("loss.w_color_moment", 10.0),
        ("loss.w_swd", 100.0),
        ("loss.w_identity", 3.0),
        ("loss.swd_num_projections", 128),
        ("loss.swd_patch_sizes", [3]) # 去掉 5
    ]))

    # =========================================================
    # Group D: 暴力风格化 (Aggressive Style)
    # 目标：解决“风格不明显”。如果 A01 还是太像照片，这组负责把它拉弯。
    # 手段：降低 Identity 约束，SWD 权重翻倍。
    # =========================================================
    exps.append(Exp("D01_Aggressive_Style", "D", [
        ("loss.w_color_moment", 5.0),
        ("loss.w_swd", 200.0),       # 强度翻倍
        ("loss.w_identity", 1.0),    # 约束减半
        ("loss.swd_num_projections", 128),
        ("loss.swd_patch_sizes", [3, 5])
    ]))

    # =========================================================
    # Group E: Semigroup 消融 (固定其余损失，仅扫描 w_semigroup)
    # 目标：验证 semigroup 正则对风格稳定性/结构保持的贡献。
    # =========================================================
    exps.append(Exp("E01_Semigroup_Off", "E", [
        ("loss.w_color_moment", 10.0),
        ("loss.w_swd", 100.0),
        ("loss.w_identity", 3.0),
        ("loss.swd_num_projections", 128),
        ("loss.swd_patch_sizes", [3, 5]),
        ("loss.w_semigroup", 0.0),
        ("loss.semigroup_every_n_steps", 1),
    ]))
    
    exps.append(Exp("E03_Semigroup_0p5", "E", [
        ("loss.w_color_moment", 10.0),
        ("loss.w_swd", 100.0),
        ("loss.w_identity", 3.0),
        ("loss.swd_num_projections", 128),
        ("loss.swd_patch_sizes", [3, 5]),
        ("loss.w_semigroup", 0.5),
        ("loss.semigroup_every_n_steps", 1),
    ]))
    exps.append(Exp("E04_Semigroup_2p0", "E", [
        ("loss.w_color_moment", 10.0),
        ("loss.w_swd", 100.0),
        ("loss.w_identity", 3.0),
        ("loss.swd_num_projections", 128),
        ("loss.swd_patch_sizes", [3, 5]),
        ("loss.w_semigroup", 2.0),
        ("loss.semigroup_every_n_steps", 1),
    ]))

    return exps


def apply_overrides(base_cfg: Dict[str, Any], overrides: List[Tuple[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides:
        deep_set(cfg, k, v)
    return cfg


def apply_infra_profile(cfg: Dict[str, Any], profile: str) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    p = str(profile).strip().lower()
    
    if "rtx30" in p:
        deep_set(out, "training.use_amp", True)
        deep_set(out, "training.amp_dtype", "fp16") 
        deep_set(out, "training.use_grad_scaler", True)
        deep_set(out, "training.use_compile", False)
        deep_set(out, "training.batch_size", 192)
        deep_set(out, "training.allow_tf32", True)
        return out

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, default="config.json")
    ap.add_argument("--run_py", type=str, default="run.py")
    ap.add_argument("--out_dir", type=str, default="../sdxl-ablation-swd-semigroup-100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groups", type=str, default="A,B,C,D,E") # 默认全跑
    ap.add_argument("--infra_profile", type=str, default="rtx3060")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_path = (script_dir / args.base_config).resolve()
    run_py = (script_dir / args.run_py).resolve()
    out_root = (script_dir / args.out_dir).resolve()

    if not base_path.exists():
        print(f"Error: Base config not found: {base_path}")
        return

    base_cfg = load_json(base_path)
    
    # 强制关闭 Projector (因为我们现在直接算 Latent)
    deep_set(base_cfg, "model.loss_projector_use", False)

    exps = make_experiments(base_cfg)
    
    group_set = {g.strip().upper() for g in args.groups.split(",") if g.strip()}
    exps = [e for e in exps if e.group.upper() in group_set]

    if not exps:
        print("No experiments matched.")
        return

    python_bin = sys.executable
    summary = []

    for i, exp in enumerate(exps, 1):
        run_name = safe_name(exp.name)
        run_dir = out_root / run_name
        cfg = apply_overrides(base_cfg, exp.overrides)
        cfg = apply_infra_profile(cfg, args.infra_profile)
        
        deep_set(cfg, "training.seed", args.seed)
        deep_set(cfg, "training.num_epochs", int(exp.epochs)) # 强制 200
        
        deep_set(cfg, "checkpoint.save_dir", str(run_dir / "checkpoints"))
        
        cfg_path = run_dir / "config.json"
        save_json(cfg_path, cfg)
        
        print(f"[{i}/{len(exps)}] {exp.name} (Group {exp.group}) | Epochs: {exp.epochs}")
        
        if args.dry_run:
            summary.append((exp.name, 0))
            continue

        log_path = run_dir / "train.log"
        cmd = [python_bin, str(run_py), "--config", str(cfg_path)]
        
        env = os.environ.copy()
        code = run_cmd(cmd, log_path=log_path, cwd=script_dir)
        
        if code != 0:
            print(f"  FAILED code={code}")
            if args.stop_on_error: break
        
        summary.append((exp.name, code))

    save_json(out_root / "summary.json", {"experiments": summary})
    print("Done.")

if __name__ == "__main__":
    main()
