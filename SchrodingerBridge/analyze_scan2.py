#!/usr/bin/env python3
"""Parse the scan output, produce a clean markdown report saved to file."""
import json, re
from pathlib import Path
from collections import defaultdict

INPUT = r"C:\Users\xy\AppData\Local\Temp\trae-agent-toolhost\jobs\job-a87fb28f31e242b9bd09c27853eef557\output.log"
OUT = r"g:\GitHub\Latent_Style\SchrodingerBridge\scan_report.md"

with open(INPUT, encoding="utf-8") as f:
    lines = f.readlines()
json_start = next(i for i, l in enumerate(lines) if l.strip().startswith("{"))
data = json.loads("".join(lines[json_start:]))

root_summaries = data["root_summaries"]
experiments = data["experiments"]

def parse_size(s):
    """Parse size string like '1.5G', '20M', '4.0K' to bytes."""
    if not isinstance(s, str): return 0
    m = re.match(r"^([\d.]+)([KMGTP]?)(\+?)$", s.strip())
    if not m: return 0
    num = float(m.group(1))
    unit = m.group(2)
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}.get(unit, 1)
    return num * mult

def classify_model(name, root):
    n = name.lower()
    if root == "final_works":
        if "cut" in n: return "CUT"
        if "samst" in n: return "SaMST"
        if "star-gan" in n or "stargan" in n: return "StarGAN"
        if "str_0.40" in n: return "SDEdit"
        if "trial" in n: return "ours-final"
        return "other"
    if "samam" in n: return "SaMam"
    if "samst" in n: return "SaMST"
    if "s2wat" in n: return "S2WAT"
    if "sdedit" in n or "str_0p" in n: return "SDEdit"
    if "sdturbo" in n: return "SDTurbo"
    if "img2img_turbo" in n: return "Img2ImgTurbo"
    if "styleid" in n or "style_id" in n: return "StyleID"
    if "zimage" in n: return "ZImageTurbo"
    if "cyclegan" in n: return "CycleGAN"
    if "cut" in n and "ablate" not in n: return "CUT"
    if "lbm" in n: return "LBM"
    if "flux2" in n: return "Flux2"
    if "aaai2027" in n: return "ours"
    if root == "experiments": return "ours"
    return "other"

def classify_dataset(name):
    n = name.lower()
    if "distinct5" in n: return "distinct5"
    if "overfit50" in n: return "overfit50"
    if "wikiart5" in n or "wikiarts5" in n: return "wikiart5"
    if "5x5" in n: return "5x5"
    if "5style" in n: return "5style"
    if "2style" in n: return "2style"
    if "8style" in n: return "8style"
    if "legacy256" in n: return "legacy256"
    if "512" in n: return "512"
    if "256" in n: return "256"
    return "-"

def fmt_wall(sf):
    if not sf: return "-"
    for k in ("wall_seconds","WALL_SECONDS","training_wall_time","train_runtime_sec","runtime_seconds","elapsed_seconds","WALL_TIME","wall_time","training_wall_time","train_runtime_sec"):
        if k in sf:
            try:
                v = float(sf[k])
                if v < 60: return f"{v:.0f}s"
                if v < 3600: return f"{v/60:.1f}m"
                return f"{v/3600:.2f}h"
            except: return str(sf[k])
    return "-"

# Annotate
for exp in experiments:
    exp["model_type"] = classify_model(exp["name"], exp["root"])
    exp["dataset"] = classify_dataset(exp["name"])
    exp["wall"] = fmt_wall(exp.get("summary_fields", {}))
    exp["size_bytes"] = parse_size(exp.get("size", ""))

# Build markdown
md = []
md.append("# 远程实验数据整理探查报告")
md.append("")
md.append(f"**扫描时间**: 2026-07-02")
md.append(f"**远程服务器**: 100.115.18.62 (Windows + WSL2)")
md.append(f"**根路径**: /mnt/i/Github/Latent_Style/")
md.append(f"**总实验数**: {len(experiments)}")
md.append("")
md.append("## 总磁盘占用统计")
md.append("")
md.append("| 根目录 | 总大小 |")
md.append("|--------|--------|")
total_known = 0
for rs in root_summaries:
    sz = rs["size"]
    md.append(f"| `{rs['root']}` | {sz} |")
    if sz not in ("TIMEOUT", "ERR", "?"):
        total_known += parse_size(sz)
md.append(f"| **已知合计** | **{total_known/1024**3:.1f}G** |")
md.append(f"| (experiments/ 扫描超时，未计入) | - |")
md.append("")

# Group A: Baseline评估目录 (baseline_pipeline/results + runs)
md.append("## A. Baseline评估目录 (Related_Works/baseline_pipeline/results + runs)")
md.append("")
baseline = [e for e in experiments if e["root"] == "results"]
baseline.sort(key=lambda e: e["mtime"], reverse=True)
md.append("### A.1 baseline_pipeline/results/ ({} 个实验)".format(len(baseline)))
md.append("")
md.append("| 目录名 | mtime | 大小 | 模型 | 数据集 | 训练时长 | ckpt | img | metrics.csv | 备注 |")
md.append("|--------|-------|------|------|--------|----------|------|-----|-------------|------|")
for e in baseline:
    ckpt = e["ckpt_count"]
    img = e["img_count"]
    mc = "✓" if e["has_metrics_csv"] else ""
    note = ""
    if e["size"] in ("TIMEOUT","ERR"): note = "扫描超时"
    if isinstance(ckpt, int) and ckpt > 0: note = (note + " " if note else "") + f"有{ckpt}ckpt"
    md.append(f"| `{e['name']}` | {e['mtime']} | {e['size']} | {e['model_type']} | {e['dataset']} | {e['wall']} | {ckpt} | {img} | {mc} | {note} |")
md.append("")

# runs subtable
runs = [e for e in experiments if e["root"] == "runs"]
runs.sort(key=lambda e: e["size_bytes"], reverse=True)
md.append(f"### A.2 Related_Works/runs/ ({len(runs)} 个实验)")
md.append("")
md.append("| 目录名 | mtime | 大小 | 模型 | 备注 |")
md.append("|--------|-------|------|------|------|")
for e in runs:
    note = ""
    if e["name"].startswith("img2img_turbo_distinct5_remote_smoke_datasets_20260606_"):
        note = "smoke测试重复"
    elif e["name"].startswith("img2img_turbo_distinct5_remote_smoke_20260606_"):
        note = "smoke测试"
    md.append(f"| `{e['name'][:70]}` | {e['mtime']} | {e['size']} | {e['model_type']} | {note} |")
md.append("")

# Group B: SaMam训练目录 (all samam_*)
md.append("## B. SaMam训练目录 (所有samam_*)")
md.append("")
samam = [e for e in experiments if e["model_type"] == "SaMam"]
samam.sort(key=lambda e: e["mtime"], reverse=True)
md.append(f"共 {len(samam)} 个SaMam实验")
md.append("")
md.append("| 目录名 | 路径 | mtime | 大小 | 数据集 | ckpt | img | 备注 |")
md.append("|--------|------|-------|------|--------|------|-----|------|")
for e in samam:
    ckpt = e["ckpt_count"]
    img = e["img_count"]
    note = ""
    if "scratch_7k" in e["name"]: note = "**★Tier1重点训练**"
    if e["size_bytes"] > 1e9: note = (note+" " if note else "") + "大型训练"
    md.append(f"| `{e['name'][:55]}` | {e['root']}/ | {e['mtime']} | {e['size']} | {e['dataset']} | {ckpt} | {img} | {note} |")
md.append("")

# Group C: 我们模型 - aaai2027系列
md.append("## C. 我们模型 - aaai2027系列 (exp/aaai2027_* + exp/inmortal-exp + exp/phase2_eval_rgbcal)")
md.append("")
aaai = [e for e in experiments if e["root"] == "exp" and (e["model_type"] == "ours" or e["name"] in ("inmortal-exp","phase2_eval_rgbcal","highres","620_spatial_bridge"))]
aaai.sort(key=lambda e: e["mtime"], reverse=True)
md.append(f"共 {len(aaai)} 个实验")
md.append("")
md.append("| 目录名 | mtime | 大小 | ckpt | img | 备注 |")
md.append("|--------|-------|------|------|-----|------|")
for e in aaai:
    ckpt = e["ckpt_count"]
    img = e["img_count"]
    note = ""
    if e["size"] == "TIMEOUT": note = "扫描超时(目录大)"
    if "invalid" in e["name"].lower(): note = (note+" " if note else "") + "**失效**"
    if isinstance(ckpt, int) and ckpt == 0 and e["size_bytes"] < 1e6: note = (note+" " if note else "") + "空/失败"
    md.append(f"| `{e['name'][:75]}` | {e['mtime']} | {e['size']} | {ckpt} | {img} | {note} |")
md.append("")

# Group D: 历史实验
md.append("## D. 我们模型 - 历史实验 (experiments/)")
md.append("")
hist = [e for e in experiments if e["root"] == "experiments"]
hist_valid = [e for e in hist if e["size_bytes"] > 0 or e["summary_count"] > 0 or (isinstance(e["ckpt_count"],int) and e["ckpt_count"]>0) or (isinstance(e["img_count"],int) and e["img_count"]>0)]
hist_empty = [e for e in hist if e not in hist_valid]
hist.sort(key=lambda e: e["size_bytes"], reverse=True)
md.append(f"共 {len(hist)} 个实验（其中 {len(hist_valid)} 个有内容，{len(hist_empty)} 个为空目录或4K占位）")
md.append("")
md.append("### D.1 Top 30 最大历史实验")
md.append("")
md.append("| 目录名 | mtime | 大小 | summary | ckpt | img |")
md.append("|--------|-------|------|---------|------|-----|")
for e in hist[:30]:
    ckpt = e["ckpt_count"]
    img = e["img_count"]
    md.append(f"| `{e['name'][:55]}` | {e['mtime']} | {e['size']} | {e['summary_count']} | {ckpt} | {img} |")
md.append("")
md.append("### D.2 历史实验按mtime分组")
md.append("")
by_month = defaultdict(list)
for e in hist:
    m = e["mtime"][:7] if e["mtime"] != "?" else "unknown"
    by_month[m].append(e)
md.append("| 月份 | 实验数 | 总大小(估算) |")
md.append("|------|--------|-------------|")
for m in sorted(by_month.keys(), reverse=True):
    exps = by_month[m]
    total = sum(e["size_bytes"] for e in exps)
    md.append(f"| {m} | {len(exps)} | {total/1024**3:.2f}G |")
md.append("")

# Group E: 其他重要目录
md.append("## E. 其他重要目录")
md.append("")
md.append("### E.1 final_works/ (最终展示用)")
md.append("")
fw = [e for e in experiments if e["root"] == "final_works"]
fw.sort(key=lambda e: e["name"])
md.append("| 目录名 | mtime | 大小 | 模型 |")
md.append("|--------|-------|------|------|")
for e in fw:
    md.append(f"| `{e['name']}` | {e['mtime']} | {e['size']} | {e['model_type']} |")
md.append("")

# 决策清单
md.append("## 决策清单")
md.append("")
md.append("### 可删除候选清单")
md.append("")
md.append("#### 1. 显式smoke/probe/debug测试 (重复且无价值)")
md.append("")
deletable_smoke = []
for e in experiments:
    n = e["name"].lower()
    if any(k in n for k in ["smoke", "_probe", "debug", "_diag", "fast_eval"]):
        if e["size_bytes"] < 100*1024*1024:  # < 100M
            deletable_smoke.append(e)
deletable_smoke.sort(key=lambda e: e["size_bytes"], reverse=True)
md.append(f"共 {len(deletable_smoke)} 个小型smoke/probe/debug目录")
md.append("")
md.append("| 目录名 | 路径 | 大小 | mtime |")
md.append("|--------|------|------|-------|")
for e in deletable_smoke[:30]:
    md.append(f"| `{e['name'][:60]}` | {e['root']}/ | {e['size']} | {e['mtime']} |")
if len(deletable_smoke) > 30:
    md.append(f"| ... 还有 {len(deletable_smoke)-30} 个 | | | |")
md.append("")

md.append("#### 2. img2img_turbo重复smoke测试 (Related_Works/runs/)")
md.append("")
i2i = [e for e in experiments if "img2img_turbo_distinct5_remote_smoke" in e["name"].lower()]
i2i_total = sum(e["size_bytes"] for e in i2i)
md.append(f"共 {len(i2i)} 个，总大小 {i2i_total/1024**3:.2f}G。其中 `_datasets_` 后缀的是带数据集副本，可考虑删除数据集副本")
md.append("")

md.append("#### 3. SaMam/SaMST 失败probe (4K-50K空目录)")
md.append("")
empty_samam = [e for e in experiments if e["model_type"] in ("SaMam","SaMST") and e["size_bytes"] < 200*1024]
empty_samam.sort(key=lambda e: e["mtime"], reverse=True)
md.append(f"共 {len(empty_samam)} 个小型SaMam/SaMST目录（<200K，多为失败probe）")
md.append("")
md.append("| 目录名 | 路径 | 大小 | mtime |")
md.append("|--------|------|------|-------|")
for e in empty_samam:
    md.append(f"| `{e['name'][:60]}` | {e['root']}/ | {e['size']} | {e['mtime']} |")
md.append("")

md.append("#### 4. aaai2027系列中的invalid/空目录")
md.append("")
invalid_aaai = [e for e in experiments if "invalid" in e["name"].lower() or (e["root"]=="exp" and isinstance(e["ckpt_count"],int) and e["ckpt_count"]==0 and e["size_bytes"]<2*1024*1024 and e["model_type"]=="ours")]
invalid_aaai.sort(key=lambda e: e["mtime"], reverse=True)
md.append(f"共 {len(invalid_aaai)} 个")
md.append("")
md.append("| 目录名 | mtime | 大小 | ckpt |")
md.append("|--------|-------|------|------|")
for e in invalid_aaai:
    md.append(f"| `{e['name'][:75]}` | {e['mtime']} | {e['size']} | {e['ckpt_count']} |")
md.append("")

md.append("### 必须保留清单")
md.append("")
md.append("#### A. Baseline关键实验（用于论文对比）")
md.append("")
keep_baseline = [e for e in experiments if e["root"]=="results" and e["model_type"] in ("CUT","SaMST","SDEdit","SDTurbo","StyleID","ZImageTurbo","CycleGAN","S2WAT","Img2ImgTurbo") and e["size_bytes"] > 1*1024*1024]
keep_baseline.sort(key=lambda e: (e["model_type"], -e["size_bytes"]))
md.append("| 目录名 | 模型 | 大小 | mtime |")
md.append("|--------|------|------|-------|")
for e in keep_baseline:
    md.append(f"| `{e['name']}` | {e['model_type']} | {e['size']} | {e['mtime']} |")
md.append("")

md.append("#### B. SaMam关键训练")
md.append("")
keep_samam = [e for e in experiments if e["model_type"]=="SaMam" and (e["size_bytes"] > 500*1024*1024 or "scratch_7k" in e["name"])]
keep_samam.sort(key=lambda e: e["size_bytes"], reverse=True)
md.append("| 目录名 | 路径 | 大小 | mtime | 备注 |")
md.append("|--------|------|------|-------|------|")
for e in keep_samam:
    note = "**★Tier1重点**" if "scratch_7k" in e["name"] else "大训练"
    md.append(f"| `{e['name'][:55]}` | {e['root']}/ | {e['size']} | {e['mtime']} | {note} |")
md.append("")

md.append("#### C. aaai2027关键阶段（有大量ckpt的）")
md.append("")
keep_aaai = [e for e in experiments if e["root"]=="exp" and isinstance(e["ckpt_count"],int) and e["ckpt_count"]>=10]
keep_aaai.sort(key=lambda e: e["ckpt_count"], reverse=True)
md.append("| 目录名 | mtime | 大小 | ckpt数 |")
md.append("|--------|-------|------|--------|")
for e in keep_aaai:
    md.append(f"| `{e['name'][:75]}` | {e['mtime']} | {e['size']} | {e['ckpt_count']} |")
md.append("")

md.append("#### D. final_works/ 全部保留")
md.append("final_works/ 是最终展示用，全部保留。")
md.append("")

md.append("#### E. 历史实验保留建议")
md.append(f"experiments/ 共 {len(hist)} 个，其中：")
md.append(f"- 有summary/checkpoint/images的: {len(hist_valid)} 个 → 保留")
md.append(f"- 空/4K占位: {len(hist_empty)} 个 → 可删除（节省极少空间）")
md.append("")

# Summary
md.append("## 总结")
md.append("")
md.append(f"- **总扫描实验数**: {len(experiments)}")
md.append(f"- **已知根目录大小**: {total_known/1024**3:.1f}G (experiments/未计入)")
md.append(f"- **可删除候选**: ~{len(deletable_smoke)+len(i2i)+len(empty_samam)+len(invalid_aaai)} 个目录")
md.append(f"  - smoke/probe/debug: {len(deletable_smoke)}")
md.append(f"  - img2img_turbo smoke: {len(i2i)} ({i2i_total/1024**3:.2f}G)")
md.append(f"  - SaMam/SaMST 失败probe: {len(empty_samam)}")
md.append(f"  - aaai2027 invalid/空: {len(invalid_aaai)}")
md.append(f"- **必须保留**: baseline关键实验 + SaMam大训练 + aaai2027有ckpt的 + final_works全部 + experiments有内容的")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"Report written to: {OUT}")
print(f"Total lines: {len(md)}")
