"""FC-SB Phase 4 阶段 0 推理消融结果汇总.

读取 exp/p4_fusion_breakout/infer_ablation/D*.json, 按 clip_style 降序排列,
生成 _summary.md 表格 + Pareto 前沿标记.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ABLATION_DIR = ROOT / "exp" / "p4_fusion_breakout" / "infer_ablation"
SUMMARY_PATH = ABLATION_DIR / "_summary.md"

# 历史 Pareto 前沿点 (来自 project_memory)
HISTORY_POINTS = [
    {"name": "E4-long ep5 (baseline)", "clip": 0.727, "lpips": 0.581, "source": "FC-SB spatial"},
    {"name": "B2 V2 epoch_0001", "clip": 0.6731, "lpips": 0.2781, "source": "Spectral ODE"},
    {"name": "U4 (alpha=0.1)", "clip": 0.7225, "lpips": 0.3660, "source": "推理侧外推"},
    {"name": "V3 (k=16)", "clip": 0.7295, "lpips": 0.3963, "source": "推理侧 patch"},
    {"name": "V6 (k=32)", "clip": 0.7262, "lpips": 0.3722, "source": "推理侧 patch"},
    {"name": "I7 baseline", "clip": 0.7031, "lpips": 0.3399, "source": "FC-SB spatial"},
]


def _load_results() -> list[dict]:
    """加载所有 D*.json 结果."""
    results = []
    if not ABLATION_DIR.exists():
        return results
    for json_path in sorted(ABLATION_DIR.glob("D*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            results.append({
                "name": data.get("exp_name", json_path.stem),
                "clip": metrics.get("transfer_clip_style"),
                "lpips": metrics.get("transfer_content_lpips"),
                "wfi": metrics.get("wfi_score"),
                "params": data.get("params", {}),
                "path": str(json_path),
            })
        except Exception as e:
            print(f"[summarize] WARN: failed to load {json_path}: {e}")
    return results


def _is_pareto(point: dict, all_points: list[dict]) -> bool:
    """判断 point 是否在 Pareto 前沿 (clip 越高越好, lpips 越低越好)."""
    p_clip = point.get("clip")
    p_lpips = point.get("lpips")
    if p_clip is None or p_lpips is None:
        return False
    for other in all_points:
        o_clip = other.get("clip")
        o_lpips = other.get("lpips")
        if o_clip is None or o_lpips is None:
            continue
        # other 支配 point: clip >= 且 lpips <=, 且至少一个严格
        if o_clip >= p_clip and o_lpips <= p_lpips:
            if o_clip > p_clip or o_lpips < p_lpips:
                return False
    return True


def main() -> None:
    results = _load_results()
    if not results:
        print(f"[summarize] No results found in {ABLATION_DIR}")
        return

    # 合并历史点用于 Pareto 判断
    all_for_pareto = []
    for r in results:
        if r["clip"] is not None and r["lpips"] is not None:
            all_for_pareto.append({"clip": r["clip"], "lpips": r["lpips"]})
    for h in HISTORY_POINTS:
        all_for_pareto.append({"clip": h["clip"], "lpips": h["lpips"]})

    # 按 clip 降序排列
    results_sorted = sorted(results, key=lambda x: (x["clip"] or 0), reverse=True)

    lines = []
    lines.append("# FC-SB Phase 4 阶段 0 推理消融汇总\n")
    lines.append(f"基于 E4-long ep5 checkpoint (clip=0.727, lpips=0.581)\n")
    lines.append(f"目标: clip_style > 0.74 且 LPIPS < 0.35\n\n")

    lines.append("## 本次消融结果 (按 clip 降序)\n")
    lines.append("| 排名 | 实验 | clip_style | LPIPS | WFI | Pareto? | 关键参数 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(results_sorted, 1):
        clip = f"{r['clip']:.4f}" if r['clip'] is not None else "N/A"
        lpips = f"{r['lpips']:.4f}" if r['lpips'] is not None else "N/A"
        wfi = f"{r['wfi']:.4f}" if r['wfi'] is not None else "N/A"
        pareto = "**PARETO**" if _is_pareto(r, all_for_pareto) else ""
        params = r.get("params", {})
        key_params = []
        if params.get("lowpass_mode", "avg_pool") != "avg_pool":
            key_params.append(f"lp={params['lowpass_mode']}")
        if params.get("style_extrap_alpha", 0) > 0:
            key_params.append(f"α={params['style_extrap_alpha']}")
        if params.get("patch_adain_kernel", 0) > 0:
            key_params.append(f"k={params['patch_adain_kernel']}")
        if params.get("multiband_adain_mode", "single") != "single":
            key_params.append(f"mb={params['multiband_adain_mode']}")
        if params.get("tri_band_inference_lock", False):
            key_params.append("triband")
        param_str = ", ".join(key_params) if key_params else "baseline"
        lines.append(f"| {i} | {r['name']} | {clip} | {lpips} | {wfi} | {pareto} | {param_str} |")

    lines.append("\n## 历史 Pareto 前沿点 (对照)\n")
    lines.append("| 实验 | clip_style | LPIPS | 来源 |")
    lines.append("|---|---|---|---|")
    for h in sorted(HISTORY_POINTS, key=lambda x: x["clip"], reverse=True):
        lines.append(f"| {h['name']} | {h['clip']:.4f} | {h['lpips']:.4f} | {h['source']} |")

    lines.append("\n## 目标达成判定\n")
    target_clip = 0.74
    target_lpips = 0.35
    achieved = [r for r in results if r["clip"] is not None and r["lpips"] is not None
                and r["clip"] > target_clip and r["lpips"] < target_lpips]
    if achieved:
        lines.append(f"✅ **双指标达成 (clip>{target_clip}, LPIPS<{target_lpips})**: " +
                     ", ".join(r["name"] for r in achieved))
    else:
        clip_break = [r for r in results if r["clip"] is not None and r["clip"] > target_clip]
        lpips_break = [r for r in results if r["lpips"] is not None and r["lpips"] < target_lpips]
        lines.append(f"❌ 双指标未达。clip>{target_clip} 的: {len(clip_break)} 组; LPIPS<{target_lpips} 的: {len(lpips_break)} 组")
        if clip_break:
            lines.append(f"   - clip 突破: " + ", ".join(f"{r['name']}({r['clip']:.4f})" for r in clip_break))
        if lpips_break:
            lines.append(f"   - lpips 突破: " + ", ".join(f"{r['name']}({r['lpips']:.4f})" for r in lpips_break))

    lines.append("\n## 阶段 1 训练配置建议\n")
    best_clip = max(results, key=lambda x: x["clip"] or 0) if results else None
    best_pareto = [r for r in results if _is_pareto(r, all_for_pareto)]
    if best_pareto:
        lines.append(f"- 阶段 1 训练配置应采用 Pareto 前沿点的推理参数组合")
        lines.append(f"- 当前 Pareto 前沿点: " + ", ".join(r["name"] for r in best_pareto))
    elif best_clip:
        lines.append(f"- 阶段 1 训练配置参考最高 clip 点: {best_clip['name']} (clip={best_clip['clip']:.4f})")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize] wrote {SUMMARY_PATH}")
    print(f"[summarize] {len(results)} results loaded")
    if achieved:
        print(f"[summarize] 🎯 TARGET ACHIEVED by: {', '.join(r['name'] for r in achieved)}")
    else:
        print(f"[summarize] target not achieved yet")


if __name__ == "__main__":
    main()
