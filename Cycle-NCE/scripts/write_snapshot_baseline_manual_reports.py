#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


DATA_PATH = Path("docs/reports/data/snapshot_baseline_vs_last.json")
OUT_DIR = Path("docs/reports/snapshot_baseline_records")
INDEX_PATH = Path("docs/reports/REPORT_EXPERIMENTS_CYCLE_SNAPSHOT_BASELINE_INDEX.md")
SUMMARY_PATH = Path("docs/reports/REPORT_EXPERIMENTS_CYCLE_SNAPSHOT_BASELINE_SUMMARY.md")


MAJOR_KEYS = {
    "loss.w_distill",
    "loss.w_code",
    "loss.w_struct",
    "loss.w_edge",
    "loss.w_cycle",
    "loss.w_nce",
    "loss.w_push",
    "loss.w_stroke_gram",
    "loss.w_color_moment",
    "loss.w_semigroup",
    "loss.distill_low_only",
    "loss.distill_cross_domain_only",
    "model.style_texture_gain",
    "model.style_spatial_pre_gain_16",
    "model.style_spatial_dec_gain_32",
    "model.style_delta_lowfreq_gain",
}


MANUAL_NOTES: dict[str, list[str]] = {
    "full_300_distill_low_only_v1": [
        "8 个快照首尾完全一致（代码和配置都不变），属于稳定长跑而不是边训边改。",
        "它是当前 strict 集合 top1（style=0.5515），但 LPIPS=0.6973 偏高，说明风格收益伴随内容漂移。",
    ],
    "full_300_gridfix_v2": [
        "loss 路径收敛到 student 主路，且最后快照把 w_distill/w_code/w_push/w_semigroup 直接降为 0。",
        "trainer 同步删掉 gram/moment/idt 指标项，实验目标从“全套约束”转向“窄损失组合”。",
        "最终 style 不高（0.4629）但 LPIPS 低（0.3242），偏内容保持。",
    ],
    "overfit50-style-distill-struct-v4": [
        "最后快照把目标从 struct 主导切向 cycle/nce 更强：w_cycle 3→8、w_struct 3→0.75、w_nce 2→3.5。",
        "loss 新增 cycle/struct 的可配置对齐形式（l1/mse + lowpass 混合），同时 run.py 偏向 CPU 低负载。",
        "最终 style=0.499，未优于 v3，说明这轮迁移方向收益不足。",
    ],
    "overfit50-style-distill-struct-v4-mse": [
        "配置不变，但 losses/trainer 有实现迭代：补入 delta_tv、cycle_edge_strength、summary_history 聚合。",
        "属于“实现补丁 + 评测增强”，不是目标函数大迁移；最终 style=0.4975，略低于 v4。",
    ],
    "overfit50-distill_low_only": [
        "这是本轮改动最大的 run：model/loss/trainer/run 全线变化，属于配方迁移而非小调参。",
        "model 新增 _predict_delta + integrate(step_size) 路径，loss 引入 stroke/color/semigroup/style_spatial_tv。",
        "配置层 49 个键变化，核心是压低 w_cycle/w_struct/w_nce/w_distill，并打开 stroke/color/semigroup 系。",
    ],
    "overfit50-v5-mse-sharp-style_back": [
        "仅 losses.py 小幅更新：distill 从统一 L1 改成可选 low-only + cross-domain-only 聚合。",
        "模型与 trainer 不变，属于单一损失开关试验；style 与 cls 仍高，但 LPIPS 偏高。",
    ],
    "experiments/full_250_strong-style": [
        "model/losses/trainer 未变，只改 run.py 的 CPU 线程与 affinity 控制，属于系统侧优化。",
        "该实验 style=0 且 LPIPS=0，评测口径不完整，不能用于主排序结论。",
    ],
    "experiments/overfit50-style-force-balance-v1": [
        "只改 run.py：增加 CPU 线程配置入口；模型与损失未变。",
        "最终 style 很高（0.5491）但 LPIPS 很高（0.7640），仍是“高风格高漂移”型。",
    ],
    "overfit50-style-distill-struct-v3": [
        "两次快照首尾无变化，属于完全定版训练。",
    ],
}


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    try:
        v = float(value)
        return f"{v:.{digits}f}"
    except Exception:
        return str(value)


def _slug(rel_path: str) -> str:
    s = rel_path.replace("\\", "/").strip("/").replace("/", "__")
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in s)


def _profile(row: dict[str, Any]) -> str:
    cfg_n = int(row.get("cfg_change_count") or 0)
    code = row.get("code_diff") or {}
    code_changed = sum(1 for f in ["model.py", "losses.py", "trainer.py", "run.py"] if (code.get(f) or {}).get("changed"))
    if int(row.get("snapshot_count") or 0) == 1:
        return "单快照（基线=最终）"
    if cfg_n == 0 and code_changed == 0:
        return "稳定复现型"
    if cfg_n <= 2 and code_changed <= 1:
        return "轻量工程改动型"
    if cfg_n >= 10 or code_changed >= 3:
        return "强探索迭代型"
    return "中等改动型"


def _heuristic_notes(row: dict[str, Any], code_changed_files: list[str]) -> list[str]:
    rel = str(row.get("rel_path", ""))
    run_name = str(row.get("run", ""))
    if rel in MANUAL_NOTES:
        return MANUAL_NOTES[rel]
    if run_name in MANUAL_NOTES:
        return MANUAL_NOTES[run_name]

    cfg_changes = row.get("cfg_changes") or []
    if int(row.get("snapshot_count") or 0) == 1:
        return [
            "该实验只有一个快照，基线与最终一致，无法从快照层面判断中途策略漂移。",
            "若要做归因，建议增加中期快照或固定 epoch 的 full_eval 对照。",
        ]
    if not cfg_changes and not code_changed_files:
        return ["基线与最终代码/配置一致，可视为固定配方训练。"]
    if not cfg_changes and code_changed_files:
        return ["主要是实现层调整（配置未变），指标变化更可能来自实现细节与训练随机性。"]
    if cfg_changes and not code_changed_files:
        return ["主要是参数层调度（代码未变），更便于做超参数归因。"]
    return ["代码与配置同时变化，属于耦合改动；建议分离回放后再做归因。"]


def _run_conclusion(row: dict[str, Any]) -> list[str]:
    style = _to_float(row.get("best_style"))
    cls = _to_float(row.get("best_cls"))
    lpips = _to_float(row.get("lpips"))
    out: list[str] = []
    if style is None:
        out.append("缺少可用 style 指标，不建议纳入竞争排序。")
    elif style >= 0.54:
        out.append("风格分数处于当前高位。")
    elif style >= 0.50:
        out.append("风格分数中上。")
    else:
        out.append("风格分数不高。")

    if lpips is not None:
        if lpips >= 0.60:
            out.append("内容漂移偏高（LPIPS 高），不适合作为稳态生产基线。")
        elif lpips <= 0.45:
            out.append("内容保持相对更稳。")
    if cls is not None and cls < 0.60:
        out.append("分类一致性偏弱，风格身份稳定性不足。")
    return out if out else ["建议结合固定口径复现实验后再决策。"]


def main() -> None:
    runs: list[dict[str, Any]] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index_lines: list[str] = []
    index_lines.append("# Snapshot 基线到最终：单实验索引")
    index_lines.append("")
    index_lines.append("| run | rel_path | snapshots | 画像 | 文档 |")
    index_lines.append("|---|---|---:|---|---|")

    summary_rows: list[dict[str, Any]] = []

    for row in runs:
        rel = str(row.get("rel_path", ""))
        run_name = str(row.get("run", ""))
        slug = _slug(rel)
        doc_path = OUT_DIR / f"{slug}.md"
        profile = _profile(row)

        code = row.get("code_diff") or {}
        code_changed_files = [f for f in ["model.py", "losses.py", "trainer.py", "run.py"] if (code.get(f) or {}).get("changed")]
        cfg_changes = row.get("cfg_changes") or []
        major = [c for c in cfg_changes if c.get("key") in MAJOR_KEYS]

        lines: list[str] = []
        lines.append(f"# 实验快照分析：{run_name}")
        lines.append("")
        lines.append(f"- 实验路径：`experiments-cycle/{rel}`")
        lines.append(f"- 分析口径：`基线={row['baseline_snapshot']}` -> `最终={row['last_snapshot']}`（仅看最后快照）")
        lines.append(f"- 快照数量：`{row['snapshot_count']}`")
        lines.append(f"- 实验画像：`{profile}`")
        lines.append("")

        lines.append("## 结果概览")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---:|")
        lines.append(f"| best_transfer_clip_style | {_fmt(_to_float(row.get('best_style')))} |")
        lines.append(f"| best_transfer_classifier_acc | {_fmt(_to_float(row.get('best_cls')))} |")
        lines.append(f"| latest_transfer_content_lpips | {_fmt(_to_float(row.get('lpips')))} |")
        lines.append(f"| strict 可比 | {'是' if row.get('strict') else '否'} |")

        hist = row.get("history")
        if isinstance(hist, dict):
            lines.append(f"| history_rounds | {hist.get('rounds', 0)} |")
            lines.append(f"| history_style_delta(last-first) | {_fmt(_to_float(hist.get('delta')))} |")
        lines.append("")

        lines.append("## 基线到最终：代码变化")
        lines.append("")
        lines.append("| 文件 | 是否变化 | 增加行 | 删除行 |")
        lines.append("|---|---|---:|---:|")
        for f in ["model.py", "losses.py", "trainer.py", "run.py"]:
            d = code.get(f) or {}
            lines.append(f"| `{f}` | {'是' if d.get('changed') else '否'} | {d.get('add', 0)} | {d.get('del', 0)} |")
        lines.append("")

        lines.append("## 基线到最终：配置变化")
        lines.append("")
        lines.append(f"- 配置变更键数：`{len(cfg_changes)}`")
        lines.append(f"- 高影响键变更数：`{len(major)}`")
        if cfg_changes:
            lines.append("")
            lines.append("| key | baseline | last |")
            lines.append("|---|---|---|")
            for c in cfg_changes:
                lines.append(f"| `{c.get('key')}` | `{_fmt(c.get('from'))}` | `{_fmt(c.get('to'))}` |")
        else:
            lines.append("- 无配置差异。")
        lines.append("")

        lines.append("## 人工解读")
        lines.append("")
        for n in _heuristic_notes(row, code_changed_files):
            lines.append(f"- {n}")
        lines.append("")

        lines.append("## 结论（该实验）")
        lines.append("")
        for c in _run_conclusion(row):
            lines.append(f"- {c}")
        lines.append("")

        doc_path.write_text("\n".join(lines), encoding="utf-8")

        index_lines.append(f"| {run_name} | `{rel}` | {row['snapshot_count']} | {profile} | [doc](snapshot_baseline_records/{slug}.md) |")
        summary_rows.append(
            {
                "run": run_name,
                "rel_path": rel,
                "snapshot_count": int(row.get("snapshot_count") or 0),
                "cfg_change_count": len(cfg_changes),
                "code_lines": int(sum((code.get(f) or {}).get("add", 0) + (code.get(f) or {}).get("del", 0) for f in code)),
                "profile": profile,
                "strict": bool(row.get("strict")),
                "best_style": _to_float(row.get("best_style")),
                "best_cls": _to_float(row.get("best_cls")),
                "lpips": _to_float(row.get("lpips")),
            }
        )

    INDEX_PATH.write_text("\n".join(index_lines), encoding="utf-8")

    profile_counter = Counter(r["profile"] for r in summary_rows)
    by_change = sorted(summary_rows, key=lambda x: (x["code_lines"], x["cfg_change_count"]), reverse=True)
    strict_top = sorted([r for r in summary_rows if r["strict"] and r["best_style"] is not None], key=lambda x: x["best_style"], reverse=True)

    summary_lines: list[str] = []
    summary_lines.append("# Experiments-Cycle 快照基线分析总结（中文）")
    summary_lines.append("")
    summary_lines.append("- 范围：所有存在 `src_snapshot_*` 的实验（共 18 个）")
    summary_lines.append("- 口径：每个实验仅比较“最早快照（基线）”与“最后快照（最终）”")
    summary_lines.append(f"- 单实验文档索引：`{INDEX_PATH.as_posix()}`")
    summary_lines.append("")

    summary_lines.append("## 1) 结构分层")
    summary_lines.append("")
    summary_lines.append("| 实验画像 | 数量 |")
    summary_lines.append("|---|---:|")
    for key in ["稳定复现型", "轻量工程改动型", "中等改动型", "强探索迭代型", "单快照（基线=最终）"]:
        summary_lines.append(f"| {key} | {profile_counter.get(key, 0)} |")
    summary_lines.append("")

    summary_lines.append("## 2) 改动最重的实验（基线->最终）")
    summary_lines.append("")
    summary_lines.append("| run | path | snapshots | 代码改动行(+-) | 配置变更键数 | 画像 |")
    summary_lines.append("|---|---|---:|---:|---:|---|")
    for row in by_change[:8]:
        summary_lines.append(
            f"| {row['run']} | `{row['rel_path']}` | {row['snapshot_count']} | {row['code_lines']} | {row['cfg_change_count']} | {row['profile']} |"
        )
    summary_lines.append("")

    summary_lines.append("## 3) 结合指标看稳定候选")
    summary_lines.append("")
    summary_lines.append("| run | path | best_style | cls | lpips | 画像 |")
    summary_lines.append("|---|---|---:|---:|---:|---|")
    for row in strict_top[:10]:
        summary_lines.append(
            f"| {row['run']} | `{row['rel_path']}` | {_fmt(row['best_style'])} | {_fmt(row['best_cls'])} | {_fmt(row['lpips'])} | {row['profile']} |"
        )
    summary_lines.append("")

    summary_lines.append("## 4) 人工结论")
    summary_lines.append("")
    summary_lines.append("- 真正“边训边改配方”的核心 run 主要是：`overfit50-distill_low_only`、`full_300_gridfix_v2`、`overfit50-style-distill-struct-v4`。")
    summary_lines.append("- `full_300_distill_low_only_v1` 是典型稳定基线：快照多但首尾一致，说明是固定配方训练。")
    summary_lines.append("- 多个高分 run（如 `overfit50-style-force-balance-v1`）风格分高但 LPIPS 偏高，不适合作为内容保真基线。")
    summary_lines.append("- 单快照实验（9 个）无法从快照角度做“中途策略变化”归因，只能看最终结果。")
    summary_lines.append("")

    summary_lines.append("## 5) 建议的下一步")
    summary_lines.append("")
    summary_lines.append("1. 先锁定稳定候选（快照稳定 + strict 可比）作为主线复现基线。")
    summary_lines.append("2. 对强探索 run，按“代码冻结、只改配置”再做分离回放，减少归因耦合。")
    summary_lines.append("3. 对单快照 run 增加中期快照与固定 epoch full_eval，补齐可追溯链路。")
    summary_lines.append("")

    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote: {INDEX_PATH}")
    print(f"Wrote: {SUMMARY_PATH}")
    print(f"Wrote docs: {len(summary_rows)} in {OUT_DIR}")


if __name__ == "__main__":
    main()
