#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SNAPSHOT_BASE_PATH = Path("docs/reports/data/snapshot_baseline_vs_last.json")
RUNS_METRICS_PATH = Path("docs/experiments_cycle/data/runs_metrics.csv")
OUT_PATH = Path("docs/reports/REPORT_EXPERIMENTS_CYCLE_CODE_CHANGE_IMPACT.md")


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except Exception:
        return None


def _fmt(v: Any, digits: int = 6) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    try:
        f = float(v)
        return f"{f:.{digits}f}"
    except Exception:
        return str(v)


def _load_runs_metrics() -> dict[str, dict[str, Any]]:
    rows = list(csv.DictReader(RUNS_METRICS_PATH.open("r", encoding="utf-8", newline="")))
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        rel = str(r.get("rel_path") or "").replace("\\", "/")
        out[rel] = {
            **r,
            "run": r.get("run"),
            "rel_path": rel,
            "best_style": _to_float(r.get("best_transfer_clip_style")),
            "best_cls": _to_float(r.get("best_transfer_classifier_acc")),
            "lpips": _to_float(r.get("latest_transfer_content_lpips")),
            "eval_count": _to_float(r.get("matrix_eval_count_mean")),
            "history_rounds": _to_int(r.get("history_rounds")) or 0,
            "snapshot_count": _to_int(r.get("snapshot_count")) or 0,
            "strict": str(r.get("matrix_complete_square", "")).strip().lower() in {"1", "true", "yes"}
            and (_to_float(r.get("latest_transfer_content_lpips")) or 0.0) > 0.0
            and _to_float(r.get("best_transfer_clip_style")) is not None,
        }
    return out


def _delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in ["best_style", "best_cls", "lpips"]:
        va = a.get(k)
        vb = b.get(k)
        out[k] = None if (va is None or vb is None) else float(vb) - float(va)
    return out


def _profile(snapshot_row: dict[str, Any]) -> str:
    cfg = int(snapshot_row.get("cfg_change_count") or 0)
    code = snapshot_row.get("code_diff") or {}
    changed = sum(1 for f in ["model.py", "losses.py", "trainer.py", "run.py"] if (code.get(f) or {}).get("changed"))
    snap = int(snapshot_row.get("snapshot_count") or 0)
    if snap == 1:
        return "单快照"
    if cfg == 0 and changed == 0:
        return "稳定复现"
    if cfg <= 2 and changed <= 1:
        return "轻量改动"
    if cfg >= 10 or changed >= 3:
        return "强探索改动"
    return "中等改动"


def main() -> None:
    snapshot_rows: list[dict[str, Any]] = json.loads(SNAPSHOT_BASE_PATH.read_text(encoding="utf-8"))
    runs_metrics = _load_runs_metrics()

    snapshot_rows = [r for r in snapshot_rows if (runs_metrics.get(r["rel_path"], {}).get("snapshot_count", 0) > 0)]
    snapshot_rows.sort(
        key=lambda r: (
            int(r.get("snapshot_count") or 0),
            int(r.get("cfg_change_count") or 0),
            sum(
                ((r.get("code_diff") or {}).get(f) or {}).get("add", 0)
                + ((r.get("code_diff") or {}).get(f) or {}).get("del", 0)
                for f in ["model.py", "losses.py", "trainer.py", "run.py"]
            ),
        ),
        reverse=True,
    )

    changed_rows = []
    for r in snapshot_rows:
        code = r.get("code_diff") or {}
        files = [f for f in ["model.py", "losses.py", "trainer.py", "run.py"] if (code.get(f) or {}).get("changed")]
        if files:
            changed_rows.append((r, files))

    # 手工定义“最近同族对照”用于解释改动影响（非严格因果，仅作经验对照）。
    compare_pairs = [
        ("overfit50-style-distill-struct-v3", "overfit50-style-distill-struct-v4", "v3->v4：cycle/nce 加强 + struct 降低"),
        ("overfit50-style-distill-struct-v4", "overfit50-style-distill-struct-v4-mse", "v4->v4-mse：在 v4 上加入 edge/delta_tv 等实现修正"),
        ("overfit50-v5-mse-sharp", "overfit50-v5-mse-sharp-style_back", "v5->v5-style_back：distill 聚合方式调整"),
        ("experiments/overfit50-style-force-balance-v1", "experiments/overfit50-style-force-balance-v1-cycle4", "force-balance v1->cycle4：后续训练轮次对照"),
        ("overfit50-distill_low_only", "overfit50-strok-style", "distill_low_only->strok-style：相近配方的同类对照"),
        ("full_300_gridfix_v2", "full_300_distill_low_only_v1", "full_300 分支：gridfix_v2 vs distill_low_only_v1"),
    ]

    compare_lines: list[str] = []
    compare_lines.append("| 对照 | A(style/cls/lpips) | B(style/cls/lpips) | Δstyle | Δcls | Δlpips | 观察 |")
    compare_lines.append("|---|---|---|---:|---:|---:|---|")
    for a_rel, b_rel, title in compare_pairs:
        a = runs_metrics.get(a_rel)
        b = runs_metrics.get(b_rel)
        if not a or not b:
            continue
        d = _delta(a, b)
        compare_lines.append(
            f"| {title} | {_fmt(a['best_style'])}/{_fmt(a['best_cls'])}/{_fmt(a['lpips'])} | "
            f"{_fmt(b['best_style'])}/{_fmt(b['best_cls'])}/{_fmt(b['lpips'])} | "
            f"{_fmt(d['best_style'])} | {_fmt(d['best_cls'])} | {_fmt(d['lpips'])} | "
            "仅作经验对照，非严格 A/B 因果 |"
        )

    deep_notes: dict[str, list[str]] = {
        "full_300_gridfix_v2": [
            "代码上从 teacher/code 辅助路径收缩到 student 主路径，并在配置里把 w_distill/w_code/w_push/w_semigroup 归零。",
            "指标表现为 style 与 cls 显著偏低（0.4629 / 0.37），但 lpips 很低（0.3242）。",
            "影响判断：该类“去 teacher/code”改动更像是在换取内容保持，牺牲了风格强度与可辨识性（高置信度）。",
        ],
        "overfit50-style-distill-struct-v4": [
            "核心变更是损失权重重心从 struct 向 cycle/nce 偏移（w_cycle↑, w_nce↑, w_struct↓）并引入 edge。",
            "与 v3 对照：style -0.0175，cls -0.12，lpips -0.0768。",
            "影响判断：该权重迁移让内容更稳（lpips 下降）但风格表达和分类一致性下降（中高置信度）。",
        ],
        "overfit50-style-distill-struct-v4-mse": [
            "在 v4 基础上增加 cycle_edge_strength / delta_tv 等实现修正，配置层基本不变。",
            "与 v4 对照：style -0.0015，cls +0.09，lpips +0.0043。",
            "影响判断：属于“小幅结构化修正”，主要改善 cls，几乎不改 style 上限（中置信度）。",
        ],
        "overfit50-distill_low_only": [
            "这是最大规模迁移：model 新增 integrate(step_size)，loss 引入 stroke/color/semigroup 等新项，配置 49 键变化。",
            "最终指标（0.5185/0.93/0.5456）说明该配方可达到较好的 style+cls 平衡，但内容漂移仍中高。",
            "与 strok-style 对照时 style 基本相当但 lpips 更高，提示还存在可优化的内容保真空间（中置信度）。",
        ],
        "experiments/overfit50-style-force-balance-v1": [
            "仅 run.py 的 CPU 线程控制改动，目标是训练稳定性和资源控制，不直接改损失函数。",
            "最终 style 很高（0.5491）但 lpips 也高（0.7640）。",
            "影响判断：当前指标更像原始配方属性，run.py 工程改动不是决定性风格因素（中置信度）。",
        ],
        "overfit50-v5-mse-sharp-style_back": [
            "仅 losses.py 的 distill 聚合策略变更（支持 low-only + cross-domain-only）。",
            "与 v5 对照：style +0.0857，cls +0.92，同时 lpips +0.3647。",
            "影响判断：这是典型“强风格提升伴随强内容漂移”的单点改动（高置信度）。",
        ],
        "experiments/full_250_strong-style": [
            "仅 run.py 的 CPU 环境线程/affinity 改动，模型与损失不变。",
            "该 run 指标口径不完整（style=0, lpips=0），无法做有效风格归因。",
            "影响判断：应先补齐评测，再讨论改动对风格影响（高置信度）。",
        ],
    }

    lines: list[str] = []
    lines.append("# Experiments-Cycle 代码改动影响深度报告（中文）")
    lines.append("")
    lines.append("- 目标：评估“每个代码改动”对风格效果与相关指标（style / cls / lpips / eval_count / history）的影响。")
    lines.append("- 数据：`docs/reports/data/snapshot_baseline_vs_last.json` + `docs/experiments_cycle/data/runs_metrics.csv`。")
    lines.append("- 范围：所有有快照实验 18 个，其中有真实代码改动的实验 7 个。")
    lines.append("- 说明：多数 run 没有严格同配置 A/B；以下“影响”是基于同族对照与代码语义的经验归因，已标注置信度。")
    lines.append("")

    lines.append("## 1) 全量快照实验指标总览（18个）")
    lines.append("")
    lines.append("| run | path | snapshots | 画像 | best_style | cls | lpips | eval_count | history | strict |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---|")
    for r in snapshot_rows:
        rel = r["rel_path"]
        rm = runs_metrics.get(rel, {})
        lines.append(
            f"| {r['run']} | `{rel}` | {rm.get('snapshot_count', 0)} | {_profile(r)} | "
            f"{_fmt(rm.get('best_style'))} | {_fmt(rm.get('best_cls'))} | {_fmt(rm.get('lpips'))} | "
            f"{_fmt(rm.get('eval_count'), 0)} | {rm.get('history_rounds', 0)} | {'是' if rm.get('strict') else '否'} |"
        )
    lines.append("")

    lines.append("## 2) 同族对照：改动与指标变化")
    lines.append("")
    lines.extend(compare_lines)
    lines.append("")

    lines.append("## 3) 逐实验代码改动影响（7个改动实验）")
    lines.append("")
    for r, files in changed_rows:
        rel = r["rel_path"]
        rm = runs_metrics.get(rel, {})
        lines.append(f"### {r['run']}（`{rel}`）")
        lines.append("")
        lines.append(
            f"- 改动文件：`{', '.join(files)}`；配置变更键：`{r.get('cfg_change_count', 0)}`；快照：`{r.get('baseline_snapshot')} -> {r.get('last_snapshot')}`。"
        )
        lines.append(
            f"- 指标：style=`{_fmt(rm.get('best_style'))}`，cls=`{_fmt(rm.get('best_cls'))}`，lpips=`{_fmt(rm.get('lpips'))}`，"
            f"eval_count=`{_fmt(rm.get('eval_count'), 0)}`，history_rounds=`{rm.get('history_rounds', 0)}`。"
        )
        notes = deep_notes.get(rel, [])
        if notes:
            for n in notes:
                lines.append(f"- {n}")
        else:
            lines.append("- 该改动缺少可比 A/B 对照，建议后续做冻结代码的单因素回放。")
        lines.append("")

    lines.append("## 4) 总结：哪些改动在拉高风格，哪些在改善内容")
    lines.append("")
    lines.append("- 明显拉高 style 的单点改动：`v5 -> v5-style_back` 的 distill 聚合策略（style +0.0857），但 lpips 同时显著上升。")
    lines.append("- 更偏内容保持的改动：`v3 -> v4` 这类加大 cycle/edge 约束的迁移，通常降低 lpips，但可能牺牲 style/cls。")
    lines.append("- 风格与识别双高但漂移大的代表：`overfit50-style-force-balance-v1`、`full_300_distill_low_only_v1`（二者 lpips 都偏高）。")
    lines.append("- 目前更接近平衡点的路线：`overfit50-distill_low_only` 与 `overfit50-strok-style` 系（style>0.516、cls>=0.93，lpips 中低）。")
    lines.append("")

    lines.append("## 5) 下一步计划（可执行）")
    lines.append("")
    lines.append("### 5.1 目标与门槛")
    lines.append("")
    lines.append("- 目标：在 `style >= 0.53` 前提下，把 `lpips` 压到 `<= 0.50`，同时 `cls >= 0.85`。")
    lines.append("- 统一评测：固定 `eval_count=50`、完整矩阵、输出 `summary.json + summary_history.json`。")
    lines.append("")
    lines.append("### 5.2 三阶段实验计划")
    lines.append("")
    lines.append("1. 阶段A（复现实证，1-2天）")
    lines.append("- 复现 `overfit50-distill_low_only` 与 `overfit50-strok-style`，冻结代码，只验证结果稳定性。")
    lines.append("- 同时复现 `full_300_distill_low_only_v1`，确认其 high-style/high-lpips 是否稳定可重现。")
    lines.append("")
    lines.append("2. 阶段B（单因素改动，2-3天）")
    lines.append("- 在 `overfit50-distill_low_only` 基线上只扫以下单因子：`w_cycle`、`w_struct`、`w_stroke_gram`、`w_color_moment`。")
    lines.append("- 每次只改一个参数组，记录 style/cls/lpips 的方向性，建立“参数-指标灵敏度表”。")
    lines.append("")
    lines.append("3. 阶段C（组合收敛，2天）")
    lines.append("- 选择阶段B里最优 2-3 组组合，做 50 epoch 完整评测。")
    lines.append("- 只保留同时满足门槛（style/cls/lpips）的候选进入最终报告。")
    lines.append("")

    lines.append("### 5.3 风险控制")
    lines.append("")
    lines.append("- 禁止在同一轮同时改 `model.py + losses.py + 关键权重`，避免再次出现不可归因的耦合改动。")
    lines.append("- 所有 run 强制保存首中末快照，确保后续可以做“基线->最终”的可追溯分析。")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
