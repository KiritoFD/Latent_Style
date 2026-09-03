from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exp" / "analysis" / "tokenizer_spiral_20260528"


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return None if math.isnan(out) else out


def _read_registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = ROOT / "docs" / "logs" / "tokenizer_spiral_experiment_registry.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            clip = _float(row.get("clip_style"))
            lpips = _float(row.get("content_lpips"))
            if clip is None or lpips is None:
                continue
            rows.append(
                {
                    "id": row.get("id", ""),
                    "family": row.get("family", ""),
                    "status": row.get("status", ""),
                    "clip_style": clip,
                    "content_lpips": lpips,
                    "hayao_clip_style": _float(row.get("hayao_clip_style")),
                    "hayao_content_lpips": _float(row.get("hayao_content_lpips")),
                    "visual_gate": row.get("visual_gate", ""),
                    "verified_claim": row.get("verified_claim", ""),
                    "adjustment": row.get("adjustment", ""),
                    "artifacts": row.get("artifacts", ""),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _save_pareto(rows: list[dict[str, Any]]) -> None:
    colors = {
        "active_anchor": "#1f77b4",
        "style_adapter": "#1f77b4",
        "tokenizer_adain_gate": "#2ca02c",
        "tokenizer_g56": "#17becf",
        "tokenizer_texton_carrier": "#e377c2",
        "style_memory_bank_adapter": "#ff7f0e",
        "local_route_memory_bank": "#ff9896",
        "router_aware_backbone": "#d62728",
        "memory_residual_backbone": "#a55194",
        "stat_vocab": "#9467bd",
        "stat_reader": "#8c564b",
        "factorized_tokenizer": "#7f7f7f",
        "tokenizer_bandgate": "#bcbd22",
        "reference": "#d62728",
        "target": "#ff7f0e",
    }
    plt.figure(figsize=(9, 6))
    for row in rows:
        family = str(row.get("family", ""))
        marker = "X" if family in {"reference", "target"} else ("s" if "rejected" in str(row.get("status", "")) else "o")
        size = 120 if family in {"reference", "target"} else 55
        alpha = 0.45 if family == "factorized_tokenizer" else 0.95
        plt.scatter(
            row["content_lpips"],
            row["clip_style"],
            s=size,
            c=colors.get(family, "#333333"),
            marker=marker,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.6,
        )
        if row["id"] in {
            "m02_embspatial_highpass_style",
            "ag02_m02_g56_texture_anchor",
            "ag03_m02_g56_texture_push",
            "tc00_m02_texton_carrier_anchor",
            "tc01_m02_texton_carrier_push",
            "ra00_route_actuator_s45_e2",
            "rs00_memory_residual_s22_e2",
            "rs01_memory_residual_hp_s32_e2",
            "SaMST_target",
            "Goal_0p73_0p50",
            "Good_0p72_0p40",
            "ema_style_vocab_factorized_w36",
        }:
            plt.annotate(row["id"].replace("_", "\n"), (row["content_lpips"], row["clip_style"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.axhline(0.73, color="#d62728", linestyle="--", linewidth=1, label="clip_style 0.73")
    plt.axhline(0.72, color="#ff7f0e", linestyle=":", linewidth=1, label="SaMST/style threshold")
    plt.axvline(0.50, color="#777777", linestyle="--", linewidth=1, label="LPIPS 0.50")
    plt.axvline(0.47, color="#999999", linestyle=":", linewidth=1, label="LPIPS 0.47")
    plt.gca().invert_xaxis()
    plt.xlabel("content_lpips (lower is better; axis reversed)")
    plt.ylabel("clip_style (higher is better)")
    plt.title("Tokenizer Spiral: Style vs Content Pareto")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "tokenizer_pareto_scatter.png", dpi=220)
    plt.close()


def _save_m02_delta(rows: list[dict[str, Any]]) -> None:
    close_ids = [
        "m02_embspatial_highpass_style",
        "ag00_m02_safe_gate",
        "ag01_m02_style_gate",
        "sv00_stat_m02_conservative",
        "sv01_stat_m02_balanced",
        "sr00_stat_reader_safe",
        "sr01_stat_reader_style",
        "ag02_m02_g56_texture_anchor",
        "ag03_m02_g56_texture_push",
        "tc00_m02_texton_carrier_anchor",
        "tc01_m02_texton_carrier_push",
        "bm00_hightex_k4_blend65",
        "bm01_diverse_k4_blend65",
        "bm02_hightex_k4_boost_blend75",
        "br00_route_hightex_k4_s45",
        "br01_route_hightex_k4_s65",
        "ra00_route_actuator_s45_e2",
        "rs00_memory_residual_s22_e2",
        "rs01_memory_residual_hp_s32_e2",
    ]
    close = [row for row in rows if row["id"] in close_ids]
    plt.figure(figsize=(10, 5.5))
    xs = list(range(len(close)))
    plt.bar([i - 0.18 for i in xs], [row.get("delta_clip_vs_m02") or 0.0 for row in close], width=0.36, label="delta clip_style vs m02", color="#1f77b4")
    plt.bar([i + 0.18 for i in xs], [-(row.get("delta_lpips_vs_m02") or 0.0) for row in close], width=0.36, label="-delta LPIPS vs m02", color="#2ca02c")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(xs, [row["id"].replace("_m02_", "_").replace("_", "\n") for row in close], fontsize=8)
    plt.ylabel("delta (positive is better)")
    plt.title("Close-family deltas around m02 anchor")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "m02_family_delta_bar.png", dpi=220)
    plt.close()


def _save_hayao(rows: list[dict[str, Any]]) -> None:
    hayao = [row for row in rows if row.get("hayao_clip_style") is not None]
    hayao.sort(key=lambda row: row["hayao_clip_style"], reverse=True)
    plt.figure(figsize=(10, 5.5))
    colors = [
        "#17becf" if row["family"] == "tokenizer_g56"
        else "#e377c2" if row["family"] == "tokenizer_texton_carrier"
        else "#8c564b" if row["family"] == "stat_reader"
        else "#2ca02c"
        for row in hayao
    ]
    plt.bar(range(len(hayao)), [row["hayao_clip_style"] for row in hayao], color=colors)
    plt.xticks(range(len(hayao)), [row["id"].replace("_", "\n") for row in hayao], fontsize=8)
    plt.ylabel("Hayao cross clip_style")
    plt.title("Hayao remains the weakest diagnostic slice")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "hayao_cross_clip_bar.png", dpi=220)
    plt.close()


def _save_operator_binding() -> None:
    path = ROOT / "exp" / "diagnostics" / "m02_operator_binding_g56" / "operator_binding_rows.csv"
    op_rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            op_rows = [row for row in csv.DictReader(f) if row.get("kind") == "local_perturbation"]
    agg: dict[str, dict[str, Any]] = {}
    for row in op_rows:
        label = row.get("label", "")
        value = abs(float(row.get("endpoint_delta_rms") or 0))
        if label not in agg or value > agg[label]["endpoint_delta_rms"]:
            agg[label] = {
                "label": label,
                "endpoint_delta_rms": value,
                "mid_delta_rms": float(row.get("mid_delta_rms") or 0),
                "high_fraction": float(row.get("high_fraction") or 0),
            }
    summary = sorted(agg.values(), key=lambda item: item["endpoint_delta_rms"], reverse=True)
    _write_csv(OUT / "operator_binding_response_summary.csv", summary, ["label", "endpoint_delta_rms", "mid_delta_rms", "high_fraction"])
    if not summary:
        return
    plt.figure(figsize=(8.5, 5))
    labels = [row["label"] for row in summary]
    values = [row["endpoint_delta_rms"] for row in summary]
    colors = ["#17becf" if "grammar" in label else "#2ca02c" for label in labels]
    plt.bar(range(len(labels)), values, color=colors)
    plt.xticks(range(len(labels)), [label.replace("_", "\n") for label in labels], fontsize=8)
    plt.ylabel("max endpoint delta RMS")
    plt.title("g56 operator binding: executable fields")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "operator_binding_response.png", dpi=220)
    plt.close()


def _save_cross_target() -> None:
    per_target: list[dict[str, Any]] = []
    base = ROOT / "exp" / "tokenizer_adain_gate_calibration"
    runs = ["ag02_m02_g56_texture_anchor", "ag03_m02_g56_texture_push"]
    for run in runs:
        path = base / run / "full_eval" / "summary_reuse_generated.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for target, metrics in (payload.get("analysis", {}).get("cross_by_target_style", {}) or {}).items():
            per_target.append({"run": run, "target": target, "clip_style": metrics.get("clip_style"), "content_lpips": metrics.get("content_lpips")})
    _write_csv(OUT / "ag02_ag03_cross_by_target.csv", per_target, ["run", "target", "clip_style", "content_lpips"])
    if not per_target:
        return
    targets = sorted({row["target"] for row in per_target})
    plt.figure(figsize=(8, 5))
    width = 0.35
    for idx, run in enumerate(runs):
        values = []
        for target in targets:
            match = next((row for row in per_target if row["run"] == run and row["target"] == target), None)
            values.append(float(match["clip_style"]) if match and match.get("clip_style") is not None else 0.0)
        plt.bar([i + (idx - 0.5) * width for i in range(len(targets))], values, width=width, label=run.replace("ag0", "ag0 "))
    plt.xticks(range(len(targets)), targets)
    plt.ylabel("cross-by-target clip_style")
    plt.title("ag02/ag03 per-target style: Hayao is still weak")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "ag02_ag03_cross_target_bar.png", dpi=220)
    plt.close()


def _write_report(rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda row: (row["clip_style"], -row["content_lpips"]), reverse=True)
    lines = [
        "# Tokenizer Spiral Summary - 2026-05-28",
        "",
        "## Key Data",
        "",
        "| id | family | status | clip_style | LPIPS | Hayao style | delta clip vs m02 | delta LPIPS vs m02 | verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows_sorted:
        if row["family"] in {"sensitivity"}:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    row["family"],
                    row["status"],
                    f"{row['clip_style']:.6f}",
                    f"{row['content_lpips']:.6f}",
                    "" if row.get("hayao_clip_style") is None else f"{row['hayao_clip_style']:.6f}",
                    "" if row.get("delta_clip_vs_m02") is None else f"{row['delta_clip_vs_m02']:+.6f}",
                    "" if row.get("delta_lpips_vs_m02") is None else f"{row['delta_lpips_vs_m02']:+.6f}",
                    str(row.get("adjustment", "")).replace("|", "/")[:90],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `tokenizer_pareto_scatter.png`",
        "- `m02_family_delta_bar.png`",
        "- `hayao_cross_clip_bar.png`",
        "- `operator_binding_response.png`",
        "- `ag02_ag03_cross_target_bar.png`",
        "",
        "## Analysis",
        "",
        "- Best global style among the current tokenizer spiral rows is `bg00_band_anchor` at `0.71289 / 0.44403`, but it is a different texton-band route and still below the `0.72-0.73` target.",
        "- The active m02 family is flat around `clip_style ~= 0.7105-0.7110` and `LPIPS ~= 0.4026-0.4074`; the recent g56 runs do not break out of this plateau.",
        "- `ag02` remains the best m02-family tokenizer result by style: `0.710955 / 0.407269`, Hayao `0.605668`; the gain over m02 is only `+0.000225`, so it is evidence of stability, not a performance win.",
        "- `sr01` gives the best LPIPS in the safe m02 family (`0.402585`) but loses style, confirming that LPIPS can improve by staying close to the anchor.",
        "- g56 operator binding is diagnostically useful: `grammar_mid_texton` becomes executable (`~0.00543` endpoint RMS), while `grammar_high_texture` remains nearly dead (`~0.0002`).",
        "- The new texton-carrier tests are safe but negative: `tc00=0.710431 / 0.407304`, `tc01=0.710621 / 0.406945`; simple content-high/AdaIN-residual carriers do not beat ag02.",
        "- The id-only multi-prototype bank is also negative: `bm00=0.710854 / 0.407380`, `bm01=0.710676 / 0.407408`, `bm02=0.710693 / 0.407397`; static style-level prototype mixtures do not reproduce the reference-memory lift.",
        "- The local route-only adapter is negative as well: `br00=0.710530 / 0.407402`, `br01=0.710609 / 0.407408`; putting content-token attention before the frozen style-map actuator is still absorbed by the m02 operating region.",
        "- Router-aware training through the old style-map path is also negative: `ra00=0.710336 / 0.435838`; Hayao rises to `0.614082`, but global style and LPIPS both worsen.",
        "- Explicit memory residual injection is active but negative: `rs00=0.707073 / 0.432237`, `rs01=0.707358 / 0.429501`; untyped prototype residuals behave like average perturbations, not style semantics.",
        "- Conclusion: tokenizer values/readers, prototype availability, route-only adapter injection, and untyped residual source injection are not the main bottleneck now. The next valid route is a style-field typed prototype vocabulary or OT/contrastive prototype assignment, not another scalar high-band amplifier or strength sweep.",
    ]
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _read_registry_rows()
    refs = [
        {"id": "SaMST_target", "family": "reference", "status": "reference", "clip_style": 0.7200, "content_lpips": 0.5300, "hayao_clip_style": None, "hayao_content_lpips": None, "visual_gate": "reference", "verified_claim": "paper/user comparison point", "adjustment": "beat on both metrics", "artifacts": "Related_Works/SaMST"},
        {"id": "Goal_0p73_0p50", "family": "target", "status": "target", "clip_style": 0.7300, "content_lpips": 0.5000, "hayao_clip_style": None, "hayao_content_lpips": None, "visual_gate": "target", "verified_claim": "current target gate", "adjustment": "main objective", "artifacts": ""},
        {"id": "Good_0p72_0p40", "family": "target", "status": "target", "clip_style": 0.7200, "content_lpips": 0.4000, "hayao_clip_style": None, "hayao_content_lpips": None, "visual_gate": "target", "verified_claim": "acceptable strong Pareto region", "adjustment": "high value if achieved", "artifacts": ""},
    ]
    m02 = next((row for row in rows if row["id"] == "m02_embspatial_highpass_style"), None)
    all_rows = rows + refs
    for row in all_rows:
        if m02 is not None:
            row["delta_clip_vs_m02"] = row["clip_style"] - m02["clip_style"]
            row["delta_lpips_vs_m02"] = row["content_lpips"] - m02["content_lpips"]
        else:
            row["delta_clip_vs_m02"] = ""
            row["delta_lpips_vs_m02"] = ""
    fields = [
        "id",
        "family",
        "status",
        "clip_style",
        "content_lpips",
        "hayao_clip_style",
        "hayao_content_lpips",
        "delta_clip_vs_m02",
        "delta_lpips_vs_m02",
        "visual_gate",
        "verified_claim",
        "adjustment",
        "artifacts",
    ]
    _write_csv(OUT / "tokenizer_metrics_summary.csv", all_rows, fields)
    _save_pareto(all_rows)
    _save_m02_delta(rows)
    _save_hayao(rows)
    _save_operator_binding()
    _save_cross_target()
    _write_report(rows)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
