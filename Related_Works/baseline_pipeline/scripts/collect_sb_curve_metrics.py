from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--title", type=str, default="SB checkpoint curve")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for d in sorted(args.root.glob("step_*")):
        m = re.search(r"(\d+)", d.name)
        if not m:
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        overview = data.get("analysis", {}).get("all_pairs_overview", {}) or {}
        rows.append(
            {
                "step": int(m.group(1)),
                "clip_style": _to_float(overview.get("clip_style")),
                "content_lpips": _to_float(overview.get("content_lpips")),
                "clip_content": _to_float(overview.get("clip_content")),
                "art_fid": _to_float(overview.get("art_fid")),
                "art_fid_fid": _to_float(overview.get("art_fid_fid")),
                "art_fid_content_lpips": _to_float(overview.get("art_fid_content_lpips")),
                "summary": str(summary_path),
            }
        )
    rows.sort(key=lambda r: int(r["step"]))
    if not rows:
        raise RuntimeError(f"No step_*/summary.json files found under {args.root}")

    csv_path = args.root / "sb_curve_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "sb_curve_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lpips_rows = [r for r in rows if r["content_lpips"] is not None and r["clip_style"] is not None]
    if lpips_rows:
        plt.figure(figsize=(8, 5.8), dpi=170)
        xs = [float(r["content_lpips"]) for r in lpips_rows]
        ys = [float(r["clip_style"]) for r in lpips_rows]
        plt.plot(xs, ys, marker="o", lw=1.8)
        for r, x, y in zip(lpips_rows, xs, ys):
            plt.annotate(str(int(r["step"]) // 1000), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        plt.xlabel("content LPIPS down")
        plt.ylabel("CLIP-style up")
        plt.title(args.title)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(args.root / "sb_clip_lpips_curve.png")
        plt.close()

    art_rows = [r for r in rows if r["art_fid"] is not None]
    if art_rows:
        plt.figure(figsize=(8, 5.0), dpi=170)
        plt.plot([int(r["step"]) for r in art_rows], [float(r["art_fid"]) for r in art_rows], marker="o", lw=1.8)
        plt.xlabel("step")
        plt.ylabel("ArtFID down")
        plt.title(f"{args.title} ArtFID")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(args.root / "sb_artfid_curve.png")
        plt.close()

    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    for r in rows:
        art = r["art_fid"]
        art_text = "None" if art is None else f"{float(art):.4f}"
        print(
            f"{int(r['step']):06d} "
            f"clip_style={float(r['clip_style'] or 0):.6f} "
            f"lpips={float(r['content_lpips'] or 0):.6f} "
            f"art_fid={art_text}"
        )

    if len(rows) >= 3:
        tail = rows[-3:]
        if all(r["clip_style"] is not None and r["content_lpips"] is not None for r in tail):
            ds = float(tail[-1]["clip_style"]) - float(tail[0]["clip_style"])
            dl = float(tail[-1]["content_lpips"]) - float(tail[0]["content_lpips"])
            print(f"tail3_delta clip_style={ds:.6f} content_lpips={dl:.6f}")
        if all(r["art_fid"] is not None and not math.isnan(float(r["art_fid"])) for r in tail):
            da = float(tail[-1]["art_fid"]) - float(tail[0]["art_fid"])
            print(f"tail3_delta art_fid={da:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
