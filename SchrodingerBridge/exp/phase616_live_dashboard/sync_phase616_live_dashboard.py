from __future__ import annotations

import csv
import html
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


HOST = "administrator@100.115.18.62"
PORT = 2222
WSL_DISTRO = "Ubuntu-26.04"
REMOTE_SB_ROOT = "/mnt/i/Github/Latent_Style/SchrodingerBridge"

REMOTE_H0 = f"{REMOTE_SB_ROOT}/exp/20250618_lite_ot_vertical/h0_vertical_fm"
REMOTE_STAGE_ROOTS = {
    "stage1_auto": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical_auto",
    "stage2_auto": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_ablation_auto",
    "stage3_auto": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_best_auto",
    "style_sweep_auto": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_stage3_style_auto",
    "ot_rerun_auto": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_ot_rerun_lowrank_auto",
    "spatial620": "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge",
}

HERE = Path(__file__).resolve().parent
MIRROR_DIR = HERE / "mirror"
FIG_PNG = HERE / "phase616_live_pareto.png"
FIG_PDF = HERE / "phase616_live_pareto.pdf"
HTML_OUT = HERE / "phase616_live_dashboard.html"
DATA_JS_OUT = HERE / "phase616_live_data.js"
SNAPSHOT_JSON = HERE / "phase616_live_snapshot.json"
STATUS_JSON = HERE / "phase616_live_status.json"
CLIPT_MANIFEST = HERE / "clipt_representative_manifest.csv"
CLIPT_CSV = HERE / "clipt_representative_results.csv"
CLIPT_JSON = HERE / "clipt_representative_results.json"
STYLET_CKPT = HERE.parent.parent / "aaai2027" / "distinct5_convnext_style_classifier.pt"
STYLET_CSV = HERE / "stylet_representative_results.csv"
STYLET_JSON = HERE / "stylet_representative_results.json"
EXTERNAL_BASELINES_CSV = HERE / "external_baselines.csv"
OLD_GLOBAL_IDT_CLIP_STYLE = 0.639920825263

PAGE1_POINTS_CSV = (
    HERE.parent.parent / "aaai2027" / "page1_bundle" / "wikiart5_page1_clip_lpips_points.csv"
)
if not PAGE1_POINTS_CSV.is_file():
    PAGE1_POINTS_CSV = (
        HERE.parent.parent
        / "docs"
        / "experiments"
        / "phase2_fiber_bundle"
        / "616"
        / "csv"
        / "wikiart5_page1_clip_lpips_points.csv"
    )
ARTFID_CSV = (
    HERE.parent.parent
    / "docs"
    / "experiments"
    / "comparison_20260602"
    / "artfid_comparison_points.csv"
)

REMOTE_PY = f"""
import json
import subprocess
from pathlib import Path

targets = [
    {{"group": "manual", "label": "h0_vertical_fm", "run_dir": Path(r"{REMOTE_H0}")}},
]
stage_roots = {json.dumps(REMOTE_STAGE_ROOTS)}

runs = []
stage_scan = {{}}

def resolve_run_dir(run_dir: Path) -> Path:
    if (run_dir / "full_eval_transfer").is_dir() or (run_dir / "full_eval").is_dir() or (run_dir / "logs").is_dir():
        return run_dir
    repo_root = Path(r"{REMOTE_SB_ROOT}")
    text = str(run_dir)
    doubled = repo_root.as_posix() + repo_root.as_posix()
    if text.startswith(doubled):
        candidate = Path(text[len(repo_root.as_posix()):])
        if (candidate / "full_eval_transfer").is_dir() or (candidate / "full_eval").is_dir() or (candidate / "logs").is_dir():
            return candidate
    legacy = repo_root / text.lstrip("/")
    if (legacy / "full_eval_transfer").is_dir() or (legacy / "full_eval").is_dir() or (legacy / "logs").is_dir():
        return legacy
    if text.startswith(repo_root.as_posix() + "/mnt/"):
        candidate = Path(text[len(repo_root.as_posix()):])
        if (candidate / "full_eval_transfer").is_dir() or (candidate / "full_eval").is_dir() or (candidate / "logs").is_dir():
            return candidate
    return run_dir

def add_run(group, label, run_dir):
    run_dir = resolve_run_dir(run_dir)
    eval_dir = run_dir / "full_eval_transfer"
    if not (eval_dir / "clip_lpips_curve.csv").is_file():
        eval_dir = run_dir / "full_eval"
    curve = eval_dir / "clip_lpips_curve.csv"
    conv = eval_dir / "round2_convergence.json"
    mat = run_dir / "best_eval_materialization.json"
    if not curve.is_file():
        return
    runs.append({{
        "group": group,
        "label": label,
        "run_dir": str(run_dir),
        "curve_csv": curve.read_text(encoding="utf-8"),
        "convergence_json": conv.read_text(encoding="utf-8") if conv.is_file() else "",
        "best_eval_materialization_json": mat.read_text(encoding="utf-8") if mat.is_file() else "",
    }})

for target in targets:
    add_run(target["group"], target["label"], target["run_dir"])

for group, root_text in stage_roots.items():
    root = Path(root_text)
    group_scan = []
    if not root.is_dir():
        stage_scan[group] = group_scan
        continue
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        logs_dir = child / "logs"
        latest_training_csv = ""
        if logs_dir.is_dir():
            csvs = sorted(logs_dir.glob("training_*.csv"))
            if csvs:
                latest_training_csv = str(csvs[-1])
        group_scan.append({{
            "label": child.name,
            "run_dir": str(child),
            "has_curve": (child / "full_eval_transfer" / "clip_lpips_curve.csv").is_file() or (child / "full_eval" / "clip_lpips_curve.csv").is_file(),
            "has_auto_summary": (child / "auto_run_summary.json").is_file(),
            "has_config": (child / "config.json").is_file(),
            "latest_training_csv": latest_training_csv,
        }})
        add_run(group, child.name, child)
    stage_scan[group] = group_scan

gpu = {{}}
try:
    smi = "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").is_file() else "nvidia-smi"
    proc = subprocess.run(
        [smi, "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    gpu["lines"] = [line for line in proc.stdout.splitlines() if line.strip()]
except Exception:
    gpu["lines"] = []

ps_lines = []
try:
    proc = subprocess.run(["ps", "-eo", "pid,ppid,etimes,cmd"], text=True, capture_output=True, check=False)
    ps_lines = [
        line for line in proc.stdout.splitlines()
        if "phase616_auto.py" in line or "20250618_lite_ot_" in line or "620_spatial_bridge" in line or "src/run.py" in line or "run_phase4_plan.sh" in line
    ]
except Exception:
    ps_lines = []

active_log_lines = []
active_log_path = ""
try:
    import os, time
    search_dirs = [
        "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp",
        "/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/616/logs",
        "/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620"
    ]
    candidates = []
    for s_dir in search_dirs:
        if os.path.isdir(s_dir):
            for root, dirs, files in os.walk(s_dir):
                for file in files:
                    if file.endswith(".log") or file.endswith(".jsonl") or file.endswith("run.log"):
                        path = os.path.join(root, file)
                        try:
                            mtime = os.path.getmtime(path)
                            if time.time() - mtime < 7200:
                                candidates.append((mtime, path))
                        except Exception:
                            pass
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        active_log_path = candidates[0][1]
        with open(active_log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            active_log_lines = [line.rstrip() for line in lines[-40:]]
except Exception as e:
    active_log_lines = [f"Error scanning active logs: " + str(e)]

print(json.dumps({{"runs": runs, "stage_scan": stage_scan, "gpu": gpu, "ps_lines": ps_lines, "active_log_lines": active_log_lines, "active_log_path": active_log_path}}, ensure_ascii=False))
"""

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.4,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.6,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.65,
        "lines.markersize": 4.6,
    }
)

COLORS = {
    "manual": "#D64045",
    "stage1_auto": "#5BC0EB",
    "stage2_auto": "#2CA58D",
    "stage3_auto": "#7A5CFA",
    "style_sweep_auto": "#1D9BF0",
    "ot_rerun_auto": "#E85D75",
    "spatial620": "#D946EF",
    "idt": "#8E63C0",
    "samam": "#2F7DB7",
    "seedream": "#E48F1C",
    "samst": "#2CA02C",
    "stylegallery": "#F97316",
    "styleshot": "#14B8A6",
    "csgo_low_vram": "#64748B",
    "legacy": "#6A6A6A",
    "panel_bg": "#FCFBF8",
    "grid": "#D7D2CC",
    "text": "#2A2A2A",
    "muted": "#5F6B74",
    # Pre-registered colors for h0-h15 prefix
    "h0": "#D64045",      # Red
    "h1": "#5BC0EB",      # Blue
    "h2": "#2CA58D",      # Emerald Green
    "h3": "#7A5CFA",      # Violet/Purple
    "h4": "#F5A623",      # Orange
    "h5": "#EC4899",      # Pink
    "h6": "#00B4D8",      # Cyan/Teal
    "h7": "#84CC16",      # Lime Green
    "h8": "#FF6B6B",      # Coral Red
    "h9": "#4DABF7",      # Sky Blue
    "h10": "#37B24D",     # Forest Green
    "h11": "#F783AC",     # Pastel Pink
    "h12": "#AE3EC9",     # Magenta/Purple
    "h13": "#F76707",     # Rust Orange
    "h14": "#1098AD",     # Dark Cyan
    "h15": "#7048E8",     # Indigo
}

EXTERNAL_BASELINES = [
    {
        "id": "stylegallery",
        "label": "StyleGallery",
        "images": 750,
        "clip_style": 0.697547,
        "lpips": 0.710688,
        "label_dx": -16.0,
        "label_dy": 12.0,
    },
    {
        "id": "styleshot",
        "label": "StyleShot",
        "images": 750,
        "clip_style": 0.806562,
        "lpips": 0.698320,
        "label_dx": 10.0,
        "label_dy": -14.0,
    },
    {
        "id": "csgo_low_vram",
        "label": "CSGO low-VRAM",
        "images": 750,
        "clip_style": 0.654125,
        "lpips": 0.820927,
        "label_dx": 10.0,
        "label_dy": 14.0,
    },
]


@dataclass
class RunPoint:
    group: str
    label: str
    epoch: str
    epoch_int: int
    timestamp: str
    style: float
    lpips: float
    x: float
    clip_t: float | None = None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "run"


def _normalize_remote_wsl_path(path_text: str) -> str:
    text = str(path_text or "").strip().replace("\\", "/")
    if not text:
        return text
    duplicate = f"{REMOTE_SB_ROOT}{REMOTE_SB_ROOT}"
    if text.startswith(duplicate):
        text = text[len(REMOTE_SB_ROOT):]
    doubled_marker = f"{REMOTE_SB_ROOT}/mnt/"
    if text.startswith(doubled_marker):
        text = text[len(REMOTE_SB_ROOT):]
    idx = text.find("/mnt/")
    if idx > 0 and text.startswith(REMOTE_SB_ROOT):
        text = text[idx:]
    return text


def _fetch_remote_snapshot() -> dict:
    cmd = [
        "ssh",
        "-p",
        str(PORT),
        "-o",
        "LogLevel=ERROR",
        HOST,
        "wsl",
        "-d",
        WSL_DISTRO,
        "python3",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        input=REMOTE_PY,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"ssh failed rc={proc.returncode}")
    payload = str(proc.stdout or "").strip()
    if not payload:
        raise RuntimeError("empty remote snapshot payload")
    return json.loads(payload)


def _write_mirror(snapshot: dict) -> None:
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    for run in snapshot.get("runs", []):
        run_key = _slug(f"{run['group']}__{run['label']}")
        run_dir = MIRROR_DIR / run_key
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "clip_lpips_curve.csv").write_text(run["curve_csv"], encoding="utf-8")
        if run.get("convergence_json"):
            (run_dir / "round2_convergence.json").write_text(run["convergence_json"], encoding="utf-8")
        if run.get("best_eval_materialization_json"):
            (run_dir / "best_eval_materialization.json").write_text(
                run["best_eval_materialization_json"],
                encoding="utf-8",
            )
        (run_dir / "meta.json").write_text(
            json.dumps(
                {"group": run["group"], "label": run["label"], "run_dir": run["run_dir"]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def _parse_points(snapshot: dict) -> list[RunPoint]:
    out: list[RunPoint] = []
    for run in snapshot.get("runs", []):
        reader = csv.DictReader(StringIO(run["curve_csv"]))
        for row in reader:
            if "all_pairs_clip_style" in row and str(row["all_pairs_clip_style"]).strip():
                style = float(row["all_pairs_clip_style"])
                lpips = float(row["all_pairs_content_lpips"])
                clip_t = None
                if str(row.get("all_pairs_clip_t", "")).strip():
                    clip_t = float(row["all_pairs_clip_t"])
            else:
                style = float(row["transfer_clip_style"])
                lpips = float(row["transfer_content_lpips"])
                clip_t = None
                if str(row.get("transfer_clip_t", "")).strip():
                    clip_t = float(row["transfer_clip_t"])
            out.append(
                RunPoint(
                    group=str(run["group"]),
                    label=str(run["label"]),
                    epoch=str(row["epoch"]),
                    epoch_int=int(row["epoch_int"]),
                    timestamp=str(row.get("timestamp", "") or ""),
                    style=style,
                    lpips=lpips,
                    x=1.0 - lpips,
                    clip_t=clip_t,
                )
            )
    return out


def _point_style_minus_idt(point: RunPoint, idt_style: float) -> float:
    return float(point.style) - float(idt_style)


def _series_color_key(group: str, label: str) -> str:
    prefix = label.split("_")[0] if "_" in label else label
    if prefix in COLORS:
        return prefix
    if group in COLORS:
        return group
    return "legacy"


def _wsl_to_remote_windows_path(path_text: str) -> str:
    text = _normalize_remote_wsl_path(path_text)
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if not match:
        return text
    drive = match.group(1).upper()
    rest = match.group(2).replace("\\", "/")
    return f"{drive}:/{rest}"


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def _count_local_images(images_dir: Path) -> int:
    if not images_dir.is_dir():
        return 0
    return sum(1 for p in images_dir.iterdir() if p.is_file())


def _scp_fetch_file(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", "-P", str(PORT), "-q", remote_path, str(local_path)],
        check=True,
    )


def _scp_fetch_dir(remote_path: str, local_path: Path) -> None:
    if local_path.exists():
        shutil.rmtree(local_path, ignore_errors=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", "-P", str(PORT), "-q", "-r", remote_path, str(local_path.parent)],
        check=True,
    )


def _sync_best_eval_archives(snapshot: dict) -> list[dict[str, object]]:
    synced: list[dict[str, object]] = []
    for run in snapshot.get("runs", []):
        mat_text = str(run.get("best_eval_materialization_json", "") or "").strip()
        if not mat_text:
            continue
        try:
            mat = json.loads(mat_text)
        except json.JSONDecodeError:
            continue
        archive_path = str(mat.get("archive_path", "") or "").strip()
        output_dir = str(mat.get("output_dir", "") or "").strip()
        generated_images = int(mat.get("generated_images", 0) or 0)
        best_epoch = str(mat.get("best_epoch", "") or "").strip()
        if not output_dir or generated_images <= 0 or not best_epoch:
            continue

        best_epoch_int = int(mat.get("best_epoch_int", 0) or 0)
        label = str(run.get("label", "") or "").strip()
        slug = _slug(label.split("_")[0]) if label else "run"
        local_name = f"{slug}_e{best_epoch_int}_eval"
        local_dir = HERE / local_name
        local_archive = HERE / f"{local_name}.tgz"
        local_meta = local_dir / "_materialization_meta.json"

        needs_sync = True
        if local_meta.is_file():
            try:
                old = json.loads(local_meta.read_text(encoding="utf-8"))
                needs_sync = (
                    old.get("archive_path") != archive_path
                    or int(old.get("generated_images", 0) or 0) != generated_images
                    or _count_local_images(local_dir / "images") < generated_images
                )
            except Exception:
                needs_sync = True

        if needs_sync:
            synced_via = ""
            remote_images_dir = f"{HOST}:{_wsl_to_remote_windows_path(output_dir + '/images')}"
            if archive_path:
                remote_archive = f"{HOST}:{_wsl_to_remote_windows_path(archive_path)}"
                try:
                    _scp_fetch_file(remote_archive, local_archive)
                    _extract_archive(local_archive, local_dir)
                    synced_via = "archive"
                except Exception:
                    if local_archive.exists():
                        local_archive.unlink()
            if not synced_via:
                _scp_fetch_dir(remote_images_dir, local_dir / "images")
                synced_via = "directory"
            remote_metrics = f"{HOST}:{_wsl_to_remote_windows_path(output_dir + '/metrics.csv')}"
            _scp_fetch_file(remote_metrics, local_dir / "metrics.csv")
            local_meta.write_text(
                json.dumps(
                    {
                        "group": run.get("group"),
                        "label": label,
                        "best_epoch": best_epoch,
                        "best_epoch_int": best_epoch_int,
                        "generated_images": generated_images,
                        "archive_path": archive_path,
                        "output_dir": output_dir,
                        "synced_via": synced_via,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        synced.append(
            {
                "group": run.get("group"),
                "label": label,
                "best_epoch": best_epoch,
                "best_epoch_int": best_epoch_int,
                "images_dir": str(local_dir / "images"),
                "metrics_csv": str(local_dir / "metrics.csv"),
                "generated_images": generated_images,
                "synced_via": json.loads(local_meta.read_text(encoding="utf-8")).get("synced_via", "unknown")
                if local_meta.is_file()
                else "unknown",
            }
        )
    return synced


def _write_clipt_manifest(synced_archives: list[dict[str, object]]) -> None:
    rows = []
    for item in synced_archives:
        label = str(item.get("label", "") or "")
        run_name = str(item.get("best_epoch", "") or label)
        if label.startswith("h0_"):
            run_id = "h0_best"
        elif label.startswith("h1_"):
            run_id = "h1_best"
        else:
            run_id = f"{label}_{item.get('best_epoch_int', 0)}"
        rows.append(
            {
                "method": "Phase616",
                "run": run_id,
                "images_dir": str(item["images_dir"]),
                "metrics_csv": str(item["metrics_csv"]),
                "source_root": r"F:\wikiart_distinct5_samam_512_classview_real\test",
                "generated_mode": "generated",
            }
        )
    if not rows:
        return
    with CLIPT_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "run", "images_dir", "metrics_csv", "source_root", "generated_mode"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _latest_by_run(points: list[RunPoint]) -> list[RunPoint]:
    latest: dict[tuple[str, str], RunPoint] = {}
    for point in points:
        key = (point.group, point.label)
        prev = latest.get(key)
        if prev is None or point.epoch_int > prev.epoch_int:
            latest[key] = point
    return sorted(latest.values(), key=lambda p: (p.group, p.label))


def _latest_point(
    points: list[RunPoint], predicate: Callable[[RunPoint], bool] | None = None
) -> RunPoint | None:
    candidates = points if predicate is None else [point for point in points if predicate(point)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.timestamp, p.epoch_int, p.group, p.label))


def _best_manual(points: list[RunPoint]) -> RunPoint | None:
    manual = [p for p in points if p.group == "manual"]
    if not manual:
        return None
    return max(manual, key=lambda p: (p.style, -p.lpips))


def _load_page1_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with PAGE1_POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != "transfer":
                continue
            rows.append(
                {
                    "point_id": row["point_id"] if "point_id" in row else row.get("id", ""),
                    "family": row["family"],
                    "label": row.get("label") or row.get("variant") or row["family"],
                    "trace_id": row.get("trace_id", ""),
                    "step_or_epoch": row.get("step_or_epoch", ""),
                    "clip_style": float(row["clip_style"]),
                    "lpips": float(row["content_lpips"]),
                    "x": float(row["one_minus_lpips"]),
                    "style_minus_idt": float(row.get("style_minus_idt") or 0.0),
                    "label_dx": float(row.get("label_dx") or 0.0),
                    "label_dy": float(row.get("label_dy") or 0.0),
                    "note": row.get("note", ""),
                }
            )
    return rows


def _load_artfid_rows() -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    if not ARTFID_CSV.is_file():
        return rows
    with ARTFID_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "distinct5_512" or row["scope"] != "transfer":
                continue
            rows[(row["method"], row["label"])] = {
                "method": row["method"],
                "label": row["label"],
                "artfid": float(row["aggregate_art_fid"]),
                "train_time_label": row["train_time_label"],
            }
    return rows


def _load_external_baselines() -> list[dict[str, object]]:
    if not EXTERNAL_BASELINES_CSV.is_file():
        return [dict(row) for row in EXTERNAL_BASELINES]
    rows: list[dict[str, object]] = []
    with EXTERNAL_BASELINES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "id": str(row["id"]),
                        "label": str(row["label"]),
                        "images": int(row["images"]),
                        "clip_style": float(row["clip_style"]),
                        "lpips": float(row["lpips"]),
                        "label_dx": float(row.get("label_dx") or 0.0),
                        "label_dy": float(row.get("label_dy") or 0.0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows or [dict(row) for row in EXTERNAL_BASELINES]


def _build_baseline_bundle() -> dict[str, object]:
    page1 = _load_page1_rows()
    artfid = _load_artfid_rows()
    idt = next(r for r in page1 if r["family"] == "IDT")
    seedream = next(r for r in page1 if r["family"] == "Seedream")
    samam = [r for r in page1 if r["family"] == "SaMAM"]
    samst = [r for r in page1 if r["family"] == "SaMST"]
    legacy = [r for r in page1 if r["family"] in {"LANCET", "LBM"}]
    idt["clip_style"] = OLD_GLOBAL_IDT_CLIP_STYLE
    idt["style_minus_idt"] = 0.0
    idt_clip = OLD_GLOBAL_IDT_CLIP_STYLE
    for row in [seedream, *samam, *samst, *legacy]:
        row["style_minus_idt"] = float(row["clip_style"]) - idt_clip
    external_points = []
    for row in _load_external_baselines():
        external_points.append(
            {
                "id": str(row["id"]),
                "label": str(row["label"]),
                "images": int(row["images"]),
                "clip_style": float(row["clip_style"]),
                "lpips": float(row["lpips"]),
                "x": 1.0 - float(row["lpips"]),
                "style_minus_idt": float(row["clip_style"]) - idt_clip,
                "label_dx": float(row["label_dx"]),
                "label_dy": float(row["label_dy"]),
            }
        )

    return {
        "idt": idt,
        "seedream": seedream,
        "samam_curve": samam,
        "samst_curve": samst,
        "external_points": external_points,
        "legacy_points": legacy,
        "artfid": artfid,
    }


def _manifest_has_materialized_images(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    if not rows:
        return False
    for row in rows:
        images_dir = Path(str(row.get("images_dir", "") or ""))
        if not images_dir.is_dir():
            return False
        if not any(p.is_file() for p in images_dir.iterdir()):
            return False
    return True


def _maybe_refresh_clipt_summary() -> None:
    if not _manifest_has_materialized_images(CLIPT_MANIFEST):
        return
    if not Path(r"F:\wikiart_distinct5_samam_512_classview_real\test").exists():
        return
    needs_refresh = not CLIPT_JSON.is_file() or CLIPT_JSON.stat().st_mtime < CLIPT_MANIFEST.stat().st_mtime
    if not needs_refresh:
        return
    cmd = [
        sys.executable,
        str(HERE.parent.parent / "tools" / "eval_clip_text_probe.py"),
        "--manifest",
        str(CLIPT_MANIFEST),
        "--output-csv",
        str(CLIPT_CSV),
        "--output-json",
        str(CLIPT_JSON),
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"clipt rc={proc.returncode}")


def _maybe_refresh_stylet_summary() -> None:
    if not _manifest_has_materialized_images(CLIPT_MANIFEST) or not STYLET_CKPT.is_file():
        return
    if not Path(r"F:\wikiart_distinct5_samam_512_classview_real\test").exists():
        return
    needs_refresh = (
        not STYLET_JSON.is_file()
        or STYLET_JSON.stat().st_mtime < CLIPT_MANIFEST.stat().st_mtime
        or STYLET_JSON.stat().st_mtime < STYLET_CKPT.stat().st_mtime
    )
    if not needs_refresh:
        return
    cmd = [
        sys.executable,
        str(HERE.parent.parent / "tools" / "eval_style_text_probe.py"),
        "--classifier-ckpt",
        str(STYLET_CKPT),
        "--manifest",
        str(CLIPT_MANIFEST),
        "--output-csv",
        str(STYLET_CSV),
        "--output-json",
        str(STYLET_JSON),
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"stylet rc={proc.returncode}")


def _load_clipt_rows() -> list[dict[str, object]]:
    if not CLIPT_JSON.is_file():
        return []
    rows = json.loads(CLIPT_JSON.read_text(encoding="utf-8"))
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "method": str(row.get("method", "")),
                "run": str(row.get("run", "")),
                "images": int(row.get("images", 0) or 0),
                "transfer_target_acc": float(row.get("transfer_target_acc") or 0.0),
                "identity_source_acc": float(row.get("identity_source_acc") or 0.0),
                "transfer_target_text": float(row.get("transfer_target_text") or 0.0),
                "transfer_source_text": float(row.get("transfer_source_text") or 0.0),
                "transfer_target_source_margin": float(row.get("transfer_target_source_margin") or 0.0),
                "all_pairs_target_source_margin": float(row.get("all_pairs_target_source_margin") or 0.0),
                "prompt_template": str(row.get("prompt_template", "")),
            }
        )
    return out


def _load_stylet_rows() -> list[dict[str, object]]:
    if not STYLET_JSON.is_file():
        return []
    rows = json.loads(STYLET_JSON.read_text(encoding="utf-8"))
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "method": str(row.get("method", "")),
                "run": str(row.get("run", "")),
                "images": int(row.get("images", 0) or 0),
                "transfer_target_acc": float(row.get("transfer_target_acc") or 0.0),
                "identity_source_acc": float(row.get("identity_source_acc") or 0.0),
                "transfer_target_prob": float(row.get("transfer_target_prob") or 0.0),
                "transfer_source_prob": float(row.get("transfer_source_prob") or 0.0),
                "transfer_target_source_margin": float(row.get("transfer_target_source_margin") or 0.0),
                "all_pairs_target_source_margin": float(row.get("all_pairs_target_source_margin") or 0.0),
            }
        )
    return out


def _annotate(ax, x: float, y: float, text: str, dx: float, dy: float, color: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=7.0,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.5, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=2, shrinkB=3),
        annotation_clip=True,
    )


def _compute_pareto_viewport(points: list[RunPoint], baselines: dict[str, object]) -> dict[str, float]:
    idt_clip = float(baselines["idt"]["clip_style"])
    xs = [p.x for p in points]
    ys = [_point_style_minus_idt(p, idt_clip) for p in points]

    def _extend(rows: list[dict[str, object]]) -> None:
        for row in rows:
            xs.append(float(row["x"]))
            ys.append(float(row["style_minus_idt"]))

    xs.append(float(baselines["seedream"]["x"]))
    ys.append(float(baselines["seedream"]["style_minus_idt"]))
    _extend(list(baselines["samam_curve"]))
    _extend(list(baselines["samst_curve"]))
    _extend(list(baselines.get("external_points", [])))

    def _with_padding(values: list[float], pad_frac: float, min_pad: float) -> tuple[float, float]:
        lo = min(values)
        hi = max(values)
        span = max(hi - lo, 1e-6)
        pad = max(span * pad_frac, min_pad)
        return lo - pad, hi + pad

    x_min, x_max = _with_padding(xs, pad_frac=0.06, min_pad=0.02)
    y_min, y_max = _with_padding(ys, pad_frac=0.10, min_pad=0.01)

    x_min = max(0.0, math.floor(x_min * 100.0) / 100.0)
    x_max = min(1.0, math.ceil(x_max * 100.0) / 100.0)
    y_min = math.floor(y_min * 100.0) / 100.0
    y_max = math.ceil(y_max * 100.0) / 100.0
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def _plot_png(points: list[RunPoint], baselines: dict[str, object]) -> None:
    if not points:
        raise RuntimeError("No live points available")

    latest = _latest_by_run(points)
    best_manual = _best_manual(points)
    by_run: dict[tuple[str, str], list[RunPoint]] = {}
    for point in points:
        by_run.setdefault((point.group, point.label), []).append(point)
    for series in by_run.values():
        series.sort(key=lambda p: p.epoch_int)
    viewport = _compute_pareto_viewport(points, baselines)

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.9), gridspec_kw={"width_ratios": [1.28, 0.92]})

    ax = axes[0]
    ax.set_facecolor(COLORS["panel_bg"])
    samam = baselines["samam_curve"]
    samst = baselines["samst_curve"]
    seedream = baselines["seedream"]
    idt = baselines["idt"]
    external_points = baselines.get("external_points", [])

    ax.plot([r["x"] for r in samam], [r["style_minus_idt"] for r in samam], color=COLORS["samam"], alpha=0.9, zorder=2)
    ax.scatter([r["x"] for r in samam], [r["style_minus_idt"] for r in samam], s=16, color=COLORS["samam"], edgecolor="white", linewidth=0.55, zorder=3)
    if samst:
        ax.plot([r["x"] for r in samst], [r["style_minus_idt"] for r in samst], color=COLORS["samst"], alpha=0.78, zorder=2)
        ax.scatter([r["x"] for r in samst], [r["style_minus_idt"] for r in samst], s=24, color=COLORS["samst"], edgecolor="white", linewidth=0.65, zorder=3)

    ax.axhline(0.0, color=COLORS["idt"], lw=1.15, ls=(0, (7, 4)), zorder=1)
    ax.text(0.405, 0.0035, "IDT", color=COLORS["idt"], fontsize=9.4, weight="bold")
    ax.scatter([seedream["x"]], [seedream["style_minus_idt"]], color=COLORS["seedream"], s=42, edgecolor="white", linewidth=0.8, zorder=4)
    _annotate(ax, float(seedream["x"]), float(seedream["style_minus_idt"]), "Seedream", -18, 12, COLORS["seedream"])
    for row in external_points:
        color = COLORS.get(str(row["id"]), COLORS["legacy"])
        ax.scatter([row["x"]], [row["style_minus_idt"]], color=color, s=38, edgecolor="white", linewidth=0.75, zorder=4)
        _annotate(
            ax,
            float(row["x"]),
            float(row["style_minus_idt"]),
            str(row["label"]),
            int(float(row.get("label_dx", 10.0))),
            int(float(row.get("label_dy", -10.0))),
            color,
        )

    for (group, label), series in by_run.items():
        prefix = label.split("_")[0] if "_" in label else label
        color = COLORS.get(prefix, COLORS.get(group, "#8C8C8C"))
        yvals = [_point_style_minus_idt(p, float(idt["clip_style"])) for p in series]
        ax.plot([p.x for p in series], yvals, color=color, alpha=0.85, zorder=2.8)
        ax.scatter([p.x for p in series], yvals, color=color, edgecolor="white", linewidth=0.6, s=26, zorder=4.2)

    if best_manual is not None:
        _annotate(
            ax,
            best_manual.x,
            _point_style_minus_idt(best_manual, float(idt["clip_style"])),
            f"h0 best e{best_manual.epoch_int}",
            8,
            12,
            COLORS["manual"],
        )

    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"Transfer CLIP-S $-$ IDT $\uparrow$")
    ax.set_xlim(float(viewport["x_min"]), float(viewport["x_max"]))
    ax.set_ylim(float(viewport["y_min"]), float(viewport["y_max"]))
    ax.set_title("(a) WikiArt5 page-1 surface + phase616 live", pad=3.5)

    ax = axes[1]
    ax.set_facecolor(COLORS["panel_bg"])
    labels = ["IDT", "Seedream", "SaMAM", "h0 best"]
    values = [0.0]
    colors = [COLORS["idt"]]
    values.append(float(seedream["style_minus_idt"]))
    colors.append(COLORS["seedream"])
    values.append(max(float(r["style_minus_idt"]) for r in samam))
    colors.append(COLORS["samam"])
    if best_manual is None:
        values.append(0.0)
    else:
        values.append(_point_style_minus_idt(best_manual, float(idt["clip_style"])))
    colors.append(COLORS["manual"])
    bars = ax.bar(labels, values, color=colors, width=0.66)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.0025,
            f"{value:+.3f}",
            ha="center",
            va="bottom",
            fontsize=7.7,
            color=COLORS["text"],
            weight="bold",
        )
    ax.axhline(0.0, color=COLORS["idt"], lw=1.0, ls=(0, (7, 4)))
    ax.set_ylabel(r"Transfer CLIP-S $-$ IDT")
    ax.set_ylim(-0.01, 0.065)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) Style gain checkpoint", pad=3.5)

    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.24, wspace=0.22)
    fig.savefig(FIG_PNG)
    fig.savefig(FIG_PDF)
    plt.close(fig)


def _scale_fn(x0: float, x1: float, px0: float, px1: float):
    span = x1 - x0 if x1 != x0 else 1.0
    pspan = px1 - px0
    return lambda v: px0 + (v - x0) / span * pspan


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _escape(text: object) -> str:
    return html.escape(str(text))


def _render_panel_svg(points: list[RunPoint], baselines: dict[str, object], width: int, height: int) -> str:
    margin_left = 74
    margin_right = 22
    margin_top = 28
    margin_bottom = 54
    x_min, x_max = 0.39, 0.73
    y_min, y_max = -0.07, 0.065
    sx = _scale_fn(x_min, x_max, margin_left, width - margin_right)
    sy = _scale_fn(y_min, y_max, height - margin_bottom, margin_top)

    grid_x = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    grid_y = [-0.06, -0.03, 0.00, 0.03, 0.06]

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="phase616 live dashboard">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="{COLORS["panel_bg"]}"/>',
    ]
    for gx in grid_x:
        x = sx(gx)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="{COLORS["grid"]}" stroke-width="1" opacity="0.65"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - margin_bottom + 18}" text-anchor="middle" class="axis-tick">{gx:.2f}</text>')
    for gy in grid_y:
        y = sy(gy)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="{COLORS["grid"]}" stroke-width="1" opacity="0.65"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-tick">{gy:+.02f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{sy(0.0):.2f}" x2="{width - margin_right}" y2="{sy(0.0):.2f}" stroke="{COLORS["idt"]}" stroke-width="2" stroke-dasharray="8 6"/>')
    parts.append(f'<text x="{sx(0.405):.2f}" y="{sy(0.0) - 8:.2f}" class="idt-label">IDT</text>')

    samam = baselines["samam_curve"]
    samst = baselines["samst_curve"]
    seedream = baselines["seedream"]
    external_points = baselines.get("external_points", [])
    if samam:
        pts = [(sx(float(r["x"])), sy(float(r["style_minus_idt"]))) for r in samam]
        parts.append(f'<polyline fill="none" stroke="{COLORS["samam"]}" stroke-width="2.5" points="{_polyline(pts)}"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="{COLORS["samam"]}" stroke="white" stroke-width="1"/>')
    if samst:
        pts = [(sx(float(r["x"])), sy(float(r["style_minus_idt"]))) for r in samst]
        parts.append(f'<polyline fill="none" stroke="{COLORS["samst"]}" stroke-width="2" points="{_polyline(pts)}" opacity="0.88"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="{COLORS["samst"]}" stroke="white" stroke-width="1.1"/>')

    seed_x = sx(float(seedream["x"]))
    seed_y = sy(float(seedream["style_minus_idt"]))
    parts.append(f'<circle cx="{seed_x:.2f}" cy="{seed_y:.2f}" r="5.4" fill="{COLORS["seedream"]}" stroke="white" stroke-width="1.5"/>')
    parts.append(f'<text x="{seed_x - 14:.2f}" y="{seed_y - 12:.2f}" class="label seedream-label" text-anchor="end">Seedream</text>')
    for row in external_points:
        ext_x = sx(float(row["x"]))
        ext_y = sy(float(row["style_minus_idt"]))
        color = COLORS.get(str(row["id"]), COLORS["legacy"])
        dx = float(row.get("label_dx", 10.0))
        dy = float(row.get("label_dy", -10.0))
        anchor = "end" if dx < 0 else "start"
        parts.append(f'<circle cx="{ext_x:.2f}" cy="{ext_y:.2f}" r="4.8" fill="{color}" stroke="white" stroke-width="1.2"/>')
        parts.append(
            f'<text x="{ext_x + dx:.2f}" y="{ext_y + dy:.2f}" class="label" fill="{color}" text-anchor="{anchor}">{_escape(row["label"])}</text>'
        )

    idt_style = float(baselines["idt"]["clip_style"])
    by_run: dict[tuple[str, str], list[RunPoint]] = {}
    for point in points:
        by_run.setdefault((point.group, point.label), []).append(point)
    for series in by_run.values():
        series.sort(key=lambda p: p.epoch_int)
    for (group, label), series in by_run.items():
        prefix = label.split("_")[0] if "_" in label else label
        color = COLORS.get(prefix, COLORS.get(group, "#888888"))
        pts = [(sx(p.x), sy(_point_style_minus_idt(p, idt_style))) for p in series]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.7" points="{_polyline(pts)}"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="{color}" stroke="white" stroke-width="1.1"/>')

    best_manual = _best_manual(points)
    if best_manual is not None:
        bx = sx(best_manual.x)
        by = sy(_point_style_minus_idt(best_manual, idt_style))
        best_prefix = best_manual.label.split("_")[0] if "_" in best_manual.label else best_manual.label
        best_color = COLORS.get(best_prefix, COLORS["manual"])
        parts.append(f'<circle cx="{bx:.2f}" cy="{by:.2f}" r="6.2" fill="{best_color}" stroke="white" stroke-width="2"/>')
        parts.append(f'<text x="{bx + 10:.2f}" y="{by - 10:.2f}" class="label" fill="{best_color}">h0 best e{best_manual.epoch_int}</text>')

    latest = _latest_by_run(points)
    manual_latest = _latest_point(latest, lambda p: p.group == "manual")
    auto_latest = _latest_point(latest, lambda p: p.group != "manual")
    if manual_latest is not None and (best_manual is None or manual_latest.epoch_int != best_manual.epoch_int):
        lx = sx(manual_latest.x)
        ly = sy(_point_style_minus_idt(manual_latest, idt_style))
        parts.append(f'<text x="{lx + 10:.2f}" y="{ly + 16:.2f}" class="label muted-label">latest e{manual_latest.epoch_int}</text>')
    if auto_latest is not None:
        ax = sx(auto_latest.x)
        ay = sy(_point_style_minus_idt(auto_latest, idt_style))
        auto_prefix = auto_latest.label.split("_")[0] if "_" in auto_latest.label else auto_latest.label
        auto_color = COLORS.get(auto_prefix, COLORS.get(auto_latest.group, "#888888"))
        parts.append(f'<circle cx="{ax:.2f}" cy="{ay:.2f}" r="5.6" fill="{auto_color}" stroke="white" stroke-width="1.6"/>')
        parts.append(f'<text x="{ax - 10:.2f}" y="{ay - 12:.2f}" class="label" text-anchor="end" fill="{auto_color}">{_escape(auto_latest.label)} e{auto_latest.epoch_int}</text>')

    parts.extend(
        [
            f'<text x="{width / 2:.2f}" y="{height - 10}" text-anchor="middle" class="axis-label">1 - LPIPS (content preservation)</text>',
            f'<text x="18" y="{height / 2:.2f}" text-anchor="middle" class="axis-label" transform="rotate(-90 18 {height / 2:.2f})">Transfer CLIP-S - IDT</text>',
            '<text x="20" y="20" class="panel-title">WikiArt5 page-1 frontier with live phase616 run</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def _render_html(points: list[RunPoint], baselines: dict[str, object], status: dict[str, object], snapshot: dict) -> None:
    best_manual = status.get("best_manual")
    latest_manual = _latest_point(points, lambda p: p.group == "manual")
    auto_latest_point = _latest_point(points, lambda p: p.group != "manual")
    auto_runs = [r for r in status["runs"] if r["group"] != "manual"]
    idt = baselines["idt"]
    seedream = baselines["seedream"]
    samam_best = max(baselines["samam_curve"], key=lambda r: float(r["style_minus_idt"]))
    clipt_rows = _load_clipt_rows()
    clipt_best = max(clipt_rows, key=lambda r: float(r["transfer_target_source_margin"])) if clipt_rows else None
    stylet_rows = _load_stylet_rows()
    stylet_best = max(stylet_rows, key=lambda r: float(r["transfer_target_source_margin"])) if stylet_rows else None
    viewport = _compute_pareto_viewport(points, baselines)
    # Dynamic active runs legend entries mapping to their prefix color
    active_runs = sorted(list(set((p.group, p.label) for p in points)), key=lambda x: x[1])
    legend_entries = []
    for grp, lbl in active_runs:
        legend_entries.append((_series_color_key(grp, lbl), f"{lbl} (live)"))
    
    # Baselines
    legend_entries.extend([
        ("samam", "SaMAM"),
        ("samst", "SaMST"),
        ("seedream", "Seedream"),
        ("stylegallery", "StyleGallery"),
        ("styleshot", "StyleShot"),
        ("csgo_low_vram", "CSGO low-VRAM"),
        ("idt", "IDT Floor"),
    ])
    
    legend_html = "\n".join(
        f'        <span style="color: var(--{group});"><i class="sw" style="background:var(--{group})"></i> {_escape(label)}</span>'
        for group, label in legend_entries
    )
    by_run: dict[str, list[dict[str, object]]] = {}
    for point in sorted(points, key=lambda p: (p.group, p.label, p.epoch_int)):
        key = f"{point.group}::{point.label}"
        style_minus_idt = _point_style_minus_idt(point, float(idt["clip_style"]))
        by_run.setdefault(key, []).append(
            {
                "group": point.group,
                "label": point.label,
                "epoch_int": point.epoch_int,
                "clip_style": point.style,
                "clip_s_delta_idt": style_minus_idt,
                "clip_t": point.clip_t,
                "content_lpips": point.lpips,
                "one_minus_lpips": point.x,
                "style_minus_idt": style_minus_idt,
            }
        )
    payload = {
        "updated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "stage_scan": snapshot.get("stage_scan", {}),
        "gpu": snapshot.get("gpu", {}),
        "ps_lines": snapshot.get("ps_lines", []),
        "active_log_lines": snapshot.get("active_log_lines", []),
        "active_log_path": snapshot.get("active_log_path", ""),
        "cards": {
            "current_best": None
            if best_manual is None
            else f"e{best_manual['epoch_int']} | {best_manual['clip_style']:.4f} / {best_manual['content_lpips']:.4f}",
            "current_latest": None
            if auto_latest_point is None
            else f"{auto_latest_point.label} e{auto_latest_point.epoch_int} | {auto_latest_point.style:.4f} / {auto_latest_point.lpips:.4f}",
            "manual_latest": None
            if latest_manual is None
            else f"e{latest_manual.epoch_int} | {latest_manual.style:.4f} / {latest_manual.lpips:.4f}",
            "idt_floor": float(idt["clip_style"]),
            "seedream": {
                "clip_style": float(seedream["clip_style"]),
                "lpips": float(seedream["lpips"]),
                "style_minus_idt": float(seedream["style_minus_idt"]),
            },
            "samam_best": {
                "clip_style": float(samam_best["clip_style"]),
                "lpips": float(samam_best["lpips"]),
                "style_minus_idt": float(samam_best["style_minus_idt"]),
            },
            "clipt_best": None
            if clipt_best is None
            else f"{clipt_best['run']} | {clipt_best['transfer_target_source_margin']:.4f}",
            "stylet_best": None
            if stylet_best is None
            else f"{stylet_best['run']} | {stylet_best['transfer_target_source_margin']:.4f}",
            "auto_runs": [r["label"] for r in auto_runs],
        },
        "baselines": {
            "idt": {"clip_style": float(idt["clip_style"]), "style_minus_idt": 0.0},
            "seedream": {
                "x": float(seedream["x"]),
                "style_minus_idt": float(seedream["style_minus_idt"]),
                "clip_style": float(seedream["clip_style"]),
                "lpips": float(seedream["lpips"]),
            },
            "samam_curve": [
                {
                    "x": float(r["x"]),
                    "style_minus_idt": float(r["style_minus_idt"]),
                    "clip_style": float(r["clip_style"]),
                    "lpips": float(r["lpips"]),
                    "label": str(r["label"]),
                }
                for r in baselines["samam_curve"]
            ],
            "samst_curve": [
                {
                    "x": float(r["x"]),
                    "style_minus_idt": float(r["style_minus_idt"]),
                    "clip_style": float(r["clip_style"]),
                    "lpips": float(r["lpips"]),
                    "label": str(r["label"]),
                }
                for r in baselines["samst_curve"]
            ],
            "external_points": [
                {
                    "id": str(r["id"]),
                    "label": str(r["label"]),
                    "images": int(r["images"]),
                    "x": float(r["x"]),
                    "style_minus_idt": float(r["style_minus_idt"]),
                    "clip_style": float(r["clip_style"]),
                    "lpips": float(r["lpips"]),
                    "label_dx": float(r["label_dx"]),
                    "label_dy": float(r["label_dy"]),
                }
                for r in baselines["external_points"]
            ],
        },
        "clipt": {
            "rows": clipt_rows,
            "best": clipt_best,
        },
        "stylet": {
            "rows": stylet_rows,
            "best": stylet_best,
        },
        "viewport": viewport,
        "run_series": by_run,
        "colors": COLORS,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    DATA_JS_OUT.write_text(
        "window.PHASE616_LIVE_DATA = " + payload_json + ";\n",
        encoding="utf-8",
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <title>Phase 616 Live Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f19;
      --panel: rgba(20, 26, 41, 0.75);
      --ink: #f8fafc;
      --muted: #94a3b8;
      --line: rgba(255, 255, 255, 0.08);
      --manual: {COLORS["manual"]};
      --samam: {COLORS["samam"]};
      --seedream: {COLORS["seedream"]};
      --samst: {COLORS["samst"]};
      --stylegallery: {COLORS["stylegallery"]};
      --styleshot: {COLORS["styleshot"]};
      --csgo_low_vram: {COLORS["csgo_low_vram"]};
      --spatial620: {COLORS["spatial620"]};
      --idt: {COLORS["idt"]};
      --grid: rgba(255, 255, 255, 0.05);
      
      /* Pre-registered colors for h0-h15 */
      --h0: {COLORS["h0"]};
      --h1: {COLORS["h1"]};
      --h2: {COLORS["h2"]};
      --h3: {COLORS["h3"]};
      --h4: {COLORS["h4"]};
      --h5: {COLORS["h5"]};
      --h6: {COLORS["h6"]};
      --h7: {COLORS["h7"]};
      --h8: {COLORS["h8"]};
      --h9: {COLORS["h9"]};
      --h10: {COLORS["h10"]};
      --h11: {COLORS["h11"]};
      --h12: {COLORS["h12"]};
      --h13: {COLORS["h13"]};
      --h14: {COLORS["h14"]};
      --h15: {COLORS["h15"]};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Inter', sans-serif;
      color: var(--ink);
      background: var(--bg);
      background-image: radial-gradient(circle at top center, #1e293b 0%, #0b0f19 100%);
      min-height: 100vh;
    }}
    .wrap {{
      width: min(1360px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .head {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 16px;
      align-items: end;
      margin-bottom: 20px;
    }}
    .glass {{
      background: var(--panel);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }}
    .title {{
      padding: 20px 24px;
    }}
    h1 {{
      margin: 0;
      font-size: 32px;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #ffffff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .sub {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 16px;
      justify-content: flex-end;
      align-items: center;
      padding: 16px 20px;
      height: 100%;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 500;
      color: var(--ink);
    }}
    .sw {{
      width: 14px;
      height: 14px;
      border-radius: 50%;
      display: inline-block;
      box-shadow: 0 0 10px currentColor;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .stat {{
      padding: 16px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      transition: transform 0.2s;
    }}
    .stat:hover {{
      transform: translateY(-2px);
    }}
    .stat-k {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }}
    .stat-v {{
      margin-top: 8px;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 700;
      word-break: break-word;
    }}
    .panel {{
      padding: 20px;
    }}
    .resizable-panel {{
      resize: both;
      overflow: hidden;
      min-width: 600px;
      min-height: 400px;
      max-width: 100%;
      height: 600px;
    }}
    #chart-root {{
      width: 100%;
      height: calc(100% - 48px);
    }}
    .chart-svg {{
      width: 100%;
      height: 100%;
      display: block;
      filter: drop-shadow(0 10px 20px rgba(0,0,0,0.3));
      cursor: grab;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    .toolbar button {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.05);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 16px;
      font: inherit;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .toolbar button:hover {{ 
      background: rgba(255,255,255,0.1); 
      transform: translateY(-1px);
    }}
    .axis-tick {{
      fill: var(--muted);
      font-size: 13px;
      font-family: 'Inter', sans-serif;
    }}
    .axis-label {{
      fill: var(--ink);
      font-size: 15px;
      font-weight: 600;
      font-family: 'Inter', sans-serif;
      letter-spacing: 0.02em;
    }}
    .panel-title {{
      fill: var(--ink);
      font-size: 20px;
      font-weight: 700;
      font-family: 'Inter', sans-serif;
    }}
    .label {{
      font-size: 14px;
      font-weight: 600;
      font-family: 'Inter', sans-serif;
    }}
    .idt-label {{ fill: var(--idt); font-size: 16px; font-weight: 700; }}
    .manual-label {{ fill: var(--manual); }}
    .seedream-label {{ fill: var(--seedream); }}
    .muted-label {{ fill: var(--muted); font-weight: 400; }}
    
    circle {{
      transition: r 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275), filter 0.2s;
      cursor: crosshair;
    }}
    circle:hover {{
      r: 8;
      filter: drop-shadow(0 0 10px currentColor);
    }}

    .foot {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
      margin-top: 20px;
    }}
    .stack {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 20px;
    }}
    .triple {{
      display: grid;
      grid-template-columns: 1fr 1fr 0.8fr;
      gap: 16px;
      margin-top: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.05em;
    }}
    td {{ font-variant-numeric: tabular-nums; }}
    .note {{
      font-size: 15px;
      line-height: 1.6;
      color: var(--muted);
    }}
    .note strong {{ color: var(--ink); }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.5;
      color: var(--muted);
      max-height: 250px;
      overflow-y: auto;
    }}
    .mono {{ font-family: "Cascadia Code", "Fira Code", "Consolas", monospace; }}
    
    @media (max-width: 980px) {{
      .head, .foot {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .stack, .triple {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
    }}

    /* Tooltip styles */
    #tooltip {{
      position: absolute;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      padding: 12px 16px;
      border-radius: 12px;
      color: #fff;
      font-size: 14px;
      pointer-events: none;
      opacity: 0;
      transform: translateY(10px) scale(0.95);
      transition: opacity 0.2s ease, transform 0.2s ease;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
      z-index: 1000;
      line-height: 1.5;
    }}
    #tooltip.visible {{
      opacity: 1;
      transform: translateY(0) scale(1);
    }}
    #tooltip .tt-title {{
      font-weight: 700;
      margin-bottom: 4px;
      color: #fff;
      font-size: 15px;
    }}
    #tooltip .tt-val {{
      font-family: "Cascadia Code", "Fira Code", monospace;
      color: #cbd5e1;
    }}
    #tooltip .tt-color {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
    }}
  </style>
</head>
<body>
  <div id="tooltip"></div>
  <main class="wrap">
    <section class="glass panel resizable-panel" style="margin-bottom: 20px;">
      <div class="toolbar">
        <div>Updated: <span id="updated-at" class="mono" style="color:var(--ink)">n/a</span></div>
        <div>
          <button type="button" id="reset-zoom-btn" style="margin-right: 8px;">Reset Zoom</button>
          <button type="button" id="reload-btn">Reload Data</button>
        </div>
      </div>
      <div id="chart-root"></div>
    </section>

    <section class="glass panel" style="margin-bottom: 20px;">
      <div class="toolbar">
        <div>Active Training Log Stream</div>
        <div id="active-log-source" class="mono" style="font-size: 13px; color: var(--muted)">n/a</div>
      </div>
      <pre id="active-log-box" class="mono" style="max-height: 300px; height: 220px; overflow-y: auto; background: rgba(0,0,0,0.4); padding: 12px; border-radius: 8px; border: 1px solid var(--line); font-size: 13px; line-height: 1.5; color: #a7f3d0; margin: 0;"></pre>
    </section>

    <section class="head">
      <div class="glass title">
        <h1>Phase 616 Live Frontier</h1>
        <div class="sub">
          Matched against the <code class="mono">aaai2027</code> WikiArt5 page-1 surface.<br>
          Y-axis follows the paper convention: <strong>Transfer CLIP-S minus IDT</strong>.
        </div>
      </div>
      <div class="glass legend">
{legend_html}
      </div>
    </section>
    <section class="grid" id="stats-grid"></section>
    <section class="foot">
      <section class="glass panel">
        <table>
          <thead>
            <tr>
              <th>Group</th>
              <th>Run</th>
              <th>Epoch</th>
              <th>CLIP-Style</th>
              <th>CLIP-S - IDT</th>
              <th>CLIP-T</th>
              <th>LPIPS</th>
              <th>1 - LPIPS</th>
            </tr>
          </thead>
          <tbody id="runs-table"></tbody>
        </table>
      </section>
      <section class="glass panel note" id="note-box"></section>
    </section>
    <section class="glass panel" style="margin-top: 20px;">
      <div class="toolbar"><div>External Comparison Baselines</div></div>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Images</th>
            <th>CLIP-S</th>
            <th>LPIPS</th>
            <th>1 - LPIPS</th>
            <th>CLIP-S - IDT</th>
          </tr>
        </thead>
        <tbody id="external-baselines-table"></tbody>
      </table>
    </section>
    <section class="triple">
      <section class="glass panel">
        <div class="toolbar"><div>Representative CLIP-T</div><div id="clipt-best" class="mono"></div></div>
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Run</th>
              <th>Target acc</th>
              <th>Target text</th>
              <th>Margin</th>
              <th>ID acc</th>
            </tr>
          </thead>
          <tbody id="clipt-table"></tbody>
        </table>
      </section>
      <section class="glass panel">
        <div class="toolbar"><div>Representative Style-T</div><div id="stylet-best" class="mono"></div></div>
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Run</th>
              <th>Target acc</th>
              <th>Target prob</th>
              <th>Margin</th>
              <th>ID acc</th>
            </tr>
          </thead>
          <tbody id="stylet-table"></tbody>
        </table>
      </section>
      <section class="glass panel note" id="vlm-box"></section>
    </section>
    <section class="stack">
      <section class="glass panel">
        <div class="toolbar"><div>Active remote processes</div></div>
        <pre id="ps-box" class="mono"></pre>
      </section>
      <section class="glass panel">
        <div class="toolbar"><div>Stage scan</div><div id="gpu-box" class="mono"></div></div>
        <pre id="stage-box" class="mono"></pre>
      </section>
    </section>
  </main>
  <script>
    // Inlined data payload
    window.PHASE616_LIVE_DATA = {payload_json};

    let data = window.PHASE616_LIVE_DATA || null;
    const chartRoot = document.getElementById("chart-root");
    const statsGrid = document.getElementById("stats-grid");
    const runsTable = document.getElementById("runs-table");
    const updatedAt = document.getElementById("updated-at");
    const noteBox = document.getElementById("note-box");
    const tooltip = document.getElementById("tooltip");
    const psBox = document.getElementById("ps-box");
    const stageBox = document.getElementById("stage-box");
    const gpuBox = document.getElementById("gpu-box");
    const cliptTable = document.getElementById("clipt-table");
    const cliptBest = document.getElementById("clipt-best");
    const styletTable = document.getElementById("stylet-table");
    const styletBest = document.getElementById("stylet-best");
    const externalBaselinesTable = document.getElementById("external-baselines-table");
    const vlmBox = document.getElementById("vlm-box");
    const reloadBtn = document.getElementById("reload-btn");
    const resetZoomBtn = document.getElementById("reset-zoom-btn");

    if (reloadBtn) {{
      reloadBtn.addEventListener("click", () => window.location.reload());
    }}

    // Zoom and pan state
    const initXMin = data.viewport?.x_min ?? 0.39;
    const initXMax = data.viewport?.x_max ?? 0.73;
    const initYMin = data.viewport?.y_min ?? -0.07;
    const initYMax = data.viewport?.y_max ?? 0.065;
    let currentXMin = initXMin;
    let currentXMax = initXMax;
    let currentYMin = initYMin;
    let currentYMax = initYMax;

    if (resetZoomBtn) {{
      resetZoomBtn.addEventListener("click", () => {{
        currentXMin = initXMin;
        currentXMax = initXMax;
        currentYMin = initYMin;
        currentYMax = initYMax;
        renderChart();
      }});
    }}

    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;

    chartRoot.addEventListener("mousedown", (e) => {{
      if (e.button !== 0) return;
      hideTooltip();
      isDragging = true;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      const svgEl = chartRoot.querySelector("svg");
      if (svgEl) svgEl.style.cursor = "grabbing";
    }});

    window.addEventListener("mousemove", (e) => {{
      if (!isDragging) return;
      
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      dragStartX = e.clientX;
      dragStartY = e.clientY;

      const svgEl = chartRoot.querySelector("svg");
      if (!svgEl) return;

      const width = chartRoot.clientWidth || 1120;
      const height = chartRoot.clientHeight || 560;
      const margin = {{ left: 80, right: 30, top: 30, bottom: 60 }};
      const pxWidth = width - margin.left - margin.right;
      const pxHeight = height - margin.top - margin.bottom;

      const rect = svgEl.getBoundingClientRect();
      const svgDx = dx * (width / rect.width);
      const svgDy = dy * (height / rect.height);

      const dataWidth = currentXMax - currentXMin;
      const dataHeight = currentYMax - currentYMin;

      const deltaX = (svgDx / pxWidth) * dataWidth;
      const deltaY = (svgDy / pxHeight) * dataHeight;

      currentXMin -= deltaX;
      currentXMax -= deltaX;
      currentYMin += deltaY;
      currentYMax += deltaY;

      renderChart();
    }});

    window.addEventListener("mouseup", () => {{
      if (isDragging) {{
        isDragging = false;
        const svgEl = chartRoot.querySelector("svg");
        if (svgEl) svgEl.style.cursor = "grab";
      }}
    }});

    chartRoot.addEventListener("wheel", (e) => {{
      e.preventDefault();
      hideTooltip();
      const svgEl = chartRoot.querySelector("svg");
      if (!svgEl) return;

      const zoomFactor = e.deltaY > 0 ? 1.08 : 0.92;

      const rect = svgEl.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const width = chartRoot.clientWidth || 1120;
      const height = chartRoot.clientHeight || 560;
      const margin = {{ left: 80, right: 30, top: 30, bottom: 60 }};
      const pxWidth = width - margin.left - margin.right;
      const pxHeight = height - margin.top - margin.bottom;

      const svgX = mouseX * (width / rect.width);
      const svgY = mouseY * (height / rect.height);

      if (svgX >= margin.left && svgX <= width - margin.right && svgY >= margin.top && svgY <= height - margin.bottom) {{
        const dataX = currentXMin + ((svgX - margin.left) / pxWidth) * (currentXMax - currentXMin);
        const dataY = currentYMax - ((svgY - margin.top) / pxHeight) * (currentYMax - currentYMin);

        currentXMin = dataX - (dataX - currentXMin) * zoomFactor;
        currentXMax = dataX + (currentXMax - dataX) * zoomFactor;
        currentYMin = dataY - (dataY - currentYMin) * zoomFactor;
        currentYMax = dataY + (currentYMax - dataY) * zoomFactor;
      }} else {{
        const midX = (currentXMin + currentXMax) / 2;
        const midY = (currentYMin + currentYMax) / 2;
        const halfX = (currentXMax - currentXMin) / 2 * zoomFactor;
        const halfY = (currentYMax - currentYMin) / 2 * zoomFactor;
        currentXMin = midX - halfX;
        currentXMax = midX + halfX;
        currentYMin = midY - halfY;
        currentYMax = midY + halfY;
      }}
      renderChart();
    }});

    function fmt(v, digits = 4) {{
      return Number.isFinite(v) ? v.toFixed(digits) : "n/a";
    }}

    const GLOBAL_IDT_FLOOR = {OLD_GLOBAL_IDT_CLIP_STYLE:.12f};
    if (data.cards) {{
      data.cards.idt_floor = GLOBAL_IDT_FLOOR;
    }}
    function clipDelta(row) {{
      const clipStyle = Number(row && row.clip_style);
      return Number.isFinite(clipStyle) ? clipStyle - GLOBAL_IDT_FLOOR : NaN;
    }}

    function esc(text) {{
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function showTooltip(e, title, color, styleMinusIdt, oneMinusLpips) {{
      tooltip.innerHTML = `
        <div class="tt-title"><span class="tt-color" style="background:${{color}}; box-shadow: 0 0 6px ${{color}}"></span>${{esc(title)}}</div>
        <div class="tt-val">CLIP-S - IDT: ${{fmt(styleMinusIdt)}}</div>
        <div class="tt-val">1 - LPIPS:    ${{fmt(oneMinusLpips)}}</div>
      `;
      tooltip.classList.add('visible');
      moveTooltip(e);
    }}

    function moveTooltip(e) {{
      if (!tooltip.classList.contains('visible')) return;
      const offset = 15;
      let x = e.pageX + offset;
      let y = e.pageY + offset;
      
      const box = tooltip.getBoundingClientRect();
      if (x + box.width > window.innerWidth) x = e.pageX - box.width - offset;
      if (y + box.height > window.innerHeight) y = e.pageY - box.height - offset;
      
      tooltip.style.left = x + 'px';
      tooltip.style.top = y + 'px';
    }}

    function hideTooltip() {{
      tooltip.classList.remove('visible');
    }}

    document.addEventListener('mousemove', moveTooltip);

    function renderStats() {{
      const cards = [
        ["Current best", data.cards.current_best || "n/a"],
        ["Current latest", data.cards.current_latest || "n/a"],
        ["IDT floor", fmt(data.cards.idt_floor)],
        ["Seedream", `${{fmt(data.cards.seedream.clip_style)}} / ${{fmt(data.cards.seedream.lpips)}}`],
        ["SaMAM best", `${{fmt(data.cards.samam_best.clip_style)}} / ${{fmt(data.cards.samam_best.lpips)}}`],
        ["CLIP-T best", data.cards.clipt_best || "n/a"],
        ["Style-T best", data.cards.stylet_best || "n/a"],
        ["Auto runs", data.cards.auto_runs.length ? data.cards.auto_runs.join(", ") : "none yet"],
      ];

      statsGrid.innerHTML = cards.map(([k, v]) => `
        <section class="glass stat">
          <div class="stat-k">${{esc(k)}}</div>
          <div class="stat-v">${{esc(v)}}</div>
        </section>
      `).join("");
    }}

    function renderTable() {{
      runsTable.innerHTML = data.status.runs.map((r) => `
        <tr>
          <td>${{esc(r.group)}}</td>
          <td>${{esc(r.label)}}</td>
          <td>e${{r.epoch_int}}</td>
          <td>${{fmt(r.clip_style)}}</td>
          <td>${{fmt(clipDelta(r))}}</td>
          <td>${{fmt(r.clip_t)}}</td>
          <td>${{fmt(r.content_lpips)}}</td>
          <td>${{fmt(r.one_minus_lpips)}}</td>
        </tr>
      `).join("");
    }}

    function renderExternalBaselines() {{
      const rows = (data.baselines && data.baselines.external_points) ? data.baselines.external_points : [];
      if (!rows.length) {{
        externalBaselinesTable.innerHTML = `<tr><td colspan="6" style="color:var(--muted)">no external baselines mirrored yet</td></tr>`;
        return;
      }}
      externalBaselinesTable.innerHTML = rows.map((r) => `
        <tr>
          <td>${{esc(r.label)}}</td>
          <td>${{r.images}}</td>
          <td>${{fmt(r.clip_style)}}</td>
          <td>${{fmt(r.lpips)}}</td>
          <td>${{fmt(r.x)}}</td>
          <td>${{fmt(r.style_minus_idt)}}</td>
        </tr>
      `).join("");
    }}

    function renderNote() {{
      noteBox.innerHTML = `
        <strong>Reference Anchors</strong><br><br>
        IDT floor: <span class="mono" style="color:var(--ink)">${{fmt(data.cards.idt_floor)}}</span><br>
        Seedream: <span class="mono" style="color:var(--ink)">${{fmt(data.cards.seedream.clip_style)}} / ${{fmt(data.cards.seedream.lpips)}}</span><br>
        SaMAM best page-1 point: <span class="mono" style="color:var(--ink)">${{fmt(data.cards.samam_best.clip_style)}} / ${{fmt(data.cards.samam_best.lpips)}}</span><br>
        Rep CLIP-T best: <span class="mono" style="color:var(--ink)">${{esc(data.cards.clipt_best || "n/a")}}</span><br><br>
        Rep Style-T best: <span class="mono" style="color:var(--ink)">${{esc(data.cards.stylet_best || "n/a")}}</span><br><br>
        This page reads from <span class="mono">phase616_live_data.js</span>, regenerated by the 5-minute sync job.
      `;
    }}

    function renderClipT() {{
      const rows = (data.clipt && data.clipt.rows) ? data.clipt.rows : [];
      cliptBest.textContent = data.cards.clipt_best || "n/a";
      if (!rows.length) {{
        cliptTable.innerHTML = `<tr><td colspan="6" style="color:var(--muted)">clip-t summary not ready</td></tr>`;
      }} else {{
        cliptTable.innerHTML = rows.map((r) => `
          <tr>
            <td>${{esc(r.method)}}</td>
            <td>${{esc(r.run)}}</td>
            <td>${{fmt(r.transfer_target_acc)}}</td>
            <td>${{fmt(r.transfer_target_text)}}</td>
            <td>${{fmt(r.transfer_target_source_margin)}}</td>
            <td>${{fmt(r.identity_source_acc)}}</td>
          </tr>
        `).join("");
      }}
      const prompt = rows.length ? rows[0].prompt_template : "n/a";
      vlmBox.innerHTML = `
        <strong>VLM Track</strong><br><br>
        Representative-point CLIP-T is wired through a reusable manifest.<br>
        Representative-point Style-T reuses our existing Distinct5 ConvNeXt style classifier.<br>
        Prompt template: <span class="mono" style="color:var(--ink)">${{esc(prompt)}}</span><br><br>
        Next step is to run the existing external-Qwen VLM scripts on the same representative set, then mirror the method summary CSV into this panel.
      `;
    }}

    function renderStyleT() {{
      const rows = (data.stylet && data.stylet.rows) ? data.stylet.rows : [];
      styletBest.textContent = data.cards.stylet_best || "n/a";
      if (!rows.length) {{
        styletTable.innerHTML = `<tr><td colspan="6" style="color:var(--muted)">style-t summary not ready</td></tr>`;
      }} else {{
        styletTable.innerHTML = rows.map((r) => `
          <tr>
            <td>${{esc(r.method)}}</td>
            <td>${{esc(r.run)}}</td>
            <td>${{fmt(r.transfer_target_acc)}}</td>
            <td>${{fmt(r.transfer_target_prob)}}</td>
            <td>${{fmt(r.transfer_target_source_margin)}}</td>
            <td>${{fmt(r.identity_source_acc)}}</td>
          </tr>
        `).join("");
      }}
    }}

    function renderRemoteStatus() {{
      psBox.textContent = (data.ps_lines && data.ps_lines.length) ? data.ps_lines.join("\\n") : "no tracked remote process lines";
      
      const logBox = document.getElementById("active-log-box");
      const logSource = document.getElementById("active-log-source");
      if (logBox) {{
        if (data.active_log_lines && data.active_log_lines.length) {{
          logBox.textContent = data.active_log_lines.join("\\n");
          logBox.scrollTop = logBox.scrollHeight;
        }} else {{
          logBox.textContent = "No active training logs detected in the last 2 hours.";
        }}
      }}
      if (logSource) {{
        logSource.textContent = data.active_log_path || "n/a";
      }}

      const scans = [];
      for (const [group, items] of Object.entries(data.stage_scan || {{}})) {{
        scans.push(`[${{group}}]`);
        if (!items.length) {{
          scans.push("  <empty>");
          continue;
        }}
        for (const item of items) {{
          scans.push(`  ${{item.label}} | curve=${{item.has_curve}} auto_summary=${{item.has_auto_summary}} config=${{item.has_config}} csv=${{item.latest_training_csv || '-'}}`);
        }}
      }}
      stageBox.textContent = scans.join("\\n");
      gpuBox.textContent = (data.gpu && data.gpu.lines && data.gpu.lines.length) ? data.gpu.lines[0] : "gpu n/a";
    }}

    function scale(v, a0, a1, b0, b1) {{
      const t = (v - a0) / (a1 - a0);
      return b0 + t * (b1 - b0);
    }}

    function poly(points) {{
      return points.map((p) => `${{p[0].toFixed(2)}},${{p[1].toFixed(2)}}`).join(" ");
    }}

    function renderChart() {{
      const width = chartRoot.clientWidth || 1120;
      const height = chartRoot.clientHeight || 560;
      const margin = {{ left: 80, right: 30, top: 30, bottom: 60 }};
      const xMin = currentXMin, xMax = currentXMax, yMin = currentYMin, yMax = currentYMax;
      const sx = (v) => scale(v, xMin, xMax, margin.left, width - margin.right);
      const sy = (v) => scale(v, yMin, yMax, height - margin.bottom, margin.top);

      function getTicks(min, max, targetCount = 7) {{
        const range = max - min;
        const roughStep = range / targetCount;
        const l10 = Math.floor(Math.log10(roughStep));
        const p10 = Math.pow(10, l10);
        const normalized = roughStep / p10;
        let step;
        if (normalized < 1.5) step = 1 * p10;
        else if (normalized < 3) step = 2 * p10;
        else if (normalized < 7) step = 5 * p10;
        else step = 10 * p10;
        
        const start = Math.ceil(min / step) * step;
        const ticks = [];
        for (let v = start; v <= max; v += step) {{
          ticks.push(Number(v.toFixed(10)));
        }}
        const precision = Math.max(0, -Math.floor(Math.log10(step)));
        return {{ ticks, precision }};
      }}
      
      const gridXInfo = getTicks(xMin, xMax, 7);
      const gridYInfo = getTicks(yMin, yMax, 6);
      const gridX = gridXInfo.ticks;
      const precisionX = gridXInfo.precision;
      const gridY = gridYInfo.ticks;
      const precisionY = gridYInfo.precision;
      
      const colors = {{
        samam: 'var(--samam)',
        samst: 'var(--samst)',
        seedream: 'var(--seedream)',
        manual: 'var(--manual)',
        idt: 'var(--idt)',
        grid: 'var(--grid)'
      }};

      const cursorStyle = isDragging ? "grabbing" : "grab";
      let svg = `<svg viewBox="0 0 ${{width}} ${{height}}" class="chart-svg" style="cursor: ${{cursorStyle}}" role="img" aria-label="phase616 monitor">
        <defs>
          <clipPath id="plot-clip">
            <rect x="${{margin.left}}" y="${{margin.top}}" width="${{width - margin.left - margin.right}}" height="${{height - margin.top - margin.bottom}}" />
          </clipPath>
        </defs>
        <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="12" fill="rgba(0,0,0,0.2)"/>`;
      
      for (const gx of gridX) {{
        const x = sx(gx);
        const formatted = gx.toFixed(precisionX);
        svg += `<line x1="${{x.toFixed(2)}}" y1="${{margin.top}}" x2="${{x.toFixed(2)}}" y2="${{height - margin.bottom}}" stroke="${{colors.grid}}" stroke-width="1.5"/>`;
        svg += `<text x="${{x.toFixed(2)}}" y="${{height - margin.bottom + 22}}" text-anchor="middle" class="axis-tick">${{formatted}}</text>`;
      }}
      for (const gy of gridY) {{
        const y = sy(gy);
        const formatted = gy.toFixed(precisionY);
        svg += `<line x1="${{margin.left}}" y1="${{y.toFixed(2)}}" x2="${{width - margin.right}}" y2="${{y.toFixed(2)}}" stroke="${{colors.grid}}" stroke-width="1.5"/>`;
        svg += `<text x="${{margin.left - 14}}" y="${{(y + 5).toFixed(2)}}" text-anchor="end" class="axis-tick">${{formatted}}</text>`;
      }}
      
      svg += `<g clip-path="url(#plot-clip)">`;

      const idtY = sy(0);
      svg += `<line x1="${{margin.left}}" y1="${{idtY.toFixed(2)}}" x2="${{width - margin.right}}" y2="${{idtY.toFixed(2)}}" stroke="${{colors.idt}}" stroke-width="2.5" stroke-dasharray="6 6" opacity="0.8"/>`;
      svg += `<text x="${{sx(0.405).toFixed(2)}}" y="${{(idtY - 10).toFixed(2)}}" class="idt-label">IDT Floor</text>`;

      const samamPts = data.baselines.samam_curve.map((r) => [sx(r.x), sy(r.style_minus_idt)]);
      svg += `<polyline fill="none" stroke="${{colors.samam}}" stroke-width="3" opacity="0.9" points="${{poly(samamPts)}}"/>`;
      for (const r of data.baselines.samam_curve) {{
        const x = sx(r.x), y = sy(r.style_minus_idt);
        svg += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="4.5" fill="${{colors.samam}}" stroke="#0f172a" stroke-width="2" 
                  onmouseover="showTooltip(event, '${{r.label}}', '#3b82f6', ${{r.style_minus_idt}}, ${{r.x}})" onmouseout="hideTooltip()"/>`;
      }}

      const samstPts = data.baselines.samst_curve.map((r) => [sx(r.x), sy(r.style_minus_idt)]);
      if (samstPts.length) {{
        svg += `<polyline fill="none" stroke="${{colors.samst}}" stroke-width="2.5" opacity="0.8" points="${{poly(samstPts)}}"/>`;
        for (const r of data.baselines.samst_curve) {{
          const x = sx(r.x), y = sy(r.style_minus_idt);
          svg += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="4.5" fill="${{colors.samst}}" stroke="#0f172a" stroke-width="2" 
                    onmouseover="showTooltip(event, '${{r.label}}', '#10b981', ${{r.style_minus_idt}}, ${{r.x}})" onmouseout="hideTooltip()"/>`;
        }}
      }}

      const seed = data.baselines.seedream;
      svg += `<circle cx="${{sx(seed.x).toFixed(2)}}" cy="${{sy(seed.style_minus_idt).toFixed(2)}}" r="6" fill="${{colors.seedream}}" stroke="#0f172a" stroke-width="2.5" 
                onmouseover="showTooltip(event, 'Seedream', '#f59e0b', ${{seed.style_minus_idt}}, ${{seed.x}})" onmouseout="hideTooltip()"/>`;
      svg += `<text x="${{(sx(seed.x)-16).toFixed(2)}}" y="${{(sy(seed.style_minus_idt)-14).toFixed(2)}}" class="label seedream-label" text-anchor="end">Seedream</text>`;
      for (const row of (data.baselines.external_points || [])) {{
        const rawColor = data.colors[row.id] || "#6b7280";
        const seriesColor = data.colors[row.id] ? `var(--${{row.id}})` : rawColor;
        const x = sx(row.x), y = sy(row.style_minus_idt);
        const anchor = (row.label_dx || 0) < 0 ? "end" : "start";
        svg += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="5.2" fill="${{seriesColor}}" stroke="#0f172a" stroke-width="2"
                  onmouseover="showTooltip(event, '${{row.label}}', '${{rawColor}}', ${{row.style_minus_idt}}, ${{row.x}})" onmouseout="hideTooltip()"/>`;
        svg += `<text x="${{(x + row.label_dx).toFixed(2)}}" y="${{(y + row.label_dy).toFixed(2)}}" class="label" text-anchor="${{anchor}}" fill="${{seriesColor}}">${{row.label}}</text>`;
      }}

      for (const [key, series] of Object.entries(data.run_series)) {{
        const label = series[0].label;
        const prefix = label.includes('_') ? label.split('_')[0] : label;
        const rawColor = data.colors[prefix] || data.colors[series[0].group] || "#888888";
        const seriesColor = data.colors[prefix] ? `var(--${{prefix}})` : rawColor;
        const hexForTooltip = rawColor;
        
        const pts = series.map((r) => [sx(r.one_minus_lpips), sy(clipDelta(r))]);
        svg += `<polyline fill="none" stroke="${{seriesColor}}" stroke-width="3" points="${{poly(pts)}}"/>`;
        for (const r of series) {{
          const yValue = clipDelta(r);
          const x = sx(r.one_minus_lpips), y = sy(yValue);
          const title = `${{r.label}} e${{r.epoch_int}}`;
          svg += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="4.5" fill="${{seriesColor}}" stroke="#0f172a" stroke-width="2" 
                    onmouseover="showTooltip(event, '${{title}}', '${{hexForTooltip}}', ${{yValue}}, ${{r.one_minus_lpips}})" onmouseout="hideTooltip()"/>`;
        }}
      }}

      const best = data.status.best_manual;
      if (best) {{
        const bestDelta = clipDelta(best);
        const bx = sx(best.one_minus_lpips), by = sy(bestDelta);
        const bestLabel = best.label;
        const bestPrefix = bestLabel.includes('_') ? bestLabel.split('_')[0] : bestLabel;
        const bestRawColor = data.colors[bestPrefix] || colors.manual;
        const bestColor = data.colors[bestPrefix] ? `var(--${{bestPrefix}})` : colors.manual;
        svg += `<circle cx="${{bx.toFixed(2)}}" cy="${{by.toFixed(2)}}" r="7" fill="${{bestColor}}" stroke="#fff" stroke-width="2.5" 
                  onmouseover="showTooltip(event, '${{best.label}} best e${{best.epoch_int}}', '${{bestRawColor}}', ${{bestDelta}}, ${{best.one_minus_lpips}})" onmouseout="hideTooltip()"/>`;
        svg += `<text x="${{(bx+14).toFixed(2)}}" y="${{(by-14).toFixed(2)}}" class="label" fill="${{bestColor}}">${{best.label}} best e${{best.epoch_int}}</text>`;
      }}

      svg += `</g>`;

      svg += `<text x="${{(width/2).toFixed(2)}}" y="${{height - 15}}" text-anchor="middle" class="axis-label">1 - LPIPS (Content Preservation)</text>`;
      svg += `<text x="22" y="${{(height/2).toFixed(2)}}" text-anchor="middle" class="axis-label" transform="rotate(-90 22 ${{(height/2).toFixed(2)}})">Transfer CLIP-S - IDT</text>`;
      svg += `</svg>`;
      
      chartRoot.innerHTML = svg;
    }}

    let currentData = data;

    function updateAll(newData) {{
      currentData = newData;
      updatedAt.textContent = currentData.updated_at;
      if (currentData.status && currentData.status.offline) {{
        if (updatedAt.style) updatedAt.style.color = "#ef4444";
        updatedAt.textContent += " (OFFLINE / SYNC FAILED)";
        if (currentData.status.offline_error) {{
          updatedAt.title = currentData.status.offline_error;
        }}
      }} else {{
        if (updatedAt.style) updatedAt.style.color = "var(--ink)";
      }}
      
      window.PHASE616_LIVE_DATA = currentData;
      data = currentData;
      
      renderStats();
      renderChart();
      renderTable();
      renderExternalBaselines();
      renderNote();
      renderClipT();
      renderStyleT();
      renderRemoteStatus();
    }}

    if (currentData) {{
      updateAll(currentData);
      
      const panel = document.querySelector(".resizable-panel");
      if (panel) {{
        const ro = new ResizeObserver(() => {{
          renderChart();
        }});
        ro.observe(panel);
      }}
    }} else {{
      chartRoot.innerHTML = "<div style='padding: 40px; text-align: center; color: var(--muted)'>Inlined data missing or failed to parse.</div>";
    }}

    async function pollData() {{
      try {{
        const res = await fetch("/data");
        if (res.ok) {{
          const json = await res.json();
          if (json && json.updated_at !== currentData.updated_at) {{
            updateAll(json);
          }}
        }}
      }} catch (e) {{
        console.warn("Polling failed:", e);
      }}
    }}
    setInterval(pollData, 5000);
  </script>
</body>
</html>
"""
    HTML_OUT.write_text(html_text, encoding="utf-8")



def main() -> int:
    offline = False
    error_msg = ""
    snapshot = None
    synced_archives: list[dict[str, object]] = []
    try:
        snapshot = _fetch_remote_snapshot()
        SNAPSHOT_JSON.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_mirror(snapshot)
        synced_archives = _sync_best_eval_archives(snapshot)
        _write_clipt_manifest(synced_archives)
    except Exception as e:
        offline = True
        error_msg = str(e)
        if SNAPSHOT_JSON.is_file():
            try:
                snapshot = json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))
            except Exception:
                pass

    try:
        _maybe_refresh_clipt_summary()
    except Exception:
        pass
    try:
        _maybe_refresh_stylet_summary()
    except Exception:
        pass

    if not snapshot:
        snapshot = {"runs": [], "stage_scan": {}, "gpu": {}, "ps_lines": []}

    points = _parse_points(snapshot)
    baselines = _build_baseline_bundle()
    if points:
        try:
            _plot_png(points, baselines)
        except Exception:
            pass

    latest = _latest_by_run(points)
    best_manual = _best_manual(points)
    idt_floor = float(baselines["idt"]["clip_style"])
    status = {
        "run_count": len(latest),
        "runs": [
            {
                "group": p.group,
                "label": p.label,
                "epoch": p.epoch,
                "epoch_int": p.epoch_int,
                "clip_style": p.style,
                "clip_s_delta_idt": _point_style_minus_idt(p, idt_floor),
                "clip_t": p.clip_t,
                "content_lpips": p.lpips,
                "one_minus_lpips": p.x,
            }
            for p in latest
        ],
        "best_manual": None
        if best_manual is None
        else {
            "group": best_manual.group,
            "label": best_manual.label,
            "epoch": best_manual.epoch,
            "epoch_int": best_manual.epoch_int,
            "clip_style": best_manual.style,
            "clip_s_delta_idt": _point_style_minus_idt(best_manual, idt_floor),
            "clip_t": best_manual.clip_t,
            "content_lpips": best_manual.lpips,
            "one_minus_lpips": best_manual.x,
        },
        "new_auto_stage_runs": any(p.group != "manual" for p in latest),
        "figure_png": str(FIG_PNG),
        "figure_pdf": str(FIG_PDF),
        "dashboard_html": str(HTML_OUT),
        "dashboard_data_js": str(DATA_JS_OUT),
        "synced_best_eval_archives": synced_archives,
        "offline": offline,
        "offline_error": error_msg,
    }
    _render_html(points, baselines, status, snapshot)
    STATUS_JSON.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(FIG_PNG)
    print(FIG_PDF)
    print(HTML_OUT)
    print(STATUS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
