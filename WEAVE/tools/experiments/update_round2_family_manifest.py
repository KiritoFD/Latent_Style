from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        clean: dict[str, str] = {}
        for key, value in raw.items():
            if key is None:
                continue
            normalized = str(key).replace("\ufeff", "").replace('"', "").strip()
            clean[normalized] = value
        rows.append(clean)
    return rows


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            names.append(key)
    return names


def _write_rows(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_gap_report_for_family(*, workspace: Path, family_id: str) -> tuple[dict, Path | None]:
    family_dir = workspace / "SchrodingerBridge" / "docs" / "experiments" / "round2_pure_sde" / str(family_id).strip()
    if not family_dir.is_dir():
        return {}, None
    candidates = sorted(family_dir.glob("gap_vs_*.json"))
    if not candidates:
        return {}, None
    payload = _load_json_if_exists(candidates[0])
    return payload, candidates[0] if payload else None


def _epoch_int(value) -> int:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else -1


def _curve_sort_key(payload: dict) -> tuple[int, int]:
    if not isinstance(payload, dict):
        return (-1, -1)
    latest = payload.get("latest") or {}
    row_count = int(payload.get("row_count") or 0)
    latest_epoch = _epoch_int(latest.get("epoch"))
    return (row_count, latest_epoch)


def _convergence_sort_key(payload: dict) -> tuple[int, int]:
    if not isinstance(payload, dict):
        return (-1, -1)
    pareto_epochs = payload.get("pareto_epochs") or []
    if not isinstance(pareto_epochs, list):
        pareto_epochs = []
    return (len(pareto_epochs), _epoch_int(payload.get("last_pareto_epoch")))


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _load_remote_json_if_exists(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_path: str,
) -> dict:
    proc = _run(
        [
            "ssh",
            "-p",
            str(int(port)),
            "-T",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            "wsl",
            "-d",
            str(wsl_distro),
            "--exec",
            "cat",
            str(remote_path),
        ]
    )
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _str_or_blank(value) -> str:
    if value is None:
        return ""
    return str(value)


def _resolve_run_dir_candidates(row: dict[str, str]) -> list[str]:
    out: list[str] = []
    for key in ("active_run_dir", "run_dir"):
        value = str(row.get(key, "")).strip()
        if value and value not in out:
            out.append(value)
    return out


def _run_root(run_dir: str, *, workspace: Path) -> Path:
    if run_dir.startswith("./"):
        return (workspace / run_dir[2:]).resolve()
    return Path(run_dir).expanduser().resolve()


def _remote_run_root(run_dir: str, *, remote_wsl_cwd: str) -> str:
    if run_dir.startswith("./"):
        return f"{remote_wsl_cwd.rstrip('/')}/{run_dir[2:]}"
    return run_dir.replace("\\", "/")


def _load_run_summaries(
    *,
    run_dir: str,
    workspace: Path,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_wsl_cwd: str,
) -> tuple[dict, dict, Path | None, Path | None, str, str]:
    curve_summary: dict = {}
    convergence: dict = {}
    curve_summary_path: Path | None = None
    convergence_path: Path | None = None
    remote_curve_summary_path = ""
    remote_convergence_path = ""
    if not str(run_dir).strip():
        return curve_summary, convergence, curve_summary_path, convergence_path, remote_curve_summary_path, remote_convergence_path
    run_root = _run_root(run_dir, workspace=workspace)
    curve_summary_path = run_root / "full_eval" / "curve_summary.json"
    convergence_path = run_root / "full_eval" / "round2_convergence.json"
    remote_base = _remote_run_root(run_dir, remote_wsl_cwd=remote_wsl_cwd).rstrip("/")
    remote_curve_summary_path = f"{remote_base}/full_eval/curve_summary.json"
    remote_convergence_path = f"{remote_base}/full_eval/round2_convergence.json"
    local_curve = _load_json_if_exists(curve_summary_path)
    local_convergence = _load_json_if_exists(convergence_path)
    remote_curve = _load_remote_json_if_exists(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        remote_path=remote_curve_summary_path,
    )
    remote_convergence = _load_remote_json_if_exists(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        remote_path=remote_convergence_path,
    )

    use_remote_curve = _curve_sort_key(remote_curve) > _curve_sort_key(local_curve)
    curve_summary = remote_curve if use_remote_curve else local_curve
    if not curve_summary:
        curve_summary = local_curve or remote_curve

    use_remote_convergence = False
    if use_remote_curve:
        use_remote_convergence = bool(remote_convergence)
    elif _convergence_sort_key(remote_convergence) > _convergence_sort_key(local_convergence):
        use_remote_convergence = True
    convergence = remote_convergence if use_remote_convergence else local_convergence
    if not convergence:
        convergence = local_convergence or remote_convergence

    chosen_curve_path = None if use_remote_curve else curve_summary_path
    chosen_convergence_path = None if use_remote_convergence else convergence_path
    return (
        curve_summary,
        convergence,
        chosen_curve_path,
        chosen_convergence_path,
        remote_curve_summary_path,
        remote_convergence_path,
    )


def _write_summary_fields(
    row: dict[str, str],
    *,
    prefix: str,
    curve_summary: dict,
    convergence: dict,
    curve_summary_path: Path | None,
    convergence_path: Path | None,
    remote_curve_summary_path: str,
    remote_convergence_path: str,
) -> None:
    latest = curve_summary.get("latest") or {}
    best_transfer = curve_summary.get("best_transfer") or {}
    best_all_pairs = curve_summary.get("best_all_pairs") or {}
    key = lambda name: f"{prefix}{name}" if prefix else name
    row[key("settled_eval_rows")] = _str_or_blank(curve_summary.get("row_count"))
    row[key("latest_epoch")] = _str_or_blank(latest.get("epoch"))
    row[key("latest_checkpoint")] = _str_or_blank(latest.get("checkpoint"))
    row[key("latest_transfer_clip_style")] = _str_or_blank(latest.get("transfer_clip_style"))
    row[key("latest_transfer_content_lpips")] = _str_or_blank(latest.get("transfer_content_lpips"))
    row[key("latest_all_pairs_clip_style")] = _str_or_blank(latest.get("all_pairs_clip_style"))
    row[key("latest_all_pairs_content_lpips")] = _str_or_blank(latest.get("all_pairs_content_lpips"))
    row[key("best_transfer_epoch")] = _str_or_blank(best_transfer.get("epoch"))
    row[key("best_transfer_clip_style")] = _str_or_blank(best_transfer.get("transfer_clip_style"))
    row[key("best_transfer_content_lpips")] = _str_or_blank(best_transfer.get("transfer_content_lpips"))
    row[key("best_all_pairs_epoch")] = _str_or_blank(best_all_pairs.get("epoch"))
    row[key("best_all_pairs_clip_style")] = _str_or_blank(best_all_pairs.get("all_pairs_clip_style"))
    row[key("best_all_pairs_content_lpips")] = _str_or_blank(best_all_pairs.get("all_pairs_content_lpips"))
    row[key("curve_summary_path")] = (
        str(curve_summary_path.resolve())
        if curve_summary_path is not None and curve_summary_path.exists()
        else remote_curve_summary_path
        if curve_summary
        else ""
    )
    row[key("pareto_epochs")] = ",".join(str(x).strip() for x in (convergence.get("pareto_epochs") or []) if str(x).strip())
    row[key("last_pareto_epoch")] = _str_or_blank(convergence.get("last_pareto_epoch"))
    row[key("since_last_pareto")] = _str_or_blank(convergence.get("since_last_pareto"))
    row[key("best_in_newest_2")] = _str_or_blank(convergence.get("best_in_newest_2"))
    row[key("tail_flat")] = _str_or_blank(convergence.get("tail_flat"))
    row[key("converged")] = _str_or_blank(convergence.get("converged"))
    row[key("convergence_path")] = (
        str(convergence_path.resolve())
        if convergence_path is not None and convergence_path.exists()
        else remote_convergence_path
        if convergence
        else ""
    )


def _write_gap_fields(
    row: dict[str, str],
    *,
    prefix: str,
    gap_payload: dict,
    gap_path: Path | None,
) -> None:
    key = lambda name: f"{prefix}{name}" if prefix else name
    latest = gap_payload.get("latest") or {}
    best_transfer = gap_payload.get("best_transfer") or {}
    best_all_pairs = gap_payload.get("best_all_pairs") or {}
    reference = gap_payload.get("reference") or {}
    row[key("gap_reference_name")] = _str_or_blank(reference.get("name"))
    row[key("gap_report_path")] = str(gap_path.resolve()) if gap_path is not None and gap_path.exists() else ""
    for source, tag in (
        (latest, "latest"),
        (best_transfer, "best_transfer"),
        (best_all_pairs, "best_all_pairs"),
    ):
        row[key(f"{tag}_transfer_style_gap")] = _str_or_blank(source.get("transfer_style_gap"))
        row[key(f"{tag}_transfer_lpips_gap")] = _str_or_blank(source.get("transfer_lpips_gap"))
        row[key(f"{tag}_all_pairs_style_gap")] = _str_or_blank(source.get("all_pairs_style_gap"))
        row[key(f"{tag}_all_pairs_lpips_gap")] = _str_or_blank(source.get("all_pairs_lpips_gap"))


def _clear_gap_fields(
    row: dict[str, str],
    *,
    prefix: str,
) -> None:
    key = lambda name: f"{prefix}{name}" if prefix else name
    for name in (
        "gap_reference_name",
        "gap_report_path",
        "latest_transfer_style_gap",
        "latest_transfer_lpips_gap",
        "latest_all_pairs_style_gap",
        "latest_all_pairs_lpips_gap",
        "best_transfer_transfer_style_gap",
        "best_transfer_transfer_lpips_gap",
        "best_transfer_all_pairs_style_gap",
        "best_transfer_all_pairs_lpips_gap",
        "best_all_pairs_transfer_style_gap",
        "best_all_pairs_transfer_lpips_gap",
        "best_all_pairs_all_pairs_style_gap",
        "best_all_pairs_all_pairs_lpips_gap",
    ):
        row[key(name)] = ""


def _clear_summary_fields(
    row: dict[str, str],
    *,
    prefix: str,
) -> None:
    key = lambda name: f"{prefix}{name}" if prefix else name
    for name in (
        "settled_eval_rows",
        "latest_epoch",
        "latest_checkpoint",
        "latest_transfer_clip_style",
        "latest_transfer_content_lpips",
        "latest_all_pairs_clip_style",
        "latest_all_pairs_content_lpips",
        "best_transfer_epoch",
        "best_transfer_clip_style",
        "best_transfer_content_lpips",
        "best_all_pairs_epoch",
        "best_all_pairs_clip_style",
        "best_all_pairs_content_lpips",
        "curve_summary_path",
        "pareto_epochs",
        "last_pareto_epoch",
        "since_last_pareto",
        "best_in_newest_2",
        "tail_flat",
        "converged",
        "convergence_path",
    ):
        row[key(name)] = ""


def _update_row(
    row: dict[str, str],
    *,
    workspace: Path,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_wsl_cwd: str,
) -> dict[str, str]:
    family_id = str(row.get("family_id", "")).strip()
    primary_run_dir = str(row.get("run_dir", "")).strip()
    curve_summary, convergence, curve_summary_path, convergence_path, remote_curve_summary_path, remote_convergence_path = _load_run_summaries(
        run_dir=primary_run_dir,
        workspace=workspace,
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        remote_wsl_cwd=remote_wsl_cwd,
    )
    _write_summary_fields(
        row,
        prefix="",
        curve_summary=curve_summary,
        convergence=convergence,
        curve_summary_path=curve_summary_path,
        convergence_path=convergence_path,
        remote_curve_summary_path=remote_curve_summary_path,
        remote_convergence_path=remote_convergence_path,
    )
    gap_payload, gap_path = _load_gap_report_for_family(workspace=workspace, family_id=family_id)
    _clear_gap_fields(row, prefix="")
    if gap_payload:
        _write_gap_fields(row, prefix="", gap_payload=gap_payload, gap_path=gap_path)
    active_run_dir = str(row.get("active_run_dir", "")).strip()
    _clear_summary_fields(row, prefix="active_")
    _clear_gap_fields(row, prefix="active_")
    if active_run_dir and active_run_dir != primary_run_dir:
        active_curve, active_conv, active_curve_path, active_conv_path, active_remote_curve, active_remote_conv = _load_run_summaries(
            run_dir=active_run_dir,
            workspace=workspace,
            host=host,
            port=port,
            user=user,
            wsl_distro=wsl_distro,
            remote_wsl_cwd=remote_wsl_cwd,
        )
        _write_summary_fields(
            row,
            prefix="active_",
            curve_summary=active_curve,
            convergence=active_conv,
            curve_summary_path=active_curve_path,
            convergence_path=active_conv_path,
            remote_curve_summary_path=active_remote_curve,
            remote_convergence_path=active_remote_conv,
        )
        if gap_payload and active_curve:
            _write_gap_fields(row, prefix="active_", gap_payload=gap_payload, gap_path=gap_path)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the round-2 manifest with latest curve-summary fields.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--family-id", default="")
    parser.add_argument("--decision-status", default="")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    args = parser.parse_args()

    manifest = Path(args.manifest_csv).expanduser().resolve()
    workspace = manifest.parents[4]
    rows = _read_rows(manifest)
    family_id = str(args.family_id).strip()
    updated_rows: list[dict[str, str]] = []
    for row in rows:
        if family_id and str(row.get("family_id", "")).strip() != family_id:
            updated_rows.append(row)
            continue
        updated = _update_row(
            row,
            workspace=workspace,
            host=str(args.host),
            port=int(args.port),
            user=str(args.user),
            wsl_distro=str(args.wsl_distro),
            remote_wsl_cwd=str(args.remote_wsl_cwd),
        )
        if str(args.decision_status).strip():
            updated["decision_status"] = str(args.decision_status).strip()
        updated_rows.append(updated)
    fieldnames = _fieldnames(updated_rows)
    summary_keys = [
        "settled_eval_rows",
        "latest_epoch",
        "latest_checkpoint",
        "latest_transfer_clip_style",
        "latest_transfer_content_lpips",
        "latest_all_pairs_clip_style",
        "latest_all_pairs_content_lpips",
        "best_transfer_epoch",
        "best_transfer_clip_style",
        "best_transfer_content_lpips",
        "best_all_pairs_epoch",
        "best_all_pairs_clip_style",
        "best_all_pairs_content_lpips",
        "curve_summary_path",
        "pareto_epochs",
        "last_pareto_epoch",
        "since_last_pareto",
        "best_in_newest_2",
        "tail_flat",
        "converged",
        "convergence_path",
    ]
    gap_keys = [
        "gap_reference_name",
        "gap_report_path",
        "latest_transfer_style_gap",
        "latest_transfer_lpips_gap",
        "latest_all_pairs_style_gap",
        "latest_all_pairs_lpips_gap",
        "best_transfer_transfer_style_gap",
        "best_transfer_transfer_lpips_gap",
        "best_transfer_all_pairs_style_gap",
        "best_transfer_all_pairs_lpips_gap",
        "best_all_pairs_transfer_style_gap",
        "best_all_pairs_transfer_lpips_gap",
        "best_all_pairs_all_pairs_style_gap",
        "best_all_pairs_all_pairs_lpips_gap",
    ]
    for key in summary_keys + gap_keys + [f"active_{name}" for name in summary_keys + gap_keys]:
        if key not in fieldnames:
            fieldnames.append(key)
    _write_rows(manifest, updated_rows, fieldnames=fieldnames)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
