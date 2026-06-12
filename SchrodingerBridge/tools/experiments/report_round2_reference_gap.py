from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import subprocess


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


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


def _load_remote_json(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_path: str,
) -> dict[str, Any]:
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
        raise FileNotFoundError(f"Remote json not found: {remote_path}")
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected remote JSON object: {remote_path}")
    return payload


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _gap(point: dict[str, Any], ref: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for point_key, ref_key, out_key in (
        ("transfer_clip_style", "transfer_clip_style", "transfer_style_gap"),
        ("transfer_content_lpips", "transfer_content_lpips", "transfer_lpips_gap"),
        ("all_pairs_clip_style", "all_pairs_clip_style", "all_pairs_style_gap"),
        ("all_pairs_content_lpips", "all_pairs_content_lpips", "all_pairs_lpips_gap"),
    ):
        p = _metric(point, point_key)
        r = _metric(ref, ref_key)
        out[out_key] = None if p is None or r is None else p - r
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a round-2 curve summary against a chosen reference point.")
    parser.add_argument("--curve-summary-json", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--reference-transfer-style", type=float, required=True)
    parser.add_argument("--reference-transfer-lpips", type=float, required=True)
    parser.add_argument("--reference-allpairs-style", type=float, required=True)
    parser.add_argument("--reference-allpairs-lpips", type=float, required=True)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    args = parser.parse_args()

    raw_curve = str(args.curve_summary_json).strip()
    curve_path = Path(raw_curve).expanduser()
    if curve_path.is_file():
        payload = _load_json(curve_path.resolve())
        curve_summary_json = str(curve_path.resolve())
    elif raw_curve.startswith("/"):
        payload = _load_remote_json(
            host=str(args.host),
            port=int(args.port),
            user=str(args.user),
            wsl_distro=str(args.wsl_distro),
            remote_path=raw_curve,
        )
        curve_summary_json = raw_curve
    else:
        raise FileNotFoundError(f"Curve summary json not found: {raw_curve}")
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        rows = []

    ref = {
        "name": str(args.reference_name),
        "transfer_clip_style": float(args.reference_transfer_style),
        "transfer_content_lpips": float(args.reference_transfer_lpips),
        "all_pairs_clip_style": float(args.reference_allpairs_style),
        "all_pairs_content_lpips": float(args.reference_allpairs_lpips),
    }

    def _with_gap(item: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        out = dict(item)
        out.update(_gap(item, ref))
        return out

    result = {
        "curve_summary_json": curve_summary_json,
        "reference": ref,
        "latest": _with_gap(payload.get("latest")),
        "best_transfer": _with_gap(payload.get("best_transfer")),
        "best_all_pairs": _with_gap(payload.get("best_all_pairs")),
        "rows": [_with_gap(row) for row in rows if isinstance(row, dict)],
    }

    json_out = str(args.json_out).strip()
    if json_out:
        out_path = Path(json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out_path)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
