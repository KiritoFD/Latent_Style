from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _update_config(
    path: Path,
    *,
    data_root: str,
    latent_cache_dir: str,
    pairing_cache_path: str,
    dino_cache_path: str | None,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data_cfg = payload.setdefault("data", {})
    data_cfg["data_root"] = str(data_root)
    data_cfg["latent_cache_dir"] = str(latent_cache_dir)
    data_cfg["pairing_cache_path"] = str(pairing_cache_path)
    if dino_cache_path is not None:
        data_cfg["dino_cache_path"] = str(dino_cache_path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retarget planned round-1 configs to a new latent train root/cache set.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--pairing-cache-path", required=True)
    parser.add_argument("--dino-cache-path", default="")
    parser.add_argument("--only-status", default="planned", help="Comma-separated decision_status values to retarget.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest_csv).resolve()
    rows = _read_rows(manifest)
    wanted_status = {x.strip().lower() for x in str(args.only_status).split(",") if x.strip()}
    touched: list[str] = []
    for row in rows:
        status = str(row.get("decision_status", "")).strip().lower()
        if wanted_status and status not in wanted_status:
            continue
        cfg_path = Path(str(row.get("config_path", "")).strip())
        if not cfg_path.exists():
            continue
        touched.append(str(row.get("family_id", "")).strip())
        if not bool(args.dry_run):
            _update_config(
                cfg_path,
                data_root=str(args.data_root),
                latent_cache_dir=str(args.latent_cache_dir),
                pairing_cache_path=str(args.pairing_cache_path),
                dino_cache_path=str(args.dino_cache_path).strip() or None,
            )
    if not bool(args.dry_run):
        _write_rows(manifest, rows)
    print("\n".join(touched))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
