"""Download the art-domain Inception checkpoint used by ArtFID."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

URL = "https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth"
DST = r"G:\GitHub\Latent_Style\eval_cache\artfid\art_inception.pth"


def main() -> int:
    dst = Path(DST)
    if dst.exists() and dst.stat().st_size > 10 * 1024 * 1024:
        print(f"Already exists: {dst} ({dst.stat().st_size} bytes)")
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".pth.tmp")
    print(f"Downloading {URL} -> {dst}")
    with requests.get(URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        print(f"Total size: {total} bytes ({total / 1024 / 1024:.1f} MB)")
        done = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if done % (16 << 20) == 0:
                        pct = (done / total * 100) if total else 0
                        print(f"  {done / 1024 / 1024:.1f} MB ({pct:.1f}%)")
    os.replace(tmp, dst)
    print(f"Done: {dst} ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
