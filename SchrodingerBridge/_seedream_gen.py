import os, sys, json, time, base64, io, csv, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

csv.field_size_limit(10**7)

ENDPOINT = "https://windhub.cc/v1/images/generations"
MODEL = "doubao-seedream-4-5-251128"
DEFAULT_KEY = "sk-pzIjYc4Eyoe75pmp201WnJinxHM8jfCaln1KoCaNoET8sXsV"

ROOT = r"G:\GitHub\Latent_Style\SchrodingerBridge"
MANIFEST = os.path.join(ROOT, "results", "R5-WikiArt", "seedream", "seedream_manifest.csv")
OUT_ROOT = os.path.join(ROOT, "results", "R5-WikiArt", "seedream")


def b64_img(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def call(endpoint, key, model, prompt, img_b64, size="512x512", timeout=180):
    body = {
        "model": model,
        "prompt": prompt,
        "image": "data:image/jpeg;base64," + img_b64,
        "size": size,
        "response_format": "b64_json",
        "n": 1,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer %s" % key)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    req.add_header("Accept", "*/*")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8", "replace"))
            dt = time.time() - t0
            return True, dt, 200, resp
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        try:
            msg = e.read().decode("utf-8", "replace")
        except Exception:
            msg = str(e)
        return False, dt, e.code, msg
    except Exception as e:
        dt = time.time() - t0
        return False, dt, -1, str(e)[:300]


def load_manifest(path):
    if not os.path.exists(path):
        return [], {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["row_id"]: r for r in rows if r.get("row_id")}
    return rows, by_id


def build_todos(manifest_rows, manifest_by_id):
    # determine styles from manifest union
    styles = set()
    for r in manifest_rows:
        if r.get("src_style"):
            styles.add(r["src_style"])
        if r.get("tgt_style"):
            styles.add(r["tgt_style"])
    styles = sorted(styles)
    # dataset root from a sample src_path
    ds_root = None
    for r in manifest_rows:
        if r.get("src_path"):
            ds_root = os.path.dirname(os.path.dirname(r["src_path"]))
            break
    assert ds_root and os.path.isdir(ds_root), "cannot infer dataset root: %r" % ds_root
    print("STYLES:", styles)
    print("DATASET test root:", ds_root)

    # src images per style (from dataset, 30 each)
    src_imgs = {}
    for s in styles:
        d = os.path.join(ds_root, s)
        if not os.path.isdir(d):
            print("WARN no dataset dir for style", s)
            continue
        imgs = sorted([os.path.join(d, fn) for fn in os.listdir(d) if fn.lower().endswith((".jpg", ".png", ".jpeg"))])
        src_imgs[s] = imgs
        print("  %s: %d src images" % (s, len(imgs)))

    todos = []
    for src_style in styles:
        for src_path in src_imgs.get(src_style, []):
            bn = os.path.splitext(os.path.basename(src_path))[0]
            for tgt_style in styles:
                row_id = "%s/%s->%s" % (src_style, bn, tgt_style)
                stem = "%s_%s" % (src_style, bn)
                out_path = os.path.join(OUT_ROOT, tgt_style, "%s_to_%s.png" % (stem, tgt_style))
                existing = manifest_by_id.get(row_id)
                done = existing and existing.get("status") == "ok" and os.path.exists(out_path)
                if done:
                    continue
                prompt = "转成%s风格" % tgt_style
                todos.append({
                    "row_id": row_id, "src_style": src_style, "tgt_style": tgt_style,
                    "src_path": src_path, "out_path": out_path, "prompt": prompt,
                    "existing": existing,
                })
    return todos


def generate(todos, key, workers=4, limit=None):
    if limit:
        todos = todos[:limit]
    # cache b64 per src_path
    b64_cache = {}
    for t in todos:
        if t["src_path"] not in b64_cache:
            b64_cache[t["src_path"]] = b64_img(t["src_path"])

    lock = threading.Lock()
    results = {}  # row_id -> dict(status, elapsed, ...)
    n_done = [0]
    n_fail = [0]
    elapseds = []

    def worker(t):
        img_b64 = b64_cache[t["src_path"]]
        last_err = ""
        for attempt in range(1, 4):
            ok, dt, st, resp = call(ENDPOINT, key, MODEL, t["prompt"], img_b64)
            if ok:
                try:
                    b64 = resp["data"][0]["b64_json"]
                    os.makedirs(os.path.dirname(t["out_path"]), exist_ok=True)
                    with open(t["out_path"], "wb") as f:
                        f.write(base64.b64decode(b64))
                    with lock:
                        results[t["row_id"]] = {
                            "status": "ok", "elapsed_sec": round(dt, 3),
                            "request_elapsed_sec": round(dt, 3), "attempts": attempt,
                            "error": "", "out_path": t["out_path"], "prompt": t["prompt"],
                            "src_style": t["src_style"], "tgt_style": t["tgt_style"],
                            "src_path": t["src_path"],
                        }
                        elapseds.append(dt)
                        n_done[0] += 1
                        avg = sum(elapseds) / len(elapseds)
                        print("[OK %d/%d] %s (%.1fs, avg %.1fs, attempt %d)" % (
                            n_done[0], len(todos), t["row_id"], dt, avg, attempt), flush=True)
                    return
                except Exception as e:
                    last_err = "decode:%s" % e
            else:
                last_err = "%s:%s" % (st, str(resp)[:120])
            time.sleep(min(2 * attempt, 10))
        with lock:
            n_fail[0] += 1
            results[t["row_id"]] = {
                "status": "failed", "elapsed_sec": 0, "request_elapsed_sec": 0,
                "attempts": 3, "error": last_err, "out_path": t["out_path"],
                "prompt": t["prompt"], "src_style": t["src_style"], "tgt_style": t["tgt_style"],
                "src_path": t["src_path"],
            }
            print("[FAIL %d] %s -> %s" % (n_fail[0], t["row_id"], last_err), flush=True)

    print("=== generating %d todos with %d workers ===" % (len(todos), workers))
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, t) for t in todos]
        for fu in as_completed(futs):
            pass
    total = time.time() - t_start
    print("=== DONE: %d ok, %d failed, wall %.1fs, avg req %.1fs ===" % (
        n_done[0], n_fail[0], total, (sum(elapseds) / len(elapseds)) if elapseds else 0))
    return results


def update_manifest(manifest_rows, manifest_by_id, results):
    # merge results into rows
    for rid, res in results.items():
        if rid in manifest_by_id:
            for k, v in res.items():
                manifest_by_id[rid][k] = v
        else:
            row = {
                "row_id": rid, "status": res.get("status", ""),
                "src_style": res.get("src_style", ""), "tgt_style": res.get("tgt_style", ""),
                "src_path": res.get("src_path", ""), "out_path": res.get("out_path", ""),
                "prompt": res.get("prompt", ""), "size": "512x512",
                "elapsed_sec": res.get("elapsed_sec", ""),
                "request_elapsed_sec": res.get("request_elapsed_sec", ""),
                "download_elapsed_sec": "", "write_elapsed_sec": "", "rate_wait_sec": "",
                "attempts": res.get("attempts", ""), "source": "b64_json",
                "error": res.get("error", ""),
            }
            manifest_rows.append(row)
            manifest_by_id[rid] = row
    fields = ["row_id", "status", "src_style", "tgt_style", "src_path", "out_path",
              "prompt", "size", "elapsed_sec", "request_elapsed_sec", "download_elapsed_sec",
              "write_elapsed_sec", "rate_wait_sec", "attempts", "source", "error"]
    # backup
    if os.path.exists(MANIFEST):
        os.replace(MANIFEST, MANIFEST + ".bak")
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in manifest_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("manifest updated:", MANIFEST, "rows:", len(manifest_rows))


def main():
    global MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--manifest", default=MANIFEST)
    args = ap.parse_args()

    MANIFEST = args.manifest
    rows, by_id = load_manifest(MANIFEST)
    print("existing manifest rows:", len(rows))
    todos = build_todos(rows, by_id)
    print("TODO (to generate):", len(todos))
    if args.dry_run or args.test:
        if args.test and todos:
            t = todos[0]
            print("TEST call:", t["row_id"])
            ok, dt, st, resp = call(ENDPOINT, args.key, MODEL, t["prompt"], b64_img(t["src_path"]))
            print("test:", ok, dt, st, str(resp)[:200])
        return
    results = generate(todos, args.key, workers=args.workers, limit=args.limit or None)
    update_manifest(rows, by_id, results)


if __name__ == "__main__":
    main()
