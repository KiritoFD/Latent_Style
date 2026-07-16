"""
Exp C: ArtFID manifest audit (Section 6 of reviewer_audit_and_required_experiments.md).

Goals
-----
1. Build the canonical D5-512 pair manifest from the source dataset
   (G:\\GitHub\\Latent_Style\\Dataset\\distinct5_512\\test\\{style}\\*.jpg).
2. For each paper method (IDT, WEAVE, SaMam, Seedream, Z-STAR, StyleAligned),
   walk its D5-512 output directory and:
     - count .png files
     - extract the referenced source image (style + slug)
     - flag missing/extra entries vs the canonical manifest
3. Read results/_artfid_details.json for per-target ArtFID/raw-FID/source-LPIPS
   and emit a component table indexed by (method, target_style).
4. Save:
     - canonical_pair_manifest.csv   (750 rows: src_style, src_slug, src_path, tgt_style)
     - per_method_file_audit.csv     (method, target_style, file_count, expected, missing, extra)
     - artfid_component_table.csv    (method, target_style, count, raw_fid, source_lpips, artfid)
     - artfid_audit_summary.json     (high-level findings + manifest_inconsistency evidence)
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (local G: drive where the dataset + generated outputs live).
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test")
RESULTS_ROOT = Path(r"G:\GitHub\Latent_Style\WEAVE\results")
D5_ROOT = RESULTS_ROOT / "D5-512"
ARTFID_DETAILS = RESULTS_ROOT / "_artfid_details.json"

OUT_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\rebuttal_exps\experiments\rebuttal_20260716\artfid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

# Method name mapping: paper label -> directory under D5-512.
PAPER_METHODS = {
    "IDT":          "identity",
    "WEAVE":        "weave_oriented_e4",
    "SaMam":        "samam",
    "Seedream 4.5": "seedream",
    "Z-STAR":       "zstar",
    "StyleAligned": "stylealigned",
}


# ---------------------------------------------------------------------------
# Step 1. Build canonical manifest from source dataset.
# ---------------------------------------------------------------------------
def build_canonical_manifest() -> list[dict]:
    rows: list[dict] = []
    for style in STYLES:
        style_dir = DATASET_ROOT / style
        if not style_dir.is_dir():
            print(f"[WARN] source dir missing: {style_dir}")
            continue
        for jpg in sorted(style_dir.glob("*.jpg")):
            # Source filename convention: "{style}__{artist}_{title}.jpg"
            # We keep the full stem as the slug so it can be matched against
            # generated filenames of the form "{style}_{slug}_to_{tgt}.png".
            slug = jpg.stem
            rows.append({
                "src_style": style,
                "src_slug": slug,
                "src_path": str(jpg),
            })
    return rows


def write_manifest_csv(manifest: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["src_style", "src_slug", "src_path"])
        w.writeheader()
        w.writerows(manifest)


# ---------------------------------------------------------------------------
# Step 2. Per-method file audit.
# ---------------------------------------------------------------------------
# Generated filename patterns observed in the wild (two distinct conventions):
#
#   Convention A (IDT/SaMam/Z-STAR/StyleAligned and older methods):
#       "{src_style}__{slug}__to__{tgt}.png"
#       where {slug} = "{style}__{artist}_{title}" (the source .jpg stem).
#       Example:
#         Early_Renaissance__Early_Renaissance__andrea-mantegna_adoration-...__to__Early_Renaissance.png
#       Separator is "__to__" (double underscore on both sides).
#
#   Convention B (weave_oriented_e4 / seedream):
#       "{src_style}_{slug}_to_{tgt}.png"
#       where {slug} = "{style}__{artist}_{title}".
#       Example:
#         Early_Renaissance_Early_Renaissance__andrea-mantegna_adoration-..._to_Early_Renaissance.png
#       Separator is "_to_" (single underscore on both sides).
#
# We must handle both. Strategy: look for the LAST occurrence of a token
# matching "__to__{tgt}" or "_to_{tgt}" where {tgt} is one of the 5 known
# style names. The src_desc is everything before that token.
TO_TOKEN_DOUBLE = "__to__"
TO_TOKEN_SINGLE = "_to_"


def parse_generated_name(fname: str) -> tuple[str | None, str | None]:
    """Return (src_descriptor, tgt_style) or (None, None) if unparseable.

    Tries the double-underscore separator first, then the single-underscore
    separator. Always picks the LAST occurrence so that "_to_" inside a slug
    (rare but possible) doesn't cause a false split.
    """
    stem = fname
    if stem.endswith(".png"):
        stem = stem[:-4]

    # Try convention A: "__to__{tgt}".
    for tok in (TO_TOKEN_DOUBLE, TO_TOKEN_SINGLE):
        # Search from the right for "{tok}{tgt_style}".
        for tgt in STYLES:
            suffix = f"{tok}{tgt}"
            idx = stem.rfind(suffix)
            if idx >= 0 and stem[idx + len(suffix):] == "":
                src_desc = stem[:idx]
                if src_desc:
                    return src_desc, tgt
    return None, None


def src_descriptor_to_slug(src_desc: str) -> tuple[str | None, str | None]:
    """Map a source descriptor back to (src_style, src_slug).

    The canonical src_slug is the stem of the source .jpg file, which has the
    form "{style}__{artist}_{title}". Generated filenames encode it as:
      - Convention A: "{src_style}__{slug}"  -> drop the leading "{style}__".
      - Convention B: "{src_style}_{slug}"   -> drop the leading "{style}_".
    In both cases, if the descriptor already starts with "{style}__{style}__"
    (i.e. the style prefix is doubled), we strip just the first "{style}_"
    or "{style}__" and keep the rest as the slug.
    """
    for style in STYLES:
        # Convention B: "{style}_{style}__..." -> slug = "{style}__..."
        prefix_b = f"{style}_{style}__"
        if src_desc.startswith(prefix_b):
            return style, src_desc[len(style) + 1:]
        # Convention A: "{style}__{style}__..." -> slug = "{style}__..."
        prefix_a = f"{style}__{style}__"
        if src_desc.startswith(prefix_a):
            return style, src_desc[len(style) + 2:]
        # Convention A simpler: "{style}__{artist}_..." -> slug = descriptor
        # (only if the second token is NOT a style name).
        prefix_a_simple = f"{style}__"
        if src_desc.startswith(prefix_a_simple):
            rest = src_desc[len(style) + 2:]
            # If rest itself starts with a style + "__", it's the doubled case
            # already handled above. Otherwise rest is the artist_title and
            # the slug is the whole descriptor.
            rest_starts_with_style = any(rest.startswith(f"{s}__") for s in STYLES)
            if not rest_starts_with_style:
                return style, src_desc
    return None, None


def audit_method(method_label: str, method_dir: Path, manifest: list[dict]) -> dict:
    """Walk the method directory, parse filenames, compare against manifest."""
    # Build expected set: (src_style, src_slug, tgt_style) for all 5 targets.
    expected = set()
    manifest_lookup = {(row["src_style"], row["src_slug"]): row for row in manifest}
    for row in manifest:
        for tgt in STYLES:
            expected.add((row["src_style"], row["src_slug"], tgt))

    found: set[tuple[str, str, str]] = set()
    file_count = 0
    unparseable: list[str] = []
    per_tgt_counts: dict[str, int] = {s: 0 for s in STYLES}

    if not method_dir.is_dir():
        return {
            "method": method_label,
            "dir": str(method_dir),
            "exists": False,
            "file_count": 0,
            "expected": len(expected),
            "missing": len(expected),
            "extra": 0,
            "per_tgt_counts": per_tgt_counts,
            "unparseable": [],
            "missing_examples": [],
            "extra_examples": [],
        }

    for png in method_dir.rglob("*.png"):
        if png.name == "_DONE":
            continue
        file_count += 1
        src_desc, tgt = parse_generated_name(png.name)
        if src_desc is None or tgt is None or tgt not in STYLES:
            unparseable.append(png.name)
            continue
        src_style, src_slug = src_descriptor_to_slug(src_desc)
        if src_style is None or src_slug is None:
            unparseable.append(png.name)
            continue
        found.add((src_style, src_slug, tgt))
        per_tgt_counts[tgt] += 1

    missing = expected - found
    extra = found - expected

    return {
        "method": method_label,
        "dir": str(method_dir),
        "exists": True,
        "file_count": file_count,
        "expected": len(expected),
        "missing": len(missing),
        "extra": len(extra),
        "per_tgt_counts": per_tgt_counts,
        "unparseable": unparseable[:10],
        "missing_examples": sorted(list(missing))[:10],
        "extra_examples": sorted(list(extra))[:10],
    }


# ---------------------------------------------------------------------------
# Step 3. Read existing ArtFID component data.
# ---------------------------------------------------------------------------
def load_artfid_components() -> dict:
    if not ARTFID_DETAILS.exists():
        return {}
    with ARTFID_DETAILS.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_component_table(artfid_data: dict, method_dir_map: dict[str, str]) -> list[dict]:
    """Return rows of (method, target_style, count, raw_fid, source_lpips, artfid)."""
    rows: list[dict] = []
    d5 = artfid_data.get("D5-512", {})
    for label, dirname in method_dir_map.items():
        entry = d5.get(dirname)
        if entry is None:
            # Try alternate spellings.
            for alt in (dirname.lower(), dirname.replace("_", "")):
                entry = d5.get(alt)
                if entry is not None:
                    break
        if entry is None:
            rows.append({
                "method": label,
                "dir": dirname,
                "target_style": "ALL",
                "count": None,
                "raw_fid": None,
                "source_lpips": None,
                "artfid": None,
                "note": "not found in _artfid_details.json",
            })
            continue
        # Aggregate row.
        rows.append({
            "method": label,
            "dir": dirname,
            "target_style": "ALL",
            "count": entry.get("count"),
            "raw_fid": entry.get("art_fid_fid"),
            "source_lpips": entry.get("art_fid_content_lpips"),
            "artfid": entry.get("art_fid"),
            "note": "",
        })
        # Per-target rows.
        for pt in entry.get("per_target", []):
            rows.append({
                "method": label,
                "dir": dirname,
                "target_style": pt.get("target_style"),
                "count": pt.get("count"),
                "raw_fid": pt.get("art_fid_fid"),
                "source_lpips": pt.get("art_fid_content_lpips"),
                "artfid": pt.get("art_fid"),
                "note": "",
            })
    return rows


def write_component_csv(rows: list[dict], path: Path) -> None:
    fields = ["method", "dir", "target_style", "count", "raw_fid", "source_lpips", "artfid", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_audit_csv(audits: list[dict], path: Path) -> None:
    fields = ["method", "dir", "exists", "file_count", "expected", "missing", "extra",
              "Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e",
              "unparseable_examples", "missing_examples", "extra_examples"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in audits:
            row = {
                "method": a["method"],
                "dir": a["dir"],
                "exists": a["exists"],
                "file_count": a["file_count"],
                "expected": a["expected"],
                "missing": a["missing"],
                "extra": a["extra"],
            }
            for s in STYLES:
                row[s] = a["per_tgt_counts"].get(s, 0)
            row["unparseable_examples"] = "; ".join(a.get("unparseable", [])[:3])
            row["missing_examples"] = "; ".join(
                f"{m[0]}/{m[1][:40]}/{m[2]}" for m in a.get("missing_examples", [])[:3]
            )
            row["extra_examples"] = "; ".join(
                f"{m[0]}/{m[1][:40]}/{m[2]}" for m in a.get("extra_examples", [])[:3]
            )
            w.writerow(row)


# ---------------------------------------------------------------------------
# Step 4. Summary + manifest inconsistency evidence.
# ---------------------------------------------------------------------------
def build_summary(manifest: list[dict], audits: list[dict], components: list[dict]) -> dict:
    # Per-style manifest counts.
    manifest_per_style = defaultdict(int)
    for row in manifest:
        manifest_per_style[row["src_style"]] += 1

    # Method-level summary.
    method_summary = []
    for a in audits:
        method_summary.append({
            "method": a["method"],
            "file_count": a["file_count"],
            "expected": a["expected"],
            "missing": a["missing"],
            "extra": a["extra"],
            "manifest_consistent": (a["missing"] == 0 and a["extra"] == 0),
        })

    # Identify manifest inconsistency evidence (methods with missing > 0).
    inconsistent = [m for m in method_summary if not m["manifest_consistent"]]

    # Component-table aggregate view (one row per method, ALL target).
    aggregate_components = [r for r in components if r["target_style"] == "ALL"]

    # Cross-method source-manifest overlap: for each pair of methods, compute
    # the Jaccard index of their (src_style, src_slug) sets. This tells us
    # whether Z-STAR and StyleAligned at least use the same manifest as each
    # other (even if it differs from the canonical one).
    method_src_sets: dict[str, set[tuple[str, str]]] = {}
    for a in audits:
        s: set[tuple[str, str]] = set()
        for ex in a.get("missing_examples", []):
            # missing_examples are (src_style, src_slug, tgt) tuples from the
            # canonical manifest that the method did NOT produce. We want the
            # method's OWN source set, so we need to recompute from found.
            pass
        # Recompute from the audit's stored extra_examples (which ARE in the
        # method's manifest but not canonical). But extra_examples is capped
        # at 10. Instead, we re-walk the directory here for accuracy.
        method_dir = Path(a["dir"])
        if method_dir.is_dir():
            for png in method_dir.rglob("*.png"):
                if png.name == "_DONE":
                    continue
                src_desc, tgt = parse_generated_name(png.name)
                if src_desc is None or tgt is None:
                    continue
                src_style, src_slug = src_descriptor_to_slug(src_desc)
                if src_style is not None and src_slug is not None:
                    s.add((src_style, src_slug))
        method_src_sets[a["method"]] = s

    canonical_src_set = {(row["src_style"], row["src_slug"]) for row in manifest}
    method_src_sets["CANONICAL"] = canonical_src_set

    cross_overlap = []
    method_names = list(method_src_sets.keys())
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i + 1:]:
            s1, s2 = method_src_sets[m1], method_src_sets[m2]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            jaccard = inter / union if union > 0 else 0.0
            cross_overlap.append({
                "method_a": m1,
                "method_b": m2,
                "intersection": inter,
                "union": union,
                "jaccard": round(jaccard, 4),
            })

    return {
        "canonical_manifest_size": len(manifest),
        "manifest_per_style": dict(manifest_per_style),
        "expected_pairs_per_method": len(manifest) * len(STYLES),
        "method_summary": method_summary,
        "manifest_inconsistent_methods": inconsistent,
        "aggregate_components": aggregate_components,
        "cross_method_manifest_overlap": cross_overlap,
        "protocol_note": (
            "ArtFID values are read as-is from results/_artfid_details.json. "
            "Recomputation with a frozen canonical manifest is documented as a "
            "second-batch task if any method's missing/extra count is non-zero."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[expC] output dir: {OUT_DIR}")

    print("[expC] step 1: build canonical manifest from source dataset...")
    manifest = build_canonical_manifest()
    manifest_path = OUT_DIR / "canonical_pair_manifest.csv"
    write_manifest_csv(manifest, manifest_path)
    print(f"[expC]   manifest size = {len(manifest)} (expected 750 = 5 styles x 150)")
    per_style = defaultdict(int)
    for row in manifest:
        per_style[row["src_style"]] += 1
    for s in STYLES:
        print(f"[expC]   {s}: {per_style[s]} source images")

    print("[expC] step 2: per-method file audit...")
    audits = []
    for label, dirname in PAPER_METHODS.items():
        method_dir = D5_ROOT / dirname
        print(f"[expC]   auditing {label} ({dirname}) -> {method_dir}")
        a = audit_method(label, method_dir, manifest)
        audits.append(a)
        print(f"[expC]     files={a['file_count']} expected={a['expected']} "
              f"missing={a['missing']} extra={a['extra']}")
    audit_path = OUT_DIR / "per_method_file_audit.csv"
    write_audit_csv(audits, audit_path)

    print("[expC] step 3: read existing ArtFID components...")
    artfid_data = load_artfid_components()
    components = extract_component_table(artfid_data, PAPER_METHODS)
    comp_path = OUT_DIR / "artfid_component_table.csv"
    write_component_csv(components, comp_path)
    print(f"[expC]   component rows = {len(components)}")

    print("[expC] step 4: build summary...")
    summary = build_summary(manifest, audits, components)
    summary_path = OUT_DIR / "artfid_audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("Exp C ArtFID manifest audit - SUMMARY")
    print("=" * 72)
    print(f"Canonical manifest size: {summary['canonical_manifest_size']} source images")
    print(f"Expected pairs per method: {summary['expected_pairs_per_method']} "
          f"(manifest x 5 target styles)")
    print()
    print("Per-method file audit:")
    print(f"  {'Method':<16} {'Files':>6} {'Expected':>9} {'Missing':>8} {'Extra':>6} {'OK':>4}")
    for m in summary["method_summary"]:
        ok = "Y" if m["manifest_consistent"] else "N"
        print(f"  {m['method']:<16} {m['file_count']:>6} {m['expected']:>9} "
              f"{m['missing']:>8} {m['extra']:>6} {ok:>4}")
    print()
    print("Manifest-inconsistent methods (evidence of unequal source manifests):")
    for m in summary["manifest_inconsistent_methods"]:
        print(f"  - {m['method']}: missing={m['missing']} extra={m['extra']} "
              f"(file_count={m['file_count']}, expected={m['expected']})")
    print()
    print("Aggregate ArtFID components (read from existing _artfid_details.json):")
    print(f"  {'Method':<16} {'Count':>6} {'raw FID':>10} {'src LPIPS':>10} {'ArtFID':>10}")
    for r in summary["aggregate_components"]:
        cnt = r.get("count") if r.get("count") is not None else "-"
        fid = f"{r['raw_fid']:.4f}" if r.get("raw_fid") is not None else "-"
        lp = f"{r['source_lpips']:.4f}" if r.get("source_lpips") is not None else "-"
        af = f"{r['artfid']:.4f}" if r.get("artfid") is not None else "-"
        print(f"  {r['method']:<16} {cnt:>6} {fid:>10} {lp:>10} {af:>10}")
    print()
    print("Cross-method source-manifest overlap (Jaccard index):")
    print(f"  {'Method A':<16} {'Method B':<16} {'Inter':>6} {'Union':>6} {'Jaccard':>8}")
    for co in summary.get("cross_method_manifest_overlap", []):
        print(f"  {co['method_a']:<16} {co['method_b']:<16} "
              f"{co['intersection']:>6} {co['union']:>6} {co['jaccard']:>8.4f}")
    print()
    print(f"Outputs written to: {OUT_DIR}")
    print(f"  - {manifest_path.name}")
    print(f"  - {audit_path.name}")
    print(f"  - {comp_path.name}")
    print(f"  - {summary_path.name}")


if __name__ == "__main__":
    main()
