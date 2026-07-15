from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input_csv.open("r", encoding="utf-8", newline="")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["comparison"]].append(row)

    out_rows = []
    for comparison, bucket in sorted(grouped.items()):
        for field in ["style_vote", "content_vote", "artifact_vote"]:
            winner_counts = Counter()
            for row in bucket:
                vote = row[field]
                if vote == "A":
                    winner_counts[row["candidate_a_method"]] += 1
                elif vote == "B":
                    winner_counts[row["candidate_b_method"]] += 1
                else:
                    winner_counts["Tie"] += 1
            methods = sorted({row["candidate_a_method"] for row in bucket} | {row["candidate_b_method"] for row in bucket})
            out_rows.append(
                {
                    "comparison": comparison,
                    "question": field.replace("_vote", ""),
                    "method_1": methods[0] if methods else "",
                    "method_1_wins": winner_counts.get(methods[0], 0) if methods else 0,
                    "method_2": methods[1] if len(methods) > 1 else "",
                    "method_2_wins": winner_counts.get(methods[1], 0) if len(methods) > 1 else 0,
                    "Tie": winner_counts.get("Tie", 0),
                    "n": len(bucket),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    args.output_json.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output_csv)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
