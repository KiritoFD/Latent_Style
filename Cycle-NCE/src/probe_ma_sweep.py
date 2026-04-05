from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import torch

from probe_ma import MAProbe, _build_probe_batch, _load_config, _load_model, _resolve_device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch sweep MA probes and build an interactive viewer.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to probe. Required unless --input-glob is used.")
    parser.add_argument("--config", type=str, default=None, help="Optional external config json.")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu. Defaults to CUDA when available.")
    parser.add_argument("--num-samples", type=int, default=8, help="Latents per probe batch.")
    parser.add_argument("--source-style", type=str, default="photo", help="Source style when sweeping checkpoint probes.")
    parser.add_argument("--target-style", type=str, default=None, help="Single target style. Ignored when sweeping all targets/pairs.")
    parser.add_argument("--all-targets", action="store_true", help="Probe all target styles for one source style.")
    parser.add_argument("--all-style-pairs", action="store_true", help="Probe every source->target style pair.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--top-k-spatial", type=int, default=6, help="Hotspots kept in each per-layer summary.")
    parser.add_argument("--input-glob", type=str, default=None, help="Glob of existing ma_probe*.json files to aggregate.")
    parser.add_argument("--output-dir", type=str, default=".", help="Where to write summary csv/html/json.")
    parser.add_argument("--output-prefix", type=str, default="ma_probe_sweep", help="Basename for generated files.")
    return parser.parse_args()


def _style_pairs(style_names: list[str], source_style: str | None, target_style: str | None, *, all_targets: bool, all_pairs: bool) -> list[tuple[str, str]]:
    if all_pairs:
        return [(src, tgt) for src in style_names for tgt in style_names if src != tgt]
    if all_targets:
        src = source_style or style_names[0]
        return [(src, tgt) for tgt in style_names if tgt != src]
    src = source_style or style_names[0]
    tgt = target_style
    if tgt is None:
        tgt = next((name for name in style_names if name != src), style_names[0])
    return [(src, tgt)]


def _infer_run_name(payload: dict[str, Any], fallback_name: str) -> str:
    styles = payload.get("styles") or []
    source_id = payload.get("source_style_id")
    target_id = payload.get("target_style_id")
    try:
        src = styles[int(source_id)]
        tgt = styles[int(target_id)]
        return f"{src} -> {tgt}"
    except Exception:
        return fallback_name


def _stage_stats(summaries: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for stage in ("stem", "hires", "body", "skip", "decoder", "output", "other"):
        vals = [float(item["max_ma_ratio"]) for item in summaries if item.get("stage") == stage]
        out[f"{stage}_max"] = max(vals) if vals else None
        out[f"{stage}_mean"] = (sum(vals) / len(vals)) if vals else None
    return out


def _top_layer(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"name": None, "stage": None, "max_ma_ratio": None, "mean_ma_ratio": None}
    best = max(summaries, key=lambda item: float(item.get("max_ma_ratio", float("-inf"))))
    return {
        "name": best.get("name"),
        "stage": best.get("stage"),
        "max_ma_ratio": float(best.get("max_ma_ratio", 0.0)),
        "mean_ma_ratio": float(best.get("mean_ma_ratio", 0.0)),
    }


def _make_record(payload: dict[str, Any], run_name: str, source: str) -> dict[str, Any]:
    summaries = list(payload.get("summaries") or [])
    stage = _stage_stats(summaries)
    top = _top_layer(summaries)
    styles = list(payload.get("styles") or [])
    source_id = payload.get("source_style_id")
    target_id = payload.get("target_style_id")
    source_style = styles[int(source_id)] if styles and source_id is not None and int(source_id) < len(styles) else None
    target_style = styles[int(target_id)] if styles and target_id is not None and int(target_id) < len(styles) else None
    record = {
        "run_name": run_name,
        "source": source,
        "checkpoint": payload.get("checkpoint"),
        "source_style": source_style,
        "target_style": target_style,
        "top_layer_name": top["name"],
        "top_layer_stage": top["stage"],
        "top_layer_max_ma_ratio": top["max_ma_ratio"],
        "top_layer_mean_ma_ratio": top["mean_ma_ratio"],
        "num_layers": len(summaries),
    }
    record.update(stage)
    return record


def _probe_single(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    source_style: str,
    target_style: str,
    num_samples: int,
    seed: int,
    top_k_spatial: int,
) -> dict[str, Any]:
    probe_batch, style_order = _build_probe_batch(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        num_samples=num_samples,
        source_style=source_style,
        target_style=target_style,
        content_latents=None,
        target_latents=None,
        seed=seed,
    )

    probe = MAProbe(model, top_k_spatial=top_k_spatial)
    probe.attach()
    with torch.no_grad():
        _ = model(
            probe_batch.content,
            style_id=probe_batch.target_style_ids,
            target_style_latent=probe_batch.target,
        )
    probe.remove()

    return {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "styles": style_order,
        "source_style_id": int(probe_batch.source_style_id),
        "target_style_id": int(probe_batch.target_style_ids[0].item()),
        "content_paths": [str(p) for p in probe_batch.content_paths],
        "target_paths": [str(p) for p in probe_batch.target_paths],
        "summaries": probe.summarize(),
    }


def _load_payloads_from_glob(pattern: str) -> list[tuple[str, dict[str, Any]]]:
    matched = sorted(Path(p).resolve() for p in glob.glob(pattern, recursive=True))
    payloads = []
    for path in matched:
        payloads.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
    return payloads


def _collect_common_layers(records: list[dict[str, Any]], payloads: list[dict[str, Any]], limit: int = 20) -> list[str]:
    score: dict[str, float] = {}
    for payload in payloads:
        for summary in payload.get("summaries", []):
            name = str(summary.get("name"))
            score[name] = max(score.get(name, 0.0), float(summary.get("max_ma_ratio", 0.0)))
    ordered = sorted(score.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ordered[:limit]]


def _write_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _build_html(records: list[dict[str, Any]], payloads: list[dict[str, Any]], out_path: Path) -> None:
    layer_names = _collect_common_layers(records, payloads, limit=24)
    layer_matrix = []
    for payload in payloads:
        layer_map = {str(s["name"]): float(s["max_ma_ratio"]) for s in payload.get("summaries", [])}
        layer_matrix.append([layer_map.get(name, None) for name in layer_names])

    page_data = {
        "records": records,
        "layer_names": layer_names,
        "layer_matrix": layer_matrix,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MA Probe Sweep Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #16202a;
      --muted: #5f6b76;
      --line: #d9dee5;
      --accent: #17324d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    .header {{ margin-bottom: 18px; }}
    .header h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
    .header p {{ margin: 0; color: var(--muted); }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 20px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); }}
    .stat-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .stat-value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    .plot {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 8px; margin-bottom: 16px; }}
    .table-wrap {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: #f9fafb; z-index: 1; }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>MA Probe Sweep Viewer</h1>
      <p>Batch overview of stage-level MA heat and hottest layers.</p>
    </div>

    <div class="stats">
      <div class="card"><div class="stat-label">Runs</div><div class="stat-value" id="statRuns">0</div></div>
      <div class="card"><div class="stat-label">Checkpoints</div><div class="stat-value" id="statCkpts">0</div></div>
      <div class="card"><div class="stat-label">Peak MA</div><div class="stat-value" id="statPeak">0</div></div>
      <div class="card"><div class="stat-label">Most Common Top Stage</div><div class="stat-value" id="statStage">-</div></div>
    </div>

    <div class="plot"><div id="stageBar" style="height: 480px;"></div></div>
    <div class="plot"><div id="stageScatter" style="height: 480px;"></div></div>
    <div class="plot"><div id="layerHeatmap" style="height: 720px;"></div></div>

    <div class="table-wrap">
      <table id="summaryTable">
        <thead>
          <tr>
            <th>Run</th>
            <th>Checkpoint</th>
            <th>Top Layer</th>
            <th>Top Stage</th>
            <th>Peak MA</th>
            <th>Hires</th>
            <th>Body</th>
            <th>Skip</th>
            <th>Decoder</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <script>
    const pageData = {json.dumps(page_data, ensure_ascii=False)};
    const records = pageData.records || [];

    function fmt(v) {{
      return (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(2);
    }}

    function buildStats() {{
      document.getElementById('statRuns').textContent = String(records.length);
      document.getElementById('statCkpts').textContent = String(new Set(records.map(r => r.checkpoint).filter(Boolean)).size);
      const peak = Math.max(...records.map(r => Number(r.top_layer_max_ma_ratio || 0)), 0);
      document.getElementById('statPeak').textContent = peak.toFixed(2);
      const counts = new Map();
      for (const r of records) {{
        const k = r.top_layer_stage || 'n/a';
        counts.set(k, (counts.get(k) || 0) + 1);
      }}
      let bestStage = '-';
      let bestCount = -1;
      for (const [k, v] of counts.entries()) {{
        if (v > bestCount) {{ bestCount = v; bestStage = k; }}
      }}
      document.getElementById('statStage').textContent = bestStage;
    }}

    function renderStageBar() {{
      const runNames = records.map(r => r.run_name);
      const stages = ['stem_max', 'hires_max', 'body_max', 'skip_max', 'decoder_max', 'output_max'];
      const labels = {{
        stem_max: 'stem',
        hires_max: 'hires',
        body_max: 'body',
        skip_max: 'skip',
        decoder_max: 'decoder',
        output_max: 'output'
      }};
      const colors = ['#9aa5b1', '#d94841', '#f59f00', '#2b8a3e', '#1971c2', '#5f3dc4'];
      const traces = stages.map((stageKey, idx) => {{
        return {{
          type: 'bar',
          name: labels[stageKey],
          x: runNames,
          y: records.map(r => r[stageKey]),
          marker: {{ color: colors[idx] }}
        }};
      }});
      Plotly.newPlot('stageBar', traces, {{
        barmode: 'group',
        title: 'Stage Peak MA Ratio by Run',
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: {{ tickangle: -35 }},
        yaxis: {{ title: 'max MA ratio' }},
        margin: {{ l: 60, r: 20, t: 50, b: 140 }}
      }}, {{ responsive: true, displaylogo: false }});
    }}

    function renderStageScatter() {{
      Plotly.newPlot('stageScatter', [{{
        type: 'scatter',
        mode: 'markers+text',
        x: records.map(r => r.hires_max),
        y: records.map(r => r.decoder_max),
        text: records.map(r => r.run_name),
        textposition: 'top center',
        customdata: records.map(r => [r.top_layer_name, r.top_layer_max_ma_ratio, r.skip_max]),
        marker: {{
          size: records.map(r => 10 + Number(r.top_layer_max_ma_ratio || 0)),
          color: records.map(r => r.skip_max),
          colorscale: 'YlOrRd',
          showscale: true,
          colorbar: {{ title: 'skip max' }},
          line: {{ color: '#17324d', width: 1 }}
        }},
        hovertemplate: '<b>%{{text}}</b><br>hires=%{{x:.2f}}<br>decoder=%{{y:.2f}}<br>top=%{{customdata[0]}}<br>top ratio=%{{customdata[1]:.2f}}<br>skip=%{{customdata[2]:.2f}}<extra></extra>'
      }}], {{
        title: 'Hires vs Decoder MA',
        xaxis: {{ title: 'hires max MA ratio' }},
        yaxis: {{ title: 'decoder max MA ratio' }},
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        margin: {{ l: 60, r: 20, t: 50, b: 60 }}
      }}, {{ responsive: true, displaylogo: false }});
    }}

    function renderLayerHeatmap() {{
      Plotly.newPlot('layerHeatmap', [{{
        type: 'heatmap',
        x: pageData.layer_names,
        y: records.map(r => r.run_name),
        z: pageData.layer_matrix,
        colorscale: 'YlOrRd',
        colorbar: {{ title: 'max MA ratio' }},
        hovertemplate: 'run=%{{y}}<br>layer=%{{x}}<br>ratio=%{{z:.2f}}<extra></extra>'
      }}], {{
        title: 'Top Layer Heatmap',
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: {{ tickangle: -40 }},
        margin: {{ l: 140, r: 20, t: 50, b: 180 }}
      }}, {{ responsive: true, displaylogo: false }});
    }}

    function renderTable() {{
      const tbody = document.querySelector('#summaryTable tbody');
      tbody.innerHTML = records.map(r => `
        <tr>
          <td>${{r.run_name}}</td>
          <td class="muted">${{r.checkpoint ? r.checkpoint.split(/[\\\\/]/).slice(-1)[0] : '-'}}</td>
          <td>${{r.top_layer_name || '-'}}</td>
          <td>${{r.top_layer_stage || '-'}}</td>
          <td>${{fmt(r.top_layer_max_ma_ratio)}}</td>
          <td>${{fmt(r.hires_max)}}</td>
          <td>${{fmt(r.body_max)}}</td>
          <td>${{fmt(r.skip_max)}}</td>
          <td>${{fmt(r.decoder_max)}}</td>
        </tr>
      `).join('');
    }}

    buildStats();
    renderStageBar();
    renderStageScatter();
    renderLayerHeatmap();
    renderTable();
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_sources: list[tuple[str, dict[str, Any]]] = []

    if args.input_glob:
        payload_sources.extend(_load_payloads_from_glob(args.input_glob))

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        device = _resolve_device(args.device)
        config = _load_config(checkpoint_path, args.config)
        style_names = list(config.get("data", {}).get("style_subdirs", []))
        if not style_names:
            raise KeyError("config.data.style_subdirs is required to sweep checkpoint probes")
        pairs = _style_pairs(
            style_names,
            args.source_style,
            args.target_style,
            all_targets=bool(args.all_targets),
            all_pairs=bool(args.all_style_pairs),
        )
        model = _load_model(checkpoint_path, config["model"], device)
        for idx, (src, tgt) in enumerate(pairs):
            payload = _probe_single(
                model=model,
                config=config,
                checkpoint_path=checkpoint_path,
                device=device,
                source_style=src,
                target_style=tgt,
                num_samples=max(1, int(args.num_samples)),
                seed=int(args.seed) + idx * 101,
                top_k_spatial=max(1, int(args.top_k_spatial)),
            )
            payload_sources.append((f"{src}->{tgt}", payload))

    if not payload_sources:
        raise ValueError("No input data. Provide --checkpoint or --input-glob.")

    payloads = [payload for _, payload in payload_sources]
    records = []
    for source, payload in payload_sources:
        run_name = _infer_run_name(payload, Path(source).stem)
        records.append(_make_record(payload, run_name, source))

    csv_path = output_dir / f"{args.output_prefix}.csv"
    json_path = output_dir / f"{args.output_prefix}.json"
    html_path = output_dir / f"{args.output_prefix}.html"

    _write_csv(records, csv_path)
    json_path.write_text(json.dumps({"records": records, "payloads": payloads}, indent=2), encoding="utf-8")
    _build_html(records, payloads, html_path)

    print(f"[ma-sweep] csv:  {csv_path}")
    print(f"[ma-sweep] json: {json_path}")
    print(f"[ma-sweep] html: {html_path}")


if __name__ == "__main__":
    main()
