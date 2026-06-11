from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IDT_TRANSFER = 0.6399224616587162

WIKIARTS5_CURVE_CSV = WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samam_wikiarts5_patch8_segmented_20260610_094447" / "curve_metrics.csv"
SAMST_CURVE_CSV = WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samst_wikiarts5_wsl_20260610_172206" / "eval_bundle" / "clip_lpips_curve.csv"
ATTNSA_CURVE_CSV = ROOT / "round1_attn_sa_mod_fast_local" / "full_eval_fast_local" / "clip_lpips_curve.csv"
GATED_CURVE_CSV = ROOT / "round1_attn_gated_spade_remote_full_eval_pull" / "clip_lpips_curve.csv"
UNSB_CURVE_CSV = ROOT / "round1_solver_unsb_cycle_remote_full_eval_pull" / "clip_lpips_curve.csv"
POINTS_CSV = ROOT / "round1_newdata_variant_board.csv"

OUT_HTML = OUT_DIR / "fig_round1_newdata_variant_board.html"
OUT_PNG = OUT_DIR / "fig_round1_newdata_variant_board.png"
OUT_PDF = OUT_DIR / "fig_round1_newdata_variant_board.pdf"

WIDTH = 1600
HEIGHT = 1160


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _trajectory(rows: list[dict[str, str]], *, lpips_key: str) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for row in rows:
        clip = float(row["transfer_clip_style"])
        lpips = float(row[lpips_key])
        out.append(
            {
                "epoch": str(row.get("epoch", "")),
                "x": 1.0 - lpips,
                "y": clip - IDT_TRANSFER,
            }
        )
    return out


def _point_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        clip = float(row["transfer_clip_style"])
        lpips = float(row["transfer_lpips"])
        display_label = row.get("display_label", "").strip() or row["label"]
        out.append(
            {
                "method": row["method"],
                "group": row["group"],
                "checkpoint_kind": row["checkpoint_kind"],
                "checkpoint": row["checkpoint"],
                "train_wall_hours": row["train_wall_hours"],
                "time_note": row["time_note"],
                "transfer_clip_style": clip,
                "transfer_lpips": lpips,
                "all_pairs_clip_style": float(row["all_pairs_clip_style"]) if row["all_pairs_clip_style"] else None,
                "all_pairs_lpips": float(row["all_pairs_lpips"]) if row["all_pairs_lpips"] else None,
                "label": row["label"],
                "displayLabel": display_label,
                "marker": row["marker"],
                "color": row["color"],
                "hint_dx": float(row["dx"]),
                "hint_dy": float(row["dy"]),
                "size": float(row["size"]),
                "x": 1.0 - lpips,
                "y": clip - IDT_TRANSFER,
            }
        )
    return out


def _browser_exe() -> str:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("msedge.exe"),
        shutil.which("chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No Chromium-based browser found for headless export.")


def _build_html(*, trajectories: list[dict[str, object]], points: list[dict[str, object]]) -> str:
    payload = {
        "width": WIDTH,
        "height": HEIGHT,
        "xDomain": [0.35, 0.79],
        "yDomain": [-0.11, 0.07],
        "idtFloor": 0.0,
        "trajectories": trajectories,
        "points": points,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Round1 New-data Variant Board</title>
  <style>
    :root {{
      --paper: #fcfbf8;
      --grid: rgba(124, 124, 124, 0.18);
      --axis: #111111;
      --floor: #8e63c0;
      --neg-band: rgba(242, 232, 247, 0.30);
      --font-serif: "Times New Roman", "Georgia", serif;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--paper);
      font-family: var(--font-serif);
    }}
    #chart {{
      width: {WIDTH}px;
      height: {HEIGHT}px;
      display: block;
      margin: 0 auto;
      background: var(--paper);
    }}
    .axis text {{
      fill: #111;
      font-size: 28px;
    }}
    .tick text {{
      fill: #111;
      font-size: 24px;
    }}
    .title {{
      fill: #111;
      font-size: 36px;
      font-weight: 600;
    }}
    .axis-label {{
      fill: #111;
      font-size: 34px;
    }}
    .legend-label {{
      fill: #111;
      font-size: 26px;
    }}
    .label-text {{
      font-size: 20px;
      font-weight: 600;
      paint-order: stroke;
      stroke: rgba(255,255,255,0.95);
      stroke-width: 6px;
      stroke-linejoin: round;
    }}
    .label-pill {{
      fill: rgba(255,255,255,0.84);
      stroke: rgba(0,0,0,0.06);
      stroke-width: 1.2px;
    }}
  </style>
</head>
<body>
<svg id="chart" viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" aria-label="New-data variant board"></svg>
<script>
const DATA = {data_json};

function createSvgEl(tag, attrs = {{}}, parent = null) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) {{
    if (value !== null && value !== undefined) el.setAttribute(key, String(value));
  }}
  if (parent) parent.appendChild(el);
  return el;
}}

function scale(domainMin, domainMax, rangeMin, rangeMax, value) {{
  const t = (value - domainMin) / (domainMax - domainMin);
  return rangeMin + t * (rangeMax - rangeMin);
}}

function markerPath(marker, x, y, size) {{
  const r = Math.sqrt(size) * 0.9;
  if (marker === "o") return {{ type: "circle", attrs: {{ cx: x, cy: y, r }} }};
  if (marker === "s") return {{ type: "rect", attrs: {{ x: x - r, y: y - r, width: 2*r, height: 2*r, rx: 2 }} }};
  if (marker === "^") return {{ type: "path", attrs: {{ d: `M ${{x}} ${{y-r}} L ${{x+r}} ${{y+r}} L ${{x-r}} ${{y+r}} Z` }} }};
  if (marker === "v") return {{ type: "path", attrs: {{ d: `M ${{x-r}} ${{y-r}} L ${{x+r}} ${{y-r}} L ${{x}} ${{y+r}} Z` }} }};
  if (marker === ">") return {{ type: "path", attrs: {{ d: `M ${{x-r}} ${{y-r}} L ${{x+r}} ${{y}} L ${{x-r}} ${{y+r}} Z` }} }};
  if (marker === "D") return {{ type: "path", attrs: {{ d: `M ${{x}} ${{y-r}} L ${{x+r}} ${{y}} L ${{x}} ${{y+r}} L ${{x-r}} ${{y}} Z` }} }};
  if (marker === "P") {{
    return {{ type: "g", attrs: {{}}, children: [
      {{ type: "rect", attrs: {{ x: x - r*0.25, y: y-r, width: r*0.5, height: 2*r, rx: 1 }} }},
      {{ type: "rect", attrs: {{ x: x-r, y: y-r*0.25, width: 2*r, height: r*0.5, rx: 1 }} }},
    ] }};
  }}
  if (marker === "X") {{
    return {{ type: "g", attrs: {{}}, children: [
      {{ type: "path", attrs: {{ d: `M ${{x-r}} ${{y-r}} L ${{x+r}} ${{y+r}}`, "stroke-width": 4, fill: "none" }} }},
      {{ type: "path", attrs: {{ d: `M ${{x+r}} ${{y-r}} L ${{x-r}} ${{y+r}}`, "stroke-width": 4, fill: "none" }} }},
    ] }};
  }}
  if (marker === "h") {{
    const pts = [];
    for (let i = 0; i < 6; i++) {{
      const a = Math.PI / 6 + i * Math.PI / 3;
      pts.push(`${{x + r*Math.cos(a)}},${{y + r*Math.sin(a)}}`);
    }}
    return {{ type: "polygon", attrs: {{ points: pts.join(" ") }} }};
  }}
  if (marker === "*") {{
    const pts = [];
    for (let i = 0; i < 10; i++) {{
      const rr = i % 2 === 0 ? r : r * 0.42;
      const a = -Math.PI / 2 + i * Math.PI / 5;
      pts.push(`${{x + rr*Math.cos(a)}},${{y + rr*Math.sin(a)}}`);
    }}
    return {{ type: "polygon", attrs: {{ points: pts.join(" ") }} }};
  }}
  return {{ type: "circle", attrs: {{ cx: x, cy: y, r }} }};
}}

function appendMarker(parent, markerDef, fill, stroke) {{
  const node = createSvgEl(markerDef.type, markerDef.attrs || {{}}, parent);
  if (markerDef.type !== "g") {{
    if (markerDef.type === "path" && markerDef.attrs && markerDef.attrs.fill === "none") {{
      node.setAttribute("stroke", fill);
      node.setAttribute("stroke-linecap", "round");
    }} else {{
      node.setAttribute("fill", fill);
      node.setAttribute("stroke", stroke);
      node.setAttribute("stroke-width", "2.2");
    }}
  }}
  for (const child of markerDef.children || []) {{
    const c = createSvgEl(child.type, child.attrs || {{}}, node);
    c.setAttribute("fill", fill);
    c.setAttribute("stroke", stroke);
    c.setAttribute("stroke-width", child.attrs && child.attrs["stroke-width"] ? child.attrs["stroke-width"] : "2.2");
    if (child.attrs && child.attrs.fill === "none") {{
      c.setAttribute("fill", "none");
      c.setAttribute("stroke", fill);
    }}
  }}
}}

function wrapLabel(text) {{
  if (text.includes("\\n")) return text.split("\\n").filter(Boolean);
  if (text.length <= 17 || !text.includes(" ")) return [text];
  const parts = text.split(" ");
  let best = null;
  for (let i = 1; i < parts.length; i++) {{
    const left = parts.slice(0, i).join(" ");
    const right = parts.slice(i).join(" ");
    const score = Math.abs(left.length - right.length);
    if (!best || score < best.score) best = {{ left, right, score }};
  }}
  if (!best) return [text];
  return [best.left, best.right];
}}

function renderLabelText(textEl, label) {{
  const lines = wrapLabel(label);
  for (const [idx, line] of lines.entries()) {{
    const tspan = createSvgEl("tspan", {{
      x: 0,
      dy: idx === 0 ? 0 : 23,
    }}, textEl);
    tspan.textContent = line;
  }}
}}

function unitHint(dx, dy) {{
  const norm = Math.hypot(dx, dy);
  if (norm < 1e-6) return {{ x: 1, y: -0.35 }};
  return {{ x: dx / norm, y: dy / norm }};
}}

function primaryDirections(hx, hy) {{
  const sx = hx >= 0 ? 1 : -1;
  const sy = hy >= 0 ? 1 : -1;
  if (Math.abs(hx) >= Math.abs(hy)) {{
    return [
      [sx, sy * 0.38],
      [sx, -sy * 0.38],
      [sx, 0],
      [sx * 0.58, sy * 0.9],
      [sx * 0.58, -sy * 0.9],
      [0, sy],
      [0, -sy],
      [-sx, sy * 0.32],
      [-sx, -sy * 0.32],
    ];
  }}
  return [
    [sx * 0.35, sy],
    [-sx * 0.35, sy],
    [0, sy],
    [sx * 0.92, sy * 0.55],
    [-sx * 0.92, sy * 0.55],
    [sx, 0],
    [-sx, 0],
    [sx * 0.32, -sy],
    [-sx * 0.32, -sy],
  ];
}}

function normalizeDir(vx, vy) {{
  const norm = Math.hypot(vx, vy);
  return {{ x: vx / norm, y: vy / norm }};
}}

const svg = document.getElementById("chart");
const margin = {{ top: 82, right: 42, bottom: 105, left: 122 }};
const plot = {{
  x: margin.left,
  y: margin.top,
  width: DATA.width - margin.left - margin.right,
  height: DATA.height - margin.top - margin.bottom
}};
const sx = (v) => scale(DATA.xDomain[0], DATA.xDomain[1], plot.x, plot.x + plot.width, v);
const sy = (v) => scale(DATA.yDomain[0], DATA.yDomain[1], plot.y + plot.height, plot.y, v);

createSvgEl("rect", {{ x: 0, y: 0, width: DATA.width, height: DATA.height, fill: "#fcfbf8" }}, svg);
createSvgEl("rect", {{ x: plot.x, y: sy(0), width: plot.width, height: plot.y + plot.height - sy(0), fill: "rgba(242,232,247,0.30)" }}, svg);

for (let i = 0; i <= 8; i++) {{
  const x = plot.x + (plot.width / 8) * i;
  createSvgEl("line", {{ x1: x, y1: plot.y, x2: x, y2: plot.y + plot.height, stroke: "rgba(124,124,124,0.18)", "stroke-width": 2 }}, svg);
}}
for (let i = 0; i <= 9; i++) {{
  const y = plot.y + (plot.height / 9) * i;
  createSvgEl("line", {{ x1: plot.x, y1: y, x2: plot.x + plot.width, y2: y, stroke: "rgba(124,124,124,0.18)", "stroke-width": 2 }}, svg);
}}

createSvgEl("line", {{ x1: sx(DATA.xDomain[0]), y1: sy(DATA.idtFloor), x2: sx(DATA.xDomain[1]), y2: sy(DATA.idtFloor), stroke: "#8e63c0", "stroke-width": 5, "stroke-dasharray": "18 12" }}, svg);
createSvgEl("text", {{ x: sx(DATA.xDomain[1]) - 4, y: sy(DATA.idtFloor) - 14, "text-anchor": "end", fill: "#8e63c0", "font-size": 22, "font-weight": 700 }}, svg).textContent = "IDT floor";

createSvgEl("text", {{ x: DATA.width / 2, y: 42, class: "title", "text-anchor": "middle" }}, svg).textContent = "New-data variant board on the fixed test split";
createSvgEl("text", {{ x: DATA.width / 2, y: DATA.height - 18, class: "axis-label", "text-anchor": "middle" }}, svg).textContent = "1 − LPIPS_tr";
const yLab = createSvgEl("text", {{ x: 28, y: DATA.height / 2, class: "axis-label", "text-anchor": "middle", transform: `rotate(-90 28 ${{DATA.height/2}})` }}, svg);
yLab.textContent = "Δ_IDT,tr (CLIP-S) ↑";

for (let i = 0; i < 9; i++) {{
  const xv = DATA.xDomain[0] + (DATA.xDomain[1] - DATA.xDomain[0]) * (i / 8);
  const x = sx(xv);
  createSvgEl("line", {{ x1: x, y1: plot.y + plot.height, x2: x, y2: plot.y + plot.height + 8, stroke: "#111", "stroke-width": 2.2 }}, svg);
  const t = createSvgEl("text", {{ x, y: plot.y + plot.height + 38, class: "tick", "text-anchor": "middle" }}, svg);
  t.textContent = xv.toFixed(2);
}}
for (let i = 0; i < 10; i++) {{
  const yv = DATA.yDomain[0] + (DATA.yDomain[1] - DATA.yDomain[0]) * (i / 9);
  const y = sy(yv);
  createSvgEl("line", {{ x1: plot.x - 8, y1: y, x2: plot.x, y2: y, stroke: "#111", "stroke-width": 2.2 }}, svg);
  const t = createSvgEl("text", {{ x: plot.x - 14, y: y + 8, class: "tick", "text-anchor": "end" }}, svg);
  t.textContent = yv.toFixed(2);
}}

createSvgEl("line", {{ x1: plot.x, y1: plot.y, x2: plot.x, y2: plot.y + plot.height, stroke: "#111", "stroke-width": 4 }}, svg);
createSvgEl("line", {{ x1: plot.x, y1: plot.y + plot.height, x2: plot.x + plot.width, y2: plot.y + plot.height, stroke: "#111", "stroke-width": 4 }}, svg);

for (const traj of DATA.trajectories) {{
  const pts = traj.points.map((p) => `${{sx(p.x)}},${{sy(p.y)}}`).join(" ");
  createSvgEl("polyline", {{
    points: pts,
    fill: "none",
    stroke: traj.color,
    "stroke-width": traj.width,
    "stroke-opacity": traj.opacity
  }}, svg);
  for (const p of traj.points) {{
    createSvgEl("circle", {{ cx: sx(p.x), cy: sy(p.y), r: traj.pointRadius, fill: traj.pointColor, "fill-opacity": traj.pointOpacity }}, svg);
  }}
}}

const obstaclePoints = [];
for (const traj of DATA.trajectories) {{
  for (const p of traj.points) {{
    obstaclePoints.push({{ x: sx(p.x), y: sy(p.y), r: traj.pointRadius + 4 }});
  }}
}}

const labelNodes = [];
for (const point of DATA.points) {{
  const x = sx(point.x);
  const y = sy(point.y);
  appendMarker(svg, markerPath(point.marker, x, y, point.size), point.color, "#ffffff");
  const leader = createSvgEl("line", {{ x1: x, y1: y, x2: x, y2: y, stroke: point.color, "stroke-width": 2.2, opacity: 0.95 }}, svg);
  const g = createSvgEl("g", {{}}, svg);
  const rect = createSvgEl("rect", {{ class: "label-pill", rx: 12, ry: 12 }}, g);
  const text = createSvgEl("text", {{ class: "label-text", fill: point.color, x: 0, y: 0 }}, g);
  renderLabelText(text, point.displayLabel || point.label);
  const textBox = text.getBBox();
  const hint = unitHint(point.hint_dx, point.hint_dy);
  const pointRadius = Math.sqrt(point.size) * 0.9;
  labelNodes.push({{
    point,
    anchorX: x,
    anchorY: y,
    x: 0,
    y: 0,
    hint,
    pointRadius,
    textBox0: {{ x: textBox.x, y: textBox.y, w: textBox.width, h: textBox.height }},
    group: g,
    rect,
    text,
    leader,
    home: null,
    box: null,
    priority: point.size + String(point.label || "").length * 0.45,
  }});
}}

const PAD_X = 11;
const PAD_Y = 7;
const PLOT_PAD = 12;

function candidateBox(node, textX, textY) {{
  return {{
    textX,
    textY,
    x: textX + node.textBox0.x - PAD_X,
    y: textY + node.textBox0.y - PAD_Y,
    w: node.textBox0.w + PAD_X * 2,
    h: node.textBox0.h + PAD_Y * 1.7,
  }};
}}

function clampBox(box) {{
  const minX = plot.x + PLOT_PAD;
  const maxX = plot.x + plot.width - PLOT_PAD - box.w;
  const minY = plot.y + PLOT_PAD;
  const maxY = plot.y + plot.height - PLOT_PAD - box.h;
  const shiftX = Math.min(maxX, Math.max(minX, box.x)) - box.x;
  const shiftY = Math.min(maxY, Math.max(minY, box.y)) - box.y;
  box.x += shiftX;
  box.y += shiftY;
  box.textX += shiftX;
  box.textY += shiftY;
  return box;
}}

function intersectArea(a, b, pad = 0) {{
  const ox = Math.min(a.x + a.w + pad, b.x + b.w + pad) - Math.max(a.x - pad, b.x - pad);
  const oy = Math.min(a.y + a.h + pad, b.y + b.h + pad) - Math.max(a.y - pad, b.y - pad);
  if (ox <= 0 || oy <= 0) return 0;
  return ox * oy;
}}

function boxCost(node, box, placed) {{
  let cost = 0;
  for (const other of placed) {{
    const overlap = intersectArea(box, other.box, 10);
    if (overlap > 0) cost += overlap * 18;
    const near = intersectArea(box, other.box, 22);
    if (near > 0) cost += near * 1.1;
  }}
  for (const obstacle of obstaclePoints) {{
    if (Math.abs(obstacle.x - node.anchorX) < 0.5 && Math.abs(obstacle.y - node.anchorY) < 0.5) continue;
    const nearestX = Math.max(box.x, Math.min(obstacle.x, box.x + box.w));
    const nearestY = Math.max(box.y, Math.min(obstacle.y, box.y + box.h));
    const dist = Math.hypot(obstacle.x - nearestX, obstacle.y - nearestY);
    if (dist < obstacle.r + 8) {{
      cost += (obstacle.r + 8 - dist) * 55;
    }}
  }}
  const cx = box.x + box.w * 0.5;
  const cy = box.y + box.h * 0.5;
  const leader = Math.hypot(cx - node.anchorX, cy - node.anchorY);
  cost += leader * 0.52;
  if (node.home) {{
    const homeCx = node.home.x + node.home.w * 0.5;
    const homeCy = node.home.y + node.home.h * 0.5;
    cost += Math.hypot(cx - homeCx, cy - homeCy) * 0.22;
  }}
  return cost;
}}

function applyBox(node, box) {{
  node.x = box.textX;
  node.y = box.textY;
  node.box = box;
  node.text.setAttribute("transform", `translate(${{node.x}} ${{node.y}})`);
  node.rect.setAttribute("x", box.x);
  node.rect.setAttribute("y", box.y);
  node.rect.setAttribute("width", box.w);
  node.rect.setAttribute("height", box.h);
  const cx = Math.max(box.x, Math.min(node.anchorX, box.x + box.w));
  const cy = Math.max(box.y, Math.min(node.anchorY, box.y + box.h));
  node.leader.setAttribute("x1", node.anchorX);
  node.leader.setAttribute("y1", node.anchorY);
  node.leader.setAttribute("x2", cx);
  node.leader.setAttribute("y2", cy);
}}

function buildCandidates(node) {{
  const dirs = primaryDirections(node.hint.x, node.hint.y);
  const d0 = node.pointRadius + 12;
  const distances = [d0, d0 + 12, d0 + 24, d0 + 38, d0 + 58];
  const out = [];
  for (const [vx0, vy0] of dirs) {{
    const dir = normalizeDir(vx0, vy0);
    for (const dist of distances) {{
      let textX;
      if (dir.x > 0.35) textX = node.anchorX + dist;
      else if (dir.x < -0.35) textX = node.anchorX - dist - node.textBox0.w;
      else textX = node.anchorX - node.textBox0.w * 0.5;
      let textY;
      if (dir.y > 0.45) textY = node.anchorY + dist + node.textBox0.h * 0.72;
      else if (dir.y < -0.45) textY = node.anchorY - dist - node.textBox0.h * 0.28;
      else textY = node.anchorY + dir.y * dist + node.textBox0.h * 0.34;
      if (Math.abs(dir.x) > 0.22 && Math.abs(dir.y) > 0.22) textY += dir.y * 4;
      out.push(clampBox(candidateBox(node, textX, textY)));
    }}
  }}
  out.push(clampBox(candidateBox(node, node.anchorX - node.textBox0.w * 0.5, node.anchorY - d0 - node.textBox0.h * 0.35)));
  out.push(clampBox(candidateBox(node, node.anchorX - node.textBox0.w * 0.5, node.anchorY + d0 + node.textBox0.h * 0.9)));
  return out;
}}

function greedyPlacement() {{
  const placed = [];
  const ordered = [...labelNodes].sort((a, b) => b.priority - a.priority);
  for (const node of ordered) {{
    const candidates = buildCandidates(node);
    const best = candidates.reduce((bestSoFar, cand) => {{
      const score = boxCost(node, cand, placed);
      if (!bestSoFar || score < bestSoFar.score) return {{ box: cand, score }};
      return bestSoFar;
    }}, null);
    node.home = best.box;
    applyBox(node, best.box);
    placed.push(node);
  }}
}}

function layoutLabels() {{
  greedyPlacement();
  for (let iter = 0; iter < 180; iter++) {{
    let moved = 0;
    for (let i = 0; i < labelNodes.length; i++) {{
      const node = labelNodes[i];
      let fx = 0;
      let fy = 0;
      for (let j = 0; j < labelNodes.length; j++) {{
        if (i === j) continue;
        const other = labelNodes[j];
        const overlap = intersectArea(node.box, other.box, 16);
        if (overlap <= 0) continue;
        const ax = node.box.x + node.box.w * 0.5;
        const ay = node.box.y + node.box.h * 0.5;
        const bx = other.box.x + other.box.w * 0.5;
        const by = other.box.y + other.box.h * 0.5;
        let dx = ax - bx;
        let dy = ay - by;
        if (Math.abs(dx) < 1) dx = i < j ? -1 : 1;
        if (Math.abs(dy) < 1) dy = i < j ? -1 : 1;
        const norm = Math.hypot(dx, dy);
        fx += (dx / norm) * Math.min(12, overlap / 160);
        fy += (dy / norm) * Math.min(12, overlap / 160);
      }}
      if (node.home) {{
        const homeCx = node.home.x + node.home.w * 0.5;
        const homeCy = node.home.y + node.home.h * 0.5;
        const cx = node.box.x + node.box.w * 0.5;
        const cy = node.box.y + node.box.h * 0.5;
        fx += (homeCx - cx) * 0.06;
        fy += (homeCy - cy) * 0.06;
      }}
      if (Math.abs(fx) < 0.05 && Math.abs(fy) < 0.05) continue;
      const movedBox = clampBox(candidateBox(node, node.x + Math.max(-8, Math.min(8, fx)), node.y + Math.max(-8, Math.min(8, fy))));
      applyBox(node, movedBox);
      moved += Math.abs(fx) + Math.abs(fy);
    }}
    if (moved < 0.4) break;
  }}
}}

layoutLabels();

const legend = createSvgEl("g", {{}}, svg);
const legendBox = createSvgEl("rect", {{
  x: plot.x + 12,
  y: plot.y + plot.height - 222,
  width: 430,
  height: 192,
  rx: 18, ry: 18,
  fill: "rgba(255,255,255,0.86)",
  stroke: "rgba(0,0,0,0.12)",
  "stroke-width": 2
}}, legend);
for (const [i, traj] of DATA.trajectories.entries()) {{
  const ly = plot.y + plot.height - 186 + i * 33;
  createSvgEl("line", {{ x1: plot.x + 26, y1: ly, x2: plot.x + 92, y2: ly, stroke: traj.color, "stroke-width": traj.width }}, legend);
  const t = createSvgEl("text", {{ x: plot.x + 114, y: ly + 9, class: "legend-label" }}, legend);
  t.textContent = traj.label;
}}
</script>
</body>
</html>
"""


def main() -> int:
    points = _point_rows(_read_csv(POINTS_CSV))
    trajectories = [
        {
            "label": "W5 SaMAM trajectory",
            "color": "#F07F5A",
            "width": 5,
            "opacity": 0.9,
            "pointColor": "#F0A085",
            "pointOpacity": 0.6,
            "pointRadius": 6.2,
            "points": _trajectory(_read_csv(WIKIARTS5_CURVE_CSV), lpips_key="transfer_lpips"),
        },
        {
            "label": "SaMST trajectory",
            "color": "#B76E12",
            "width": 4,
            "opacity": 0.82,
            "pointColor": "#D59A3A",
            "pointOpacity": 0.58,
            "pointRadius": 5.2,
            "points": _trajectory(_read_csv(SAMST_CURVE_CSV), lpips_key="transfer_content_lpips"),
        },
        {
            "label": "AttnSA trajectory",
            "color": "#1D4ED8",
            "width": 4,
            "opacity": 0.78,
            "pointColor": "#6E93FF",
            "pointOpacity": 0.40,
            "pointRadius": 4.4,
            "points": _trajectory(_read_csv(ATTNSA_CURVE_CSV), lpips_key="transfer_content_lpips"),
        },
        {
            "label": "GatedSPADE trajectory",
            "color": "#16A085",
            "width": 4,
            "opacity": 0.8,
            "pointColor": "#52C7B0",
            "pointOpacity": 0.4,
            "pointRadius": 4.4,
            "points": _trajectory(_read_csv(GATED_CURVE_CSV), lpips_key="transfer_content_lpips"),
        },
        {
            "label": "UNSB trajectory",
            "color": "#7C3AED",
            "width": 4,
            "opacity": 0.82,
            "pointColor": "#B79CFF",
            "pointOpacity": 0.42,
            "pointRadius": 4.4,
            "points": _trajectory(_read_csv(UNSB_CURVE_CSV), lpips_key="transfer_content_lpips"),
        },
    ]

    html = _build_html(trajectories=trajectories, points=points)
    OUT_HTML.write_text(html, encoding="utf-8")

    browser = _browser_exe()
    file_url = OUT_HTML.resolve().as_uri()
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--screenshot={OUT_PNG}",
            file_url,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--print-to-pdf={OUT_PDF}",
            file_url,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    print(OUT_HTML)
    print(OUT_PNG)
    print(POINTS_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
