"""Portable SVG plots and an offline trajectory demonstration (stdlib only)."""
import csv
import html
import json
from pathlib import Path
import statistics
from .environment import GridWorld


COLORS = ["#2563eb", "#d97706", "#0d9488"]


def grid_svg(environment, states=None, visits=None):
    env = GridWorld(**environment)
    size, pad = 32, 12
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {env.width*size+24} {env.height*size+24}" width="280" role="img" aria-label="Grid trajectory or visitation map">']
    for r, row in enumerate(env.layout):
        for c, char in enumerate(row):
            fill = "#334155" if char == "#" else "#f1f5f9"
            s = env.ids.get((r, c))
            if visits is not None and s is not None:
                intensity = visits[s] / max(1, max(visits))
                fill = f"rgb({int(240-170*intensity)},{int(248-100*intensity)},250)"
            parts.append(f'<rect x="{pad+c*size}" y="{pad+r*size}" width="{size}" height="{size}" fill="{fill}" stroke="white"/>')
            label = char if char in "SG" else str(visits[s]) if visits is not None and s is not None else ""
            if label:
                parts.append(f'<text x="{pad+c*size+16}" y="{pad+r*size+21}" text-anchor="middle" font-size="10">{label}</text>')
    if states:
        points = " ".join(f"{pad+env.cells[s][1]*size+16},{pad+env.cells[s][0]*size+16}" for s in states)
        parts.append(f'<polyline points="{points}" fill="none" stroke="#e11d48" stroke-width="3" opacity="0.8"/>')
        r, c = env.cells[states[-1]]
        parts.append(f'<circle cx="{pad+c*size+16}" cy="{pad+r*size+16}" r="6" fill="#e11d48"/>')
    return "".join(parts) + "</svg>"


def environment_demo(data, output):
    trajectories = [{"name": "Shortest-path reference", "states": data["reference"]["states"]}] + [
        {"name": f"Random policy, seed {i}; success={t['success']}", "states": t["states"]}
        for i, t in enumerate(data["random_trajectories"])]
    env = GridWorld(**data["environment"])
    payload = json.dumps({"trajectories": trajectories, "cells": env.cells, "layout": env.layout})
    output.write_text('''<!doctype html><meta charset="utf-8"><title>Exploration: environment validation</title>
<style>body{font:17px system-ui;max-width:820px;margin:35px auto;color:#172033}canvas{display:block;margin:20px 0}button,select,input{font:inherit}p{line-height:1.6}</style>
<h1>Before learning: inspect the task</h1><p>New reconstruction · environment validation · no trained policy shown here.</p>
<p>Reward is +1 only at G. Walls and boundary hits consume a step. The time limit is 60 steps.</p>
<select id="policy"></select> <button id="play">Play / pause</button><input id="step" type="range" min="0" value="0"><span id="label"></span>
<canvas id="canvas" width="420" height="420"></canvas><p>The shortest route is ''' + str(data["shortest_steps"]) + ''' steps. These random trajectories are diagnostic examples, not evidence of a trained method's performance.</p>
<script>const data=''' + payload + ''';
const select=document.querySelector('#policy'), slider=document.querySelector('#step'), ctx=document.querySelector('canvas').getContext('2d');
data.trajectories.forEach((t,i)=>select.add(new Option(t.name,i)));
function draw(){let t=data.trajectories[+select.value];slider.max=t.states.length-1;let k=Math.min(+slider.value,+slider.max);slider.value=k;ctx.clearRect(0,0,420,420);
data.layout.forEach((row,r)=>[...row].forEach((v,c)=>{ctx.fillStyle=v==='#'?'#334155':'#e2e8f0';ctx.fillRect(c*60+1,r*60+1,58,58);if(v==='S'||v==='G'){ctx.fillStyle='#172033';ctx.font='20px system-ui';ctx.fillText(v,c*60+22,r*60+38);}}));
ctx.beginPath();t.states.slice(0,k+1).forEach((s,i)=>{let [r,c]=data.cells[s];i?ctx.lineTo(c*60+30,r*60+30):ctx.moveTo(c*60+30,r*60+30)});ctx.strokeStyle='#e11d48';ctx.lineWidth=4;ctx.stroke();let [r,c]=data.cells[t.states[k]];ctx.beginPath();ctx.arc(c*60+30,r*60+30,9,0,7);ctx.fillStyle='#e11d48';ctx.fill();document.querySelector('#label').textContent=` Step ${k} / ${slider.max}`;}
select.onchange=()=>{slider.value=0;draw()};slider.oninput=draw;let playing=false;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(+slider.value+1)%(+slider.max+1);draw()}},220);draw();</script>''', encoding="utf-8")


def curve_svg(groups, metric, title):
    width, height, left, top, pw, ph = 920, 420, 75, 60, 800, 270
    horizon = max(int(row["step"]) for runs in groups.values() for rows, _ in runs for row in rows)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left}" y="26" font-family="sans-serif" font-size="18">{html.escape(title)}</text>']
    for v in (0, .25, .5, .75, 1):
        y = top + ph * (1-v)
        svg += [f'<path d="M{left},{y}h{pw}" stroke="#e2e8f0"/>', f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-size="12">{v}</text>']
    for step in (0, horizon//2, horizon):
        x = left + pw * step/horizon
        svg.append(f'<text x="{x}" y="{top+ph+22}" text-anchor="middle" font-size="12">{step}</text>')
    for index, (name, runs) in enumerate(sorted(groups.items())):
        color = COLORS[index % len(COLORS)]
        def point(step, val):
            return f"{left+pw*step/horizon:.2f},{top+ph*(1-val):.2f}"
        for rows, _ in runs:
            svg.append(f'<polyline points="{" ".join(point(int(r["step"]),float(r[metric])) for r in rows)}" fill="none" stroke="{color}" opacity="0.22" stroke-width="1.5"/>')
        means = []
        for j, row in enumerate(runs[0][0]):
            vals = [float(rows[j][metric]) for rows, _ in runs]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals)>1 else 0
            step = int(row["step"])
            x = left + pw*step/horizon
            y1, y2 = top+ph*(1-min(1,mean+sd)), top+ph*(1-max(0,mean-sd))
            svg.append(f'<path d="M{x:.1f},{y1:.1f}V{y2:.1f}" stroke="{color}" opacity="0.18" stroke-width="5"/>')
            means.append(point(step,mean))
        svg.append(f'<polyline points="{" ".join(means)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        svg.append(f'<text x="{left+index*265}" y="{height-25}" fill="{color}" font-size="13">{name} (n={len(runs)})</text>')
    svg.append(f'<text x="{left+pw/2}" y="{height-52}" text-anchor="middle" font-size="13">Training environment interactions</text></svg>')
    return "".join(svg)


def summarize(root, output):
    groups = {}
    for path in sorted(root.glob("*/seed_*/evaluations.csv")):
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        if "completed_steps" not in metadata:
            raise ValueError(f"Incomplete run: {path.parent}")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        groups.setdefault(metadata["condition"]["name"], []).append((rows, metadata))
    if not groups:
        raise ValueError("No completed runs found")
    # Refuse incompatible configurations or time axes rather than silently averaging.
    paths = list(root.glob("*/seed_*/config.json"))
    configs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if any(c != configs[0] for c in configs):
        raise ValueError("Mixed experimental configurations")
    axes = [[r["step"] for r in rows] for runs in groups.values() for rows, _ in runs]
    if any(a != axes[0] for a in axes):
        raise ValueError("Mismatched evaluation interaction axes")
    output.mkdir(parents=True, exist_ok=True)
    for metric, title in [("greedy_success", "Greedy success: pilot"), ("exploring_success", "Success with exploration enabled: pilot"), ("coverage", "Training state coverage: pilot")]:
        (output / f"{metric}.svg").write_text(curve_svg(groups, metric, title), encoding="utf-8")
    lines = ["# Tabular pilot summary", "", "New reconstruction. Pilot runs, not a confirmatory study.", "",
             "Thin curves: individual training seeds. Thick curves: seed means. Vertical bars: ±1 sample standard deviation across training seeds, clipped to [0,1] for display; these are not confidence intervals. Each greedy point is one deterministic rollout on one fixed layout. The 20 exploring-policy episodes estimate within-policy behavior; they are not independent training seeds.", "",
             "| Condition | Training seeds | Final greedy success (mean ± SD) | Final exploring success (mean ± SD) | First threshold, interactions (each seed) | Normalized greedy-return AUC (mean ± SD) | Total wall seconds |",
             "|---|---:|---:|---:|---|---:|---:|"]
    scalar_rows = []
    gallery = ['<!doctype html><meta charset="utf-8"><title>Pilot policy diagnostics</title><style>body{font:16px system-ui;margin:30px;max-width:1000px}section{display:inline-block;margin:15px;vertical-align:top}svg{display:block}p{max-width:850px;line-height:1.5}</style><h1>Final pilot policies and visitation</h1><p>All final checkpoints, without selecting favorable seeds. Left: deterministic greedy trajectory. Right: cumulative training state visits (includes reset states). A time-limit failure can reflect repeated blocked actions or loops; the summary only assigns the observed category time_limit.</p>']
    for name, runs in sorted(groups.items()):
        def describe(vals):
            return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals) if len(vals)>1 else 0:.3f}"
        aucs = []
        for rows, meta in runs:
            horizon = int(rows[-1]["step"])
            auc = sum((int(b["step"])-int(a["step"]))* (float(a["greedy_return"])+float(b["greedy_return"]))/2 for a,b in zip(rows, rows[1:]))/horizon
            aucs.append(auc)
            scalar_rows.append({"condition": name, "seed": meta["seed"], "first_threshold_step": meta["first_threshold_step"], "normalized_greedy_return_auc": auc, "wall_seconds": meta["wall_seconds"]})
            checkpoint = json.loads((root / name / f"seed_{meta['seed']}" / f"checkpoint_{horizon:07d}.json").read_text(encoding="utf-8"))
            trajectory = checkpoint["greedy_trajectory"]
            gallery += [f'<section><h2>{html.escape(name)}, seed {meta["seed"]}</h2><p>Success={trajectory["success"]}; steps={trajectory["length"]}; failure={trajectory["failure"]}</p>',
                        grid_svg(configs[0]["environment"], trajectory["states"]), grid_svg(configs[0]["environment"], visits=checkpoint["visits"]), '</section>']
        thresholds = ", ".join(f"seed {m['seed']}: {m['first_threshold_step'] if m['first_threshold_step'] is not None else 'not reached'}" for _,m in runs)
        lines.append(f"| {name} | {len(runs)} | {describe([float(r[-1]['greedy_success']) for r,_ in runs])} | {describe([float(r[-1]['exploring_success']) for r,_ in runs])} | {thresholds} | {describe(aucs)} | {sum(m['wall_seconds'] for _,m in runs):.2f} |")
    lines += ["", "Threshold: greedy success and route length ≤1.25× BFS shortest path at an evaluation checkpoint. First crossing is quantized by the evaluation interval and is not a sustained-convergence measure. Non-crossing runs are reported explicitly. AUC uses trapezoidal integration of evaluation return from 0 to the shared interaction horizon, divided by that horizon; it is an interpolation summary, not a measurement between evaluations.", "", "Do not select the final study conditions solely to favor a pilot winner. No claim here establishes generalization, reward-sparsity effects, entropy-regularized optimization, or a DQN/PPO comparison."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "per_seed.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    (output / "policy_diagnostics.html").write_text("".join(gallery), encoding="utf-8")
    print(f"Plots and summary written to {output}")
