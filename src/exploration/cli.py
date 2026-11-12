"""Run with: python scripts/run.py --help"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from .agent import QLearner, schedule
from .environment import GridWorld
from .evaluation import evaluate, rollout


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def source_hash():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train(config, condition, seed, output):
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    env = GridWorld(**config["environment"])
    agent = QLearner(env.n_states, **config["algorithm"], seed=seed)
    reference = len(env.shortest_path()) - 1
    started = time.perf_counter()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    metadata = {"phase": config["phase"], "provenance": "new reconstruction; no original results",
                "condition": condition, "seed": seed, "python": sys.version, "platform": platform.platform(),
                "processor": platform.processor(), "logical_cpus": os.cpu_count(), "device": "CPU",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit,
                "source_sha256": source_hash(), "shortest_route": reference,
                "checkpoint_scope": "inference only; resuming training is not supported"}
    save_json(output / "metadata.json", metadata)
    budget = config["experiment"]["steps"]
    every = config["experiment"]["eval_every"]
    state, _ = env.reset(seed=seed)
    visits = [0] * env.n_states
    visits[state] += 1
    ep_return, ep_length, ep_id = 0.0, 0, 0
    first_threshold = None
    with (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as ef, (output / "episodes.csv").open("w", newline="", encoding="utf-8") as tf, (output / "exploration.csv").open("w", newline="", encoding="utf-8") as xf:
        eval_writer = csv.DictWriter(ef, fieldnames=["step", "exploration_value", "greedy_return", "greedy_success", "greedy_length", "exploring_return", "exploring_success", "exploring_length", "coverage"])
        train_writer = csv.DictWriter(tf, fieldnames=["step", "episode", "return", "length", "terminated", "truncated"])
        explore_writer = csv.DictWriter(xf, fieldnames=["step", "value", "action_entropy_nats"])
        for writer in (eval_writer, train_writer, explore_writer):
            writer.writeheader()

        def checkpoint(step):
            nonlocal first_threshold
            value = schedule(condition, step, budget)
            greedy, metrics = evaluate(agent, config, condition, value)
            eval_writer.writerow({"step": step, "exploration_value": value, **metrics,
                                  "coverage": sum(v > 0 for v in visits) / env.n_states})
            threshold = config["experiment"]["threshold"]
            if first_threshold is None and metrics["greedy_success"] >= threshold["greedy_success"] and metrics["greedy_length"] <= reference * threshold["max_shortest_path_ratio"]:
                first_threshold = step
            # Every evaluation checkpoint is retained; no retrospective best selection.
            save_json(output / f"checkpoint_{step:07d}.json", {"step": step, "q": agent.q, "config": config,
                      "condition": condition, "seed": seed, "greedy_trajectory": greedy, "visits": visits})

        checkpoint(0)
        for step in range(budget):
            value = schedule(condition, step, budget)
            probs = agent.probabilities(state, condition["kind"], value)
            explore_writer.writerow({"step": step, "value": value, "action_entropy_nats": -sum(p * math.log(p) for p in probs if p > 0)})
            action = agent.act(state, condition["kind"], value)
            nxt, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, nxt, terminated)
            visits[nxt] += 1
            ep_return, ep_length = ep_return + reward, ep_length + 1
            state = nxt
            if terminated or truncated:
                train_writer.writerow({"step": step + 1, "episode": ep_id, "return": ep_return,
                    "length": ep_length, "terminated": int(terminated), "truncated": int(truncated)})
                ep_id += 1
                ep_return, ep_length = 0.0, 0
                state, _ = env.reset()
                visits[state] += 1
            if (step + 1) % every == 0 or step + 1 == budget:
                checkpoint(step + 1)
    metadata.update({"wall_seconds": time.perf_counter() - started,
                     "first_threshold_step": first_threshold, "completed_steps": budget,
                     "partial_last_episode_length": ep_length})
    save_json(output / "metadata.json", metadata)
    print(f"{condition['name']} seed={seed}: {budget} steps, threshold={first_threshold}, {metadata['wall_seconds']:.2f}s", flush=True)


def check(config, output):
    output.mkdir(parents=True, exist_ok=False)
    env = GridWorld(**config["environment"])
    random_policy = QLearner(env.n_states)
    trajectories = [rollout(random_policy, config["environment"], {"kind": "epsilon"}, 1, seed) for seed in range(5)]
    # Reference policy follows BFS, independently of learned Q values.
    path = env.shortest_path()
    state, _ = env.reset()
    reference_actions = []
    for nxt in path[1:]:
        action = next(a for a in range(4) if env.transition(state, a) == nxt)
        state, reward, terminated, truncated, _ = env.step(action)
        reference_actions.append(action)
    data = {"n_states": env.n_states, "action_space": "Discrete(4)", "observation_space": f"Discrete({env.n_states})",
            "shortest_steps": len(path) - 1, "reference": {"states": path, "actions": reference_actions, "success": int(terminated), "return": reward},
            "random_trajectories": trajectories, "environment": config["environment"]}
    save_json(output / "environment_check.json", data)
    print(json.dumps({k: v for k, v in data.items() if k not in ("reference", "random_trajectories", "environment")}, indent=2))
    from .plots import environment_demo
    environment_demo(data, output / "trajectories.html")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "train"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/tabular_pilot.json")
        p.add_argument("--output", required=True)
        if name == "train":
            p.add_argument("--seed", type=int)
            p.add_argument("--condition")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        checkpoint = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
        cfg = checkpoint["config"]
        agent = QLearner(len(checkpoint["q"]), **cfg["algorithm"])
        agent.q = checkpoint["q"]
        value = schedule(checkpoint["condition"], checkpoint["step"], cfg["experiment"]["steps"])
        _, metrics = evaluate(agent, cfg, checkpoint["condition"], value)
        print(json.dumps(metrics, indent=2))
    elif args.command == "plot":
        from .plots import summarize
        summarize(Path(args.input), Path(args.output))
    else:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.command == "check":
            check(cfg, Path(args.output))
        else:
            conditions = [c for c in cfg["conditions"] if args.condition is None or c["name"] == args.condition]
            if not conditions:
                parser.error("Unknown condition")
            for condition in conditions:
                for seed in cfg["experiment"]["seeds"] if args.seed is None else [args.seed]:
                    train(cfg, condition, seed, Path(args.output) / condition["name"] / f"seed_{seed}")
