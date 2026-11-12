"""No updates; independent environments and RNGs for all evaluation rollouts."""
import random
from .environment import GridWorld


def rollout(agent, environment, condition=None, value=None, seed=0):
    env = GridWorld(**environment)
    state, _ = env.reset(seed=seed)
    rng = random.Random(seed)
    states, actions, total = [state], [], 0.0
    while True:
        action = agent.greedy(state) if condition is None else agent.act(state, condition["kind"], value, rng)
        state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        actions.append(action)
        total += reward
        if terminated or truncated:
            return {"return": total, "success": int(terminated), "length": len(actions),
                    "states": states, "actions": actions, "failure": None if terminated else "time_limit"}


def evaluate(agent, config, condition, value):
    greedy = rollout(agent, config["environment"])
    sampled = [rollout(agent, config["environment"], condition, value, seed)
               for seed in config["experiment"]["stochastic_eval_seeds"]]
    return greedy, {
        "greedy_return": greedy["return"], "greedy_success": greedy["success"],
        "greedy_length": greedy["length"],
        "exploring_return": sum(r["return"] for r in sampled) / len(sampled),
        "exploring_success": sum(r["success"] for r in sampled) / len(sampled),
        "exploring_length": sum(r["length"] for r in sampled) / len(sampled),
    }
