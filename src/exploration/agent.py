"""Shared Q-learning update; only action probabilities differ by condition."""
import math
import random


def schedule(condition, step, budget):
    fraction = condition["decay_fraction"]
    if not 0 < fraction <= 1 or budget <= 0:
        raise ValueError("Invalid schedule duration")
    progress = min(1.0, max(0.0, step / (fraction * budget)))
    value = condition["start"] + progress * (condition["end"] - condition["start"])
    if condition["kind"] == "epsilon" and not 0 <= value <= 1:
        raise ValueError("Epsilon must be in [0, 1]")
    if condition["kind"] == "softmax" and value <= 0:
        raise ValueError("Temperature must be positive")
    return value


class QLearner:
    def __init__(self, n_states, alpha=0.15, gamma=0.97, initial_q=0.0, seed=0):
        if not 0 < alpha <= 1 or not 0 <= gamma < 1:
            raise ValueError("Invalid Q-learning parameters")
        self.q = [[float(initial_q)] * 4 for _ in range(n_states)]
        self.alpha, self.gamma = alpha, gamma
        self.rng = random.Random(seed)

    def probabilities(self, state, kind, value):
        row = self.q[state]
        if kind == "epsilon":
            best = [a for a, q in enumerate(row) if q == max(row)]
            return [value / 4 + ((1 - value) / len(best) if a in best else 0) for a in range(4)]
        if kind == "softmax":
            weights = [math.exp((q - max(row)) / value) for q in row]
            return [w / sum(weights) for w in weights]
        raise ValueError(f"Unknown exploration kind: {kind}")

    def act(self, state, kind, value, rng=None):
        return (rng or self.rng).choices(range(4), weights=self.probabilities(state, kind, value))[0]

    def greedy(self, state):
        # Fixed tie rule makes greedy evaluation a deterministic policy.
        return max(range(4), key=lambda a: self.q[state][a])

    def update(self, state, action, reward, next_state, terminated):
        # External time limits bootstrap; only reaching the goal is terminal.
        target = reward + (0 if terminated else self.gamma * max(self.q[next_state]))
        self.q[state][action] += self.alpha * (target - self.q[state][action])
