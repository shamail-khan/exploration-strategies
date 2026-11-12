"""Deterministic navigation, with time limits separate from true termination."""
from collections import deque


class GridWorld:
    # Up, right, down, left. Hitting a wall consumes an interaction.
    ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, layout, horizon=60):
        if not layout or not layout[0] or any(len(r) != len(layout[0]) for r in layout):
            raise ValueError("Layout must be a nonempty rectangle")
        if any(c not in ".#SG" for row in layout for c in row):
            raise ValueError("Unknown layout symbol")
        if horizon <= 0:
            raise ValueError("Horizon must be positive")
        self.layout, self.horizon = tuple(layout), horizon
        self.height, self.width = len(layout), len(layout[0])
        self.cells = [(r, c) for r, row in enumerate(layout) for c, v in enumerate(row) if v != "#"]
        self.ids = {cell: i for i, cell in enumerate(self.cells)}
        starts = [self.ids[(r, c)] for r, row in enumerate(layout) for c, v in enumerate(row) if v == "S"]
        goals = [self.ids[(r, c)] for r, row in enumerate(layout) for c, v in enumerate(row) if v == "G"]
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("Exactly one start and goal required")
        self.start, self.goal = starts[0], goals[0]
        self.n_states, self.n_actions = len(self.cells), 4
        self.reset()
        if self.shortest_path() is None:
            raise ValueError("Goal is unreachable")

    def transition(self, state, action):
        if not 0 <= state < self.n_states or not 0 <= action < 4:
            raise ValueError("Invalid state or action")
        r, c = self.cells[state]
        dr, dc = self.ACTIONS[action]
        return self.ids.get((r + dr, c + dc), state)

    def reset(self, seed=None):
        # No environment randomness; seed accepted for an explicit reset contract.
        self.state, self.elapsed, self.done = self.start, 0, False
        return self.state, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset after termination or truncation")
        self.state = self.transition(self.state, action)
        self.elapsed += 1
        terminated = self.state == self.goal
        truncated = self.elapsed >= self.horizon and not terminated
        self.done = terminated or truncated
        return self.state, float(terminated), terminated, truncated, {"success": terminated}

    def shortest_path(self):
        queue = deque([[self.start]])
        seen = {self.start}
        while queue:
            path = queue.popleft()
            if path[-1] == self.goal:
                return path
            for action in range(4):
                nxt = self.transition(path[-1], action)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(path + [nxt])
        return None
