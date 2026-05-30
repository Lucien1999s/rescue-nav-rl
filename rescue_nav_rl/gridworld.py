from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

ACTION_NAMES = {
    LEFT: "left",
    UP: "up",
    RIGHT: "right",
    DOWN: "down",
}

ACTION_SYMBOLS = {
    LEFT: "<",
    UP: "^",
    RIGHT: ">",
    DOWN: "v",
}

ACTION_DELTAS = {
    LEFT: (-1, 0),
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
}

SLIP_ACTIONS = {
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
}


@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool
    info: dict


class GridworldEnv:
    """
    4x4 stochastic hazardous Gridworld for MP2.

    Coordinate convention:
    - (0, 0) is top-left.
    - x increases to the right.
    - y increases downward.

    Action indices:
    - 0: left
    - 1: up
    - 2: right
    - 3: down
    """

    def __init__(self, seed: int | None = None, max_steps: int = 100):
        self.width = 4
        self.height = 4
        self.num_states = self.width * self.height
        self.num_actions = 4

        self.start_xy = (0, 0)
        self.goal_xy = (3, 3)

        # From the MP2 figure.
        self.fire_cells = {(1, 0), (2, 0)}
        self.water_cells = {(1, 2), (2, 2)}

        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.state = self.xy_to_state(*self.start_xy)
        self.steps = 0
        self.done = False

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = self.xy_to_state(*self.start_xy)
        self.steps = 0
        self.done = False
        return self.state

    def xy_to_state(self, x: int, y: int) -> int:
        return y * self.width + x

    def state_to_xy(self, state: int) -> Tuple[int, int]:
        x = state % self.width
        y = state // self.width
        return x, y

    def state_tuple(self, state: int) -> Tuple[int, int, int, int]:
        x, y = self.state_to_xy(state)
        water = int((x, y) in self.water_cells)
        fire = int((x, y) in self.fire_cells)
        return x, y, water, fire

    def is_goal(self, state: int) -> bool:
        return self.state_to_xy(state) == self.goal_xy

    def is_water(self, state: int) -> bool:
        return self.state_to_xy(state) in self.water_cells

    def is_fire(self, state: int) -> bool:
        return self.state_to_xy(state) in self.fire_cells

    def reward(self, next_state: int) -> float:
        if self.is_goal(next_state):
            return 100.0
        if self.is_water(next_state):
            return -5.0
        if self.is_fire(next_state):
            return -10.0
        return -1.0

    def _move(self, state: int, action: int) -> int:
        x, y = self.state_to_xy(state)
        dx, dy = ACTION_DELTAS[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.width and 0 <= ny < self.height:
            return self.xy_to_state(nx, ny)

        return state

    def transition_probs(self, state: int, action: int) -> List[Tuple[float, int]]:
        """
        Return aggregated transition probabilities as:
        [(probability, next_state), ...]
        """
        outcomes = [
            (0.8, self._move(state, action)),
            (0.1, self._move(state, SLIP_ACTIONS[action][0])),
            (0.1, self._move(state, SLIP_ACTIONS[action][1])),
        ]

        probs: Dict[int, float] = {}
        for prob, next_state in outcomes:
            probs[next_state] = probs.get(next_state, 0.0) + prob

        return [(prob, next_state) for next_state, prob in probs.items()]

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(
                next_state=self.state,
                reward=0.0,
                done=True,
                info={"already_done": True},
            )

        transitions = self.transition_probs(self.state, action)
        probs = np.array([p for p, _ in transitions], dtype=float)
        next_states = np.array([s for _, s in transitions], dtype=int)

        next_state = int(self.rng.choice(next_states, p=probs))
        reward = self.reward(next_state)

        self.state = next_state
        self.steps += 1
        self.done = self.is_goal(next_state) or self.steps >= self.max_steps

        return StepResult(
            next_state=next_state,
            reward=reward,
            done=self.done,
            info={
                "steps": self.steps,
                "state_tuple": self.state_tuple(next_state),
                "reached_goal": self.is_goal(next_state),
                "timeout": self.steps >= self.max_steps and not self.is_goal(next_state),
            },
        )

    def state_features(self, state: int) -> np.ndarray:
        """
        Compact features for later Actor-Critic.

        Features:
        - normalized x
        - normalized y
        - water indicator
        - fire indicator
        - normalized Manhattan distance to goal
        - bias
        """
        x, y, water, fire = self.state_tuple(state)
        gx, gy = self.goal_xy
        max_dist = (self.width - 1) + (self.height - 1)
        dist = abs(gx - x) + abs(gy - y)

        return np.array(
            [
                x / (self.width - 1),
                y / (self.height - 1),
                water,
                fire,
                dist / max_dist,
                1.0,
            ],
            dtype=float,
        )

    def format_grid(self, policy: np.ndarray | None = None) -> str:
        rows = []

        for y in range(self.height):
            row = []
            for x in range(self.width):
                state = self.xy_to_state(x, y)

                if (x, y) == self.start_xy:
                    cell = "S"
                elif (x, y) == self.goal_xy:
                    cell = "G"
                elif (x, y) in self.fire_cells:
                    cell = "F"
                elif (x, y) in self.water_cells:
                    cell = "W"
                elif policy is not None:
                    cell = ACTION_SYMBOLS[int(policy[state])]
                else:
                    cell = "."

                row.append(cell)
            rows.append(" ".join(row))

        return "\n".join(rows)
