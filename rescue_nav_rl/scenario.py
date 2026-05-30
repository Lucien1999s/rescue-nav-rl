from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from rescue_nav_rl.gridworld import (
    ACTION_DELTAS,
    ACTION_NAMES,
    ACTION_SYMBOLS,
    DOWN,
    LEFT,
    RIGHT,
    SLIP_ACTIONS,
    UP,
    StepResult,
)


Coord = Tuple[int, int]


@dataclass(frozen=True)
class ScenarioConfig:
    width: int = 10
    height: int = 10
    water_density: float = 0.10
    fire_density: float = 0.08
    slip_prob: float = 0.10
    max_steps: int = 200
    start_xy: Coord = (0, 0)
    goal_xy: Optional[Coord] = None
    scenario_type: str = "mixed"


class RescueScenarioEnv:
    """
    Configurable stochastic rescue navigation environment.

    This extends the fixed MP2 Gridworld idea into a larger
    search-and-rescue / evacuation routing simulator.

    State:
        flat state index, with factored representation:
        <x, y, water, fire>

    Actions:
        0 left, 1 up, 2 right, 3 down

    Transition:
        intended move with probability 1 - 2 * slip_prob
        side slips with probability slip_prob each

    Rewards:
        +100 for reaching goal
        -5 for entering water hazard
        -10 for entering fire hazard
        -1 otherwise
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        water_density: float = 0.10,
        fire_density: float = 0.08,
        slip_prob: float = 0.10,
        max_steps: int = 200,
        start_xy: Coord = (0, 0),
        goal_xy: Optional[Coord] = None,
        water_cells: Optional[Set[Coord]] = None,
        fire_cells: Optional[Set[Coord]] = None,
        scenario_type: str = "mixed",
        seed: Optional[int] = None,
    ):
        if width < 4 or height < 4:
            raise ValueError("width and height should be at least 4 for a meaningful rescue scenario.")
        if not (0.0 <= slip_prob <= 0.49):
            raise ValueError("slip_prob must be in [0.0, 0.49].")
        if not (0.0 <= water_density <= 0.6):
            raise ValueError("water_density must be in [0.0, 0.6].")
        if not (0.0 <= fire_density <= 0.6):
            raise ValueError("fire_density must be in [0.0, 0.6].")

        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4

        self.start_xy = start_xy
        self.goal_xy = goal_xy if goal_xy is not None else (width - 1, height - 1)

        self.water_density = water_density
        self.fire_density = fire_density
        self.slip_prob = slip_prob
        self.success_prob = 1.0 - 2.0 * slip_prob
        self.max_steps = max_steps
        self.scenario_type = scenario_type

        self.rng = np.random.default_rng(seed)

        if water_cells is None or fire_cells is None:
            generated_water, generated_fire = self._generate_hazards()
            self.water_cells = generated_water if water_cells is None else set(water_cells)
            self.fire_cells = generated_fire if fire_cells is None else set(fire_cells)
        else:
            self.water_cells = set(water_cells)
            self.fire_cells = set(fire_cells)

        self.water_cells.discard(self.start_xy)
        self.water_cells.discard(self.goal_xy)
        self.fire_cells.discard(self.start_xy)
        self.fire_cells.discard(self.goal_xy)
        self.fire_cells -= self.water_cells

        self.state = self.xy_to_state(*self.start_xy)
        self.steps = 0
        self.done = False

    def clone(self, seed: Optional[int] = None) -> "RescueScenarioEnv":
        return RescueScenarioEnv(
            width=self.width,
            height=self.height,
            water_density=self.water_density,
            fire_density=self.fire_density,
            slip_prob=self.slip_prob,
            max_steps=self.max_steps,
            start_xy=self.start_xy,
            goal_xy=self.goal_xy,
            water_cells=set(self.water_cells),
            fire_cells=set(self.fire_cells),
            scenario_type=self.scenario_type,
            seed=seed,
        )

    def _candidate_cells(self) -> List[Coord]:
        cells = []

        for y in range(self.height):
            for x in range(self.width):
                xy = (x, y)
                if xy not in {self.start_xy, self.goal_xy}:
                    cells.append(xy)

        return cells

    def _sample_cells(self, candidates: List[Coord], density: float) -> Set[Coord]:
        if not candidates or density <= 0:
            return set()

        count = int(round(len(candidates) * density))
        count = max(0, min(count, len(candidates)))

        if count == 0:
            return set()

        indices = self.rng.choice(len(candidates), size=count, replace=False)
        return {candidates[int(i)] for i in indices}

    def _generate_hazards(self) -> Tuple[Set[Coord], Set[Coord]]:
        candidates = self._candidate_cells()

        if self.scenario_type == "flood":
            water_density = max(self.water_density, 0.16)
            fire_density = min(self.fire_density, 0.03)
        elif self.scenario_type == "wildfire":
            water_density = min(self.water_density, 0.03)
            fire_density = max(self.fire_density, 0.16)
        elif self.scenario_type == "corridor":
            water_density = self.water_density
            fire_density = self.fire_density
        else:
            water_density = self.water_density
            fire_density = self.fire_density

        water_cells = self._sample_cells(candidates, water_density)
        remaining = [cell for cell in candidates if cell not in water_cells]
        fire_cells = self._sample_cells(remaining, fire_density)

        if self.scenario_type == "corridor":
            # Keep a rough diagonal corridor less hazardous, so the map
            # feels like a blocked evacuation route rather than pure noise.
            corridor = {
                (min(x, self.width - 1), min(x, self.height - 1))
                for x in range(min(self.width, self.height))
            }
            water_cells -= corridor
            fire_cells -= corridor

        return water_cells, fire_cells

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = self.xy_to_state(*self.start_xy)
        self.steps = 0
        self.done = False
        return self.state

    def xy_to_state(self, x: int, y: int) -> int:
        return y * self.width + x

    def state_to_xy(self, state: int) -> Coord:
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
        outcomes = [
            (self.success_prob, self._move(state, action)),
            (self.slip_prob, self._move(state, SLIP_ACTIONS[action][0])),
            (self.slip_prob, self._move(state, SLIP_ACTIONS[action][1])),
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
        x, y, water, fire = self.state_tuple(state)
        gx, gy = self.goal_xy

        max_dist = (self.width - 1) + (self.height - 1)
        dist = abs(gx - x) + abs(gy - y)

        return np.array(
            [
                x / max(1, self.width - 1),
                y / max(1, self.height - 1),
                water,
                fire,
                dist / max(1, max_dist),
                1.0,
            ],
            dtype=float,
        )

    def format_grid(self, policy: Optional[np.ndarray] = None) -> str:
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
