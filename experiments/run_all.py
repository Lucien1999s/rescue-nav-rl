from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rescue_nav_rl.agents import (
    train_actor_critic,
    train_q_learning,
    train_sarsa,
    train_sarsa_lambda,
)


EPISODES = 100
TRIALS = 100
BASE_SEED = 42


def run_trials(
    train_fn: Callable,
    trials: int = TRIALS,
    episodes: int = EPISODES,
    base_seed: int = BASE_SEED,
    **kwargs,
) -> Dict[str, np.ndarray]:
    all_rewards = np.zeros((trials, episodes), dtype=float)
    all_success = np.zeros((trials, episodes), dtype=bool)
    all_steps = np.zeros((trials, episodes), dtype=int)

    for trial in range(trials):
        result = train_fn(
            episodes=episodes,
            seed=base_seed + trial,
            **kwargs,
        )

        all_rewards[trial] = result["episode_rewards"]
        all_success[trial] = result["reached_goal"]
        all_steps[trial] = result["episode_steps"]

    return {
        "mean_rewards": all_rewards.mean(axis=0),
        "std_rewards": all_rewards.std(axis=0),
        "mean_success": all_success.mean(axis=0),
        "mean_steps": all_steps.mean(axis=0),
        "all_rewards": all_rewards,
    }


def summarize(name: str, result: Dict[str, np.ndarray]) -> None:
    mean_rewards = result["mean_rewards"]
    std_rewards = result["std_rewards"]
    mean_success = result["mean_success"]
    mean_steps = result["mean_steps"]

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(f"Episode 1 avg reward:       {mean_rewards[0]:8.2f} ± {std_rewards[0]:.2f}")
    print(f"Episode 100 avg reward:     {mean_rewards[-1]:8.2f} ± {std_rewards[-1]:.2f}")
    print(f"Last 10 avg reward:         {mean_rewards[-10:].mean():8.2f}")
    print(f"Last 10 success rate:       {mean_success[-10:].mean():8.2%}")
    print(f"Last 10 avg steps:          {mean_steps[-10:].mean():8.2f}")


def main() -> None:
    experiments: List[tuple[str, Callable, dict]] = [
        (
            "Q-learning",
            train_q_learning,
            {
                "alpha": 0.3,
                "epsilon": 0.1,
                "gamma": 0.95,
            },
        ),
        (
            "SARSA",
            train_sarsa,
            {
                "alpha": 0.3,
                "epsilon": 0.1,
                "gamma": 0.95,
            },
        ),
        (
            "SARSA(lambda=0.1)",
            train_sarsa_lambda,
            {
                "alpha": 0.2,
                "epsilon": 0.1,
                "gamma": 0.95,
                "lam": 0.1,
            },
        ),
        (
            "SARSA(lambda=0.5)",
            train_sarsa_lambda,
            {
                "alpha": 0.2,
                "epsilon": 0.1,
                "gamma": 0.95,
                "lam": 0.5,
            },
        ),
        (
            "SARSA(lambda=0.9)",
            train_sarsa_lambda,
            {
                "alpha": 0.1,
                "epsilon": 0.1,
                "gamma": 0.95,
                "lam": 0.9,
            },
        ),
        (
            "Actor-Critic",
            train_actor_critic,
            {
                "alpha_actor": 0.02,
                "alpha_critic": 0.1,
                "gamma": 0.95,
            },
        ),
    ]

    print(f"Running {TRIALS} trials x {EPISODES} episodes")
    print("No files will be written.")

    for name, train_fn, kwargs in experiments:
        result = run_trials(train_fn, **kwargs)
        summarize(name, result)


if __name__ == "__main__":
    main()
