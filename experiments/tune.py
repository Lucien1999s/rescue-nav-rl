from __future__ import annotations

import argparse
import itertools
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


DEFAULT_EPISODES = 100
DEFAULT_TRIALS = 10
BASE_SEED = 1000


def evaluate_config(
    train_fn: Callable,
    config: Dict,
    episodes: int,
    trials: int,
    final_window: int,
) -> Dict[str, float]:
    rewards = np.zeros((trials, episodes), dtype=float)
    success = np.zeros((trials, episodes), dtype=bool)
    steps = np.zeros((trials, episodes), dtype=int)

    for trial in range(trials):
        result = train_fn(
            episodes=episodes,
            seed=BASE_SEED + trial,
            **config,
        )

        rewards[trial] = result["episode_rewards"]
        success[trial] = result["reached_goal"]
        steps[trial] = result["episode_steps"]

    final_rewards = rewards[:, -final_window:]
    final_success = success[:, -final_window:]
    final_steps = steps[:, -final_window:]

    return {
        "score": float(final_rewards.mean()),
        "std": float(final_rewards.std()),
        "success_rate": float(final_success.mean()),
        "avg_steps": float(final_steps.mean()),
    }


def tune_grid(
    name: str,
    train_fn: Callable,
    configs: List[Dict],
    episodes: int,
    trials: int,
    final_window: int,
) -> Dict:
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    records = []

    for idx, config in enumerate(configs, start=1):
        metrics = evaluate_config(
            train_fn=train_fn,
            config=config,
            episodes=episodes,
            trials=trials,
            final_window=final_window,
        )

        record = {
            "config": config,
            **metrics,
        }
        records.append(record)

        print(
            f"[{idx:02d}/{len(configs):02d}] "
            f"config={config} | "
            f"score={metrics['score']:.2f}, "
            f"std={metrics['std']:.2f}, "
            f"success={metrics['success_rate']:.2%}, "
            f"steps={metrics['avg_steps']:.2f}"
        )

    # Higher score is better.
    # Tie-breakers: higher success, lower steps, lower std.
    best = max(
        records,
        key=lambda r: (
            r["score"],
            r["success_rate"],
            -r["avg_steps"],
            -r["std"],
        ),
    )

    print()
    print(f"Best {name}:")
    print(f"  config:       {best['config']}")
    print(f"  score:        {best['score']:.2f}")
    print(f"  std:          {best['std']:.2f}")
    print(f"  success rate: {best['success_rate']:.2%}")
    print(f"  avg steps:    {best['avg_steps']:.2f}")

    return best


def make_configs(param_grid: Dict[str, List]) -> List[Dict]:
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]

    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--final-window", type=int, default=20)
    args = parser.parse_args()

    print(
        f"Tuning with {args.trials} trials x {args.episodes} episodes "
        f"(metric: average reward over final {args.final_window} episodes)"
    )
    print("No files will be written.")

    q_learning_configs = make_configs(
        {
            "alpha": [0.1, 0.2, 0.3, 0.5],
            "epsilon": [0.05, 0.1, 0.2],
            "gamma": [0.95],
        }
    )

    sarsa_configs = make_configs(
        {
            "alpha": [0.1, 0.2, 0.3, 0.5],
            "epsilon": [0.05, 0.1, 0.2],
            "gamma": [0.95],
        }
    )

    sarsa_lambda_configs = make_configs(
        {
            "alpha": [0.05, 0.1, 0.2, 0.3],
            "epsilon": [0.05, 0.1, 0.2],
            "gamma": [0.95],
            "lam": [0.1, 0.5, 0.9],
        }
    )

    actor_critic_configs = make_configs(
        {
            "alpha_actor": [0.005, 0.01, 0.02, 0.05],
            "alpha_critic": [0.05, 0.1, 0.2],
            "gamma": [0.95],
        }
    )

    best_results = {
        "Q-learning": tune_grid(
            "Q-learning",
            train_q_learning,
            q_learning_configs,
            args.episodes,
            args.trials,
            args.final_window,
        ),
        "SARSA": tune_grid(
            "SARSA",
            train_sarsa,
            sarsa_configs,
            args.episodes,
            args.trials,
            args.final_window,
        ),
        "SARSA(lambda)": tune_grid(
            "SARSA(lambda)",
            train_sarsa_lambda,
            sarsa_lambda_configs,
            args.episodes,
            args.trials,
            args.final_window,
        ),
        "Actor-Critic": tune_grid(
            "Actor-Critic",
            train_actor_critic,
            actor_critic_configs,
            args.episodes,
            args.trials,
            args.final_window,
        ),
    }

    print()
    print("=" * 80)
    print("Final selected hyperparameters")
    print("=" * 80)

    for name, best in best_results.items():
        print(f"{name}: {best['config']} | score={best['score']:.2f}")


if __name__ == "__main__":
    main()
