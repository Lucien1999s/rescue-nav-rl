from __future__ import annotations

import argparse
import csv
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
from rescue_nav_rl.plotting import plot_learning_curves, plot_single_learning_curve


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


def summarize(name: str, result: Dict[str, np.ndarray]) -> Dict[str, float]:
    mean_rewards = result["mean_rewards"]
    std_rewards = result["std_rewards"]
    mean_success = result["mean_success"]
    mean_steps = result["mean_steps"]

    summary = {
        "episode_1_avg_reward": float(mean_rewards[0]),
        "episode_1_std_reward": float(std_rewards[0]),
        "episode_100_avg_reward": float(mean_rewards[-1]),
        "episode_100_std_reward": float(std_rewards[-1]),
        "last_10_avg_reward": float(mean_rewards[-10:].mean()),
        "last_10_success_rate": float(mean_success[-10:].mean()),
        "last_10_avg_steps": float(mean_steps[-10:].mean()),
    }

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(f"Episode 1 avg reward:       {summary['episode_1_avg_reward']:8.2f} ± {summary['episode_1_std_reward']:.2f}")
    print(f"Episode 100 avg reward:     {summary['episode_100_avg_reward']:8.2f} ± {summary['episode_100_std_reward']:.2f}")
    print(f"Last 10 avg reward:         {summary['last_10_avg_reward']:8.2f}")
    print(f"Last 10 success rate:       {summary['last_10_success_rate']:8.2%}")
    print(f"Last 10 avg steps:          {summary['last_10_avg_steps']:8.2f}")

    return summary


def save_summary_csv(
    summaries: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algorithm",
        "episode_1_avg_reward",
        "episode_1_std_reward",
        "episode_100_avg_reward",
        "episode_100_std_reward",
        "last_10_avg_reward",
        "last_10_success_rate",
        "last_10_avg_steps",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for algorithm, summary in summaries.items():
            row = {"algorithm": algorithm}
            row.update(summary)
            writer.writerow(row)


def save_report_outputs(
    results: Dict[str, Dict[str, np.ndarray]],
    summaries: Dict[str, Dict[str, float]],
) -> None:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    plot_single_learning_curve(
        results["Q-learning"],
        reports_dir / "q_learning_curve.png",
        title="Q-learning Learning Curve",
        label="Q-learning",
    )

    plot_single_learning_curve(
        results["SARSA"],
        reports_dir / "sarsa_curve.png",
        title="SARSA Learning Curve",
        label="SARSA",
    )

    sarsa_lambda_results = {
        name: result
        for name, result in results.items()
        if name.startswith("SARSA(lambda")
    }

    plot_learning_curves(
        sarsa_lambda_results,
        reports_dir / "sarsa_lambda_curves.png",
        title="SARSA(lambda) Learning Curves",
    )

    plot_single_learning_curve(
        results["Actor-Critic"],
        reports_dir / "actor_critic_curve.png",
        title="Actor-Critic Learning Curve",
        label="Actor-Critic",
    )

    plot_learning_curves(
        results,
        reports_dir / "algorithm_comparison_curves.png",
        title="Algorithm Comparison",
    )

    save_summary_csv(
        summaries,
        reports_dir / "summary_table.csv",
    )

    print()
    print("Saved report outputs to reports/:")
    print("- q_learning_curve.png")
    print("- sarsa_curve.png")
    print("- sarsa_lambda_curves.png")
    print("- actor_critic_curve.png")
    print("- algorithm_comparison_curves.png")
    print("- summary_table.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots and summary table to reports/.",
    )
    args = parser.parse_args()

    experiments: List[tuple[str, Callable, dict]] = [
        (
            "Q-learning",
            train_q_learning,
            {
                "alpha": 0.1,
                "epsilon": 0.05,
                "gamma": 0.95,
            },
        ),
        (
            "SARSA",
            train_sarsa,
            {
                "alpha": 0.1,
                "epsilon": 0.1,
                "gamma": 0.95,
            },
        ),
        (
            "SARSA(lambda=0.1)",
            train_sarsa_lambda,
            {
                "alpha": 0.3,
                "epsilon": 0.05,
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
                "alpha_critic": 0.2,
                "gamma": 0.95,
            },
        ),
    ]

    print(f"Running {TRIALS} trials x {EPISODES} episodes")

    if args.save:
        print("Saving plots and summary table to reports/.")
    else:
        print("No files will be written. Use --save to generate report outputs.")

    results: Dict[str, Dict[str, np.ndarray]] = {}
    summaries: Dict[str, Dict[str, float]] = {}

    for name, train_fn, kwargs in experiments:
        result = run_trials(train_fn, **kwargs)
        results[name] = result
        summaries[name] = summarize(name, result)

    if args.save:
        save_report_outputs(results, summaries)


if __name__ == "__main__":
    main()
