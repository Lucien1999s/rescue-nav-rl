from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    results: Dict[str, Dict[str, np.ndarray]],
    output_path: str | Path,
    title: str,
) -> None:
    """
    Plot mean episode reward with standard deviation error bars.

    results format:
        {
            algorithm_name: {
                "mean_rewards": np.ndarray,
                "std_rewards": np.ndarray,
            }
        }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for name, result in results.items():
        episodes = np.arange(1, len(result["mean_rewards"]) + 1)
        mean_rewards = result["mean_rewards"]
        std_rewards = result["std_rewards"]

        plt.plot(episodes, mean_rewards, label=name)

        # Use error bars every 5 episodes to keep the plot readable.
        plt.errorbar(
            episodes,
            mean_rewards,
            yerr=std_rewards,
            fmt="none",
            alpha=0.25,
            errorevery=5,
            capsize=2,
        )

    plt.xlabel("Episode")
    plt.ylabel("Average Undiscounted Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_learning_curve(
    result: Dict[str, np.ndarray],
    output_path: str | Path,
    title: str,
    label: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(result["mean_rewards"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, result["mean_rewards"], label=label)

    plt.errorbar(
        episodes,
        result["mean_rewards"],
        yerr=result["std_rewards"],
        fmt="none",
        alpha=0.25,
        errorevery=5,
        capsize=2,
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Undiscounted Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
