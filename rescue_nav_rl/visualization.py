from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from rescue_nav_rl.gridworld import ACTION_NAMES, ACTION_SYMBOLS, GridworldEnv


COLORS = {
    "background": "#0b1020",
    "panel": "#111827",
    "grid": "#374151",
    "safe": "#1f2937",
    "start": "#2563eb",
    "goal": "#16a34a",
    "water": "#0891b2",
    "fire": "#dc2626",
    "text": "#f9fafb",
    "muted": "#9ca3af",
    "path": "#facc15",
    "arrow": "#f9fafb",
}


def simulate_policy(
    policy: np.ndarray,
    seed: int = 0,
    max_steps: int = 100,
) -> Dict:
    """
    Simulate one episode using a deterministic policy.

    Returns a dictionary containing states, actions, rewards,
    total reward, success flag, and hazard exposure counts.
    """
    env = GridworldEnv(seed=seed, max_steps=max_steps)
    state = env.reset(seed=seed)

    states = [state]
    actions = []
    rewards = []

    water_visits = 0
    fire_visits = 0

    while True:
        action = int(policy[state])
        result = env.step(action)

        next_state = result.next_state
        reward = result.reward

        actions.append(action)
        rewards.append(reward)
        states.append(next_state)

        if env.is_water(next_state):
            water_visits += 1
        if env.is_fire(next_state):
            fire_visits += 1

        state = next_state

        if result.done:
            break

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "total_reward": float(sum(rewards)),
        "steps": len(actions),
        "reached_goal": bool(env.is_goal(states[-1])),
        "water_visits": water_visits,
        "fire_visits": fire_visits,
    }


def _cell_label(env: GridworldEnv, state: int) -> str:
    x, y = env.state_to_xy(state)

    if (x, y) == env.start_xy:
        return "S"
    if (x, y) == env.goal_xy:
        return "G"
    if (x, y) in env.water_cells:
        return "W"
    if (x, y) in env.fire_cells:
        return "F"
    return ""


def _cell_color(env: GridworldEnv, state: int) -> str:
    x, y = env.state_to_xy(state)

    if (x, y) == env.start_xy:
        return COLORS["start"]
    if (x, y) == env.goal_xy:
        return COLORS["goal"]
    if (x, y) in env.water_cells:
        return COLORS["water"]
    if (x, y) in env.fire_cells:
        return COLORS["fire"]
    return COLORS["safe"]


def plot_policy_map(
    policy: Optional[np.ndarray] = None,
    trajectory: Optional[List[int]] = None,
    title: str = "Rescue Navigation Policy",
):
    """
    Plot the hazardous gridworld with optional learned policy arrows
    and optional simulated trajectory path.
    """
    env = GridworldEnv()

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    for y in range(env.height):
        for x in range(env.width):
            state = env.xy_to_state(x, y)
            color = _cell_color(env, state)

            rect = Rectangle(
                (x, env.height - 1 - y),
                1,
                1,
                facecolor=color,
                edgecolor=COLORS["grid"],
                linewidth=2,
            )
            ax.add_patch(rect)

            label = _cell_label(env, state)
            if label:
                ax.text(
                    x + 0.5,
                    env.height - 1 - y + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                    color=COLORS["text"],
                )

            if policy is not None and label not in {"G"}:
                action = int(policy[state])
                symbol = ACTION_SYMBOLS[action]

                ax.text(
                    x + 0.5,
                    env.height - 1 - y + 0.20,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=20,
                    color=COLORS["arrow"],
                    alpha=0.85,
                )

    if trajectory is not None and len(trajectory) > 1:
        xs = []
        ys = []

        for state in trajectory:
            x, y = env.state_to_xy(state)
            xs.append(x + 0.5)
            ys.append(env.height - 1 - y + 0.5)

        ax.plot(
            xs,
            ys,
            color=COLORS["path"],
            linewidth=4,
            marker="o",
            markersize=8,
            markerfacecolor=COLORS["path"],
            markeredgecolor=COLORS["background"],
            alpha=0.95,
        )

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xticks(np.arange(0, env.width + 1, 1))
    ax.set_yticks(np.arange(0, env.height + 1, 1))
    ax.grid(color=COLORS["grid"], linewidth=1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, color=COLORS["text"], fontsize=16, fontweight="bold", pad=18)

    legend_items = [
        ("S", "Start", COLORS["start"]),
        ("G", "Goal", COLORS["goal"]),
        ("W", "Water hazard", COLORS["water"]),
        ("F", "Fire hazard", COLORS["fire"]),
    ]

    legend_text = "   ".join([f"{symbol}: {name}" for symbol, name, _ in legend_items])
    ax.text(
        0,
        -0.35,
        legend_text,
        color=COLORS["muted"],
        fontsize=11,
        transform=ax.transData,
    )

    plt.tight_layout()
    return fig


def plot_learning_curve(
    episode_rewards: np.ndarray,
    title: str = "Training Reward Curve",
):
    """
    Plot reward per episode for a single training run.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["panel"])

    episodes = np.arange(1, len(episode_rewards) + 1)

    ax.plot(
        episodes,
        episode_rewards,
        linewidth=2.5,
        color=COLORS["path"],
        label="Episode reward",
    )

    if len(episode_rewards) >= 10:
        window = 10
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode="valid",
        )
        ax.plot(
            np.arange(window, len(episode_rewards) + 1),
            moving_avg,
            linewidth=2,
            color="#38bdf8",
            label="10-episode moving average",
        )

    ax.set_title(title, color=COLORS["text"], fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Episode", color=COLORS["muted"])
    ax.set_ylabel("Undiscounted reward", color=COLORS["muted"])

    ax.tick_params(colors=COLORS["muted"])
    ax.grid(color=COLORS["grid"], alpha=0.35)

    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    legend = ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"])
    for text in legend.get_texts():
        text.set_color(COLORS["text"])

    plt.tight_layout()
    return fig


def describe_policy_rollout(rollout: Dict) -> Dict[str, str]:
    """
    Convert rollout metrics into display-friendly strings.
    """
    status = "Success" if rollout["reached_goal"] else "Timeout"

    return {
        "status": status,
        "total_reward": f"{rollout['total_reward']:.1f}",
        "steps": str(rollout["steps"]),
        "water_visits": str(rollout["water_visits"]),
        "fire_visits": str(rollout["fire_visits"]),
    }


def action_sequence_text(actions: List[int]) -> str:
    if not actions:
        return ""

    return " → ".join(ACTION_NAMES[int(action)] for action in actions)
