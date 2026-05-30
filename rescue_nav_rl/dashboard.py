from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from rescue_nav_rl.agents import (
    train_actor_critic,
    train_q_learning,
    train_sarsa,
    train_sarsa_lambda,
)
from rescue_nav_rl.scenario import RescueScenarioEnv
from rescue_nav_rl.visualization import simulate_policy


DEFAULT_AGENT_CONFIGS = {
    "Q-learning": {
        "trainer": train_q_learning,
        "params": {
            "episodes": 400,
            "alpha": 0.1,
            "epsilon": 0.08,
            "gamma": 0.95,
        },
        "profile": "Fast value-based learner; strong baseline for discrete rescue routing.",
    },
    "SARSA": {
        "trainer": train_sarsa,
        "params": {
            "episodes": 400,
            "alpha": 0.1,
            "epsilon": 0.10,
            "gamma": 0.95,
        },
        "profile": "On-policy learner; often more conservative under stochastic movement.",
    },
    "SARSA(lambda)": {
        "trainer": train_sarsa_lambda,
        "params": {
            "episodes": 400,
            "alpha": 0.2,
            "epsilon": 0.08,
            "gamma": 0.95,
            "lam": 0.1,
        },
        "profile": "Eligibility-trace agent; balances immediate and multi-step credit assignment.",
    },
    "Actor-Critic": {
        "trainer": train_actor_critic,
        "params": {
            "episodes": 400,
            "alpha_actor": 0.02,
            "alpha_critic": 0.2,
            "gamma": 0.95,
        },
        "profile": "Policy-based softmax actor with TD(0) critic; useful comparison agent.",
    },
}


def build_scenario(
    width: int,
    height: int,
    water_density: float,
    fire_density: float,
    slip_prob: float,
    scenario_type: str,
    seed: int,
    max_steps: int | None = None,
) -> RescueScenarioEnv:
    if max_steps is None:
        max_steps = max(100, width * height * 2)

    return RescueScenarioEnv(
        width=width,
        height=height,
        water_density=water_density,
        fire_density=fire_density,
        slip_prob=slip_prob,
        max_steps=max_steps,
        scenario_type=scenario_type,
        seed=seed,
    )


def make_env_factory(env_template: RescueScenarioEnv) -> Callable:
    def env_factory(seed=None, max_steps=None):
        env = env_template.clone(seed=seed)
        if max_steps is not None:
            env.max_steps = max_steps
        return env

    return env_factory


def train_agent(
    agent_name: str,
    env_template: RescueScenarioEnv,
    seed: int,
    episodes: int | None = None,
) -> Dict:
    config = DEFAULT_AGENT_CONFIGS[agent_name]
    params = dict(config["params"])

    if episodes is not None:
        params["episodes"] = episodes

    env_factory = make_env_factory(env_template)

    result = config["trainer"](
        seed=seed,
        env_factory=env_factory,
        **params,
    )

    return result


def evaluate_policy(
    policy: np.ndarray,
    env_template: RescueScenarioEnv,
    rollout_seed: int,
) -> Dict:
    env_factory = make_env_factory(env_template)
    return simulate_policy(
        policy,
        seed=rollout_seed,
        max_steps=env_template.max_steps,
        env_factory=env_factory,
    )


def evaluate_policy_many(
    policy: np.ndarray,
    env_template: RescueScenarioEnv,
    rollout_seeds: List[int],
) -> Dict:
    rollouts = [
        evaluate_policy(policy, env_template, rollout_seed=seed)
        for seed in rollout_seeds
    ]

    success_rate = np.mean([r["reached_goal"] for r in rollouts])
    avg_reward = np.mean([r["total_reward"] for r in rollouts])
    avg_steps = np.mean([r["steps"] for r in rollouts])
    avg_water = np.mean([r["water_visits"] for r in rollouts])
    avg_fire = np.mean([r["fire_visits"] for r in rollouts])
    avg_hazard = avg_water + avg_fire

    return {
        "success_rate": float(success_rate),
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
        "avg_water_visits": float(avg_water),
        "avg_fire_visits": float(avg_fire),
        "avg_hazard_visits": float(avg_hazard),
        "rollouts": rollouts,
    }


def run_agent_benchmark(
    env_template: RescueScenarioEnv,
    selected_agents: List[str],
    train_seed: int,
    rollout_count: int = 20,
    episodes: int | None = None,
) -> Dict[str, Dict]:
    rollout_seeds = list(range(10_000, 10_000 + rollout_count))
    results = {}

    for idx, agent_name in enumerate(selected_agents):
        train_result = train_agent(
            agent_name=agent_name,
            env_template=env_template,
            seed=train_seed + idx * 100,
            episodes=episodes,
        )

        evaluation = evaluate_policy_many(
            train_result["policy"],
            env_template=env_template,
            rollout_seeds=rollout_seeds,
        )

        results[agent_name] = {
            "train_result": train_result,
            "evaluation": evaluation,
            "profile": DEFAULT_AGENT_CONFIGS[agent_name]["profile"],
        }

    return results


def rank_agents(results: Dict[str, Dict]) -> List[Dict]:
    rows = []

    for agent_name, payload in results.items():
        evaluation = payload["evaluation"]

        # Mission score: prioritize success, then reward, then fewer hazards and shorter routes.
        mission_score = (
            evaluation["success_rate"] * 1000.0
            + evaluation["avg_reward"]
            - evaluation["avg_hazard_visits"] * 25.0
            - evaluation["avg_steps"] * 0.5
        )

        rows.append(
            {
                "Agent": agent_name,
                "Mission Score": mission_score,
                "Success Rate": evaluation["success_rate"],
                "Avg Reward": evaluation["avg_reward"],
                "Avg Steps": evaluation["avg_steps"],
                "Water Visits": evaluation["avg_water_visits"],
                "Fire Visits": evaluation["avg_fire_visits"],
                "Hazard Visits": evaluation["avg_hazard_visits"],
            }
        )

    rows.sort(key=lambda row: row["Mission Score"], reverse=True)

    for idx, row in enumerate(rows, start=1):
        row["Rank"] = idx

    return rows


def recommended_agent(results: Dict[str, Dict]) -> str:
    ranked = rank_agents(results)
    return ranked[0]["Agent"]
