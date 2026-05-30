from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from rescue_nav_rl.gridworld import GridworldEnv


def random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    """
    Argmax with random tie-breaking.
    This avoids always choosing action 0 when Q-values are tied.
    """
    max_value = np.max(values)
    candidates = np.flatnonzero(values == max_value)
    return int(rng.choice(candidates))


def epsilon_greedy_action(
    q: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """
    Select action using epsilon-greedy policy.
    """
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    return random_argmax(q[state], rng)


def greedy_policy(q: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Extract deterministic greedy policy from Q-table.
    """
    rng = np.random.default_rng(seed)
    policy = np.zeros(q.shape[0], dtype=int)

    for state in range(q.shape[0]):
        policy[state] = random_argmax(q[state], rng)

    return policy


def train_q_learning(
    episodes: int = 100,
    alpha: float = 0.3,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    seed: Optional[int] = None,
    env_factory: Callable[..., GridworldEnv] = GridworldEnv,
) -> Dict[str, np.ndarray]:
    """
    Tabular Q-learning.

    Off-policy update:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """
    rng = np.random.default_rng(seed)
    env = env_factory(seed=seed)

    q = np.zeros((env.num_states, env.num_actions), dtype=float)
    episode_rewards = np.zeros(episodes, dtype=float)
    episode_steps = np.zeros(episodes, dtype=int)
    reached_goal = np.zeros(episodes, dtype=bool)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0

        while True:
            action = epsilon_greedy_action(q, state, epsilon, rng)
            result = env.step(action)

            next_state = result.next_state
            reward = result.reward
            done = result.done

            if done:
                target = reward
            else:
                target = reward + gamma * np.max(q[next_state])

            q[state, action] += alpha * (target - q[state, action])

            total_reward += reward
            state = next_state

            if done:
                episode_rewards[episode] = total_reward
                episode_steps[episode] = result.info["steps"]
                reached_goal[episode] = result.info["reached_goal"]
                break

    return {
        "q": q,
        "policy": greedy_policy(q, seed=seed),
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "reached_goal": reached_goal,
    }


def train_sarsa(
    episodes: int = 100,
    alpha: float = 0.3,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    seed: Optional[int] = None,
    env_factory: Callable[..., GridworldEnv] = GridworldEnv,
) -> Dict[str, np.ndarray]:
    """
    Tabular SARSA.

    On-policy update:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
    """
    rng = np.random.default_rng(seed)
    env = env_factory(seed=seed)

    q = np.zeros((env.num_states, env.num_actions), dtype=float)
    episode_rewards = np.zeros(episodes, dtype=float)
    episode_steps = np.zeros(episodes, dtype=int)
    reached_goal = np.zeros(episodes, dtype=bool)

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_action(q, state, epsilon, rng)
        total_reward = 0.0

        while True:
            result = env.step(action)

            next_state = result.next_state
            reward = result.reward
            done = result.done

            if done:
                target = reward
                q[state, action] += alpha * (target - q[state, action])
            else:
                next_action = epsilon_greedy_action(q, next_state, epsilon, rng)
                target = reward + gamma * q[next_state, next_action]
                q[state, action] += alpha * (target - q[state, action])

                state = next_state
                action = next_action

            total_reward += reward

            if done:
                episode_rewards[episode] = total_reward
                episode_steps[episode] = result.info["steps"]
                reached_goal[episode] = result.info["reached_goal"]
                break

    return {
        "q": q,
        "policy": greedy_policy(q, seed=seed),
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "reached_goal": reached_goal,
    }


def train_sarsa_lambda(
    episodes: int = 100,
    alpha: float = 0.2,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    lam: float = 0.5,
    seed: Optional[int] = None,
    env_factory: Callable[..., GridworldEnv] = GridworldEnv,
) -> Dict[str, np.ndarray]:
    """
    Tabular SARSA(lambda) with backward view and accumulating eligibility traces.

    Accumulating trace:
        E(s,a) <- E(s,a) + 1

    TD error:
        delta = r + gamma * Q(s',a') - Q(s,a)

    Update:
        Q <- Q + alpha * delta * E
        E <- gamma * lambda * E
    """
    rng = np.random.default_rng(seed)
    env = env_factory(seed=seed)

    q = np.zeros((env.num_states, env.num_actions), dtype=float)
    episode_rewards = np.zeros(episodes, dtype=float)
    episode_steps = np.zeros(episodes, dtype=int)
    reached_goal = np.zeros(episodes, dtype=bool)

    for episode in range(episodes):
        traces = np.zeros_like(q)

        state = env.reset()
        action = epsilon_greedy_action(q, state, epsilon, rng)
        total_reward = 0.0

        while True:
            result = env.step(action)

            next_state = result.next_state
            reward = result.reward
            done = result.done

            if done:
                delta = reward - q[state, action]
                next_action = None
            else:
                next_action = epsilon_greedy_action(q, next_state, epsilon, rng)
                delta = reward + gamma * q[next_state, next_action] - q[state, action]

            traces[state, action] += 1.0
            q += alpha * delta * traces
            traces *= gamma * lam

            total_reward += reward

            if done:
                episode_rewards[episode] = total_reward
                episode_steps[episode] = result.info["steps"]
                reached_goal[episode] = result.info["reached_goal"]
                break

            state = next_state
            action = next_action

    return {
        "q": q,
        "policy": greedy_policy(q, seed=seed),
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "reached_goal": reached_goal,
    }



def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def actor_policy(actor_preferences: np.ndarray, env: GridworldEnv) -> np.ndarray:
    """
    Extract deterministic greedy policy from learned actor preferences.
    """
    policy = np.zeros(env.num_states, dtype=int)

    for state in range(env.num_states):
        policy[state] = int(np.argmax(actor_preferences[state]))

    return policy


def train_actor_critic(
    episodes: int = 100,
    alpha_actor: float = 0.02,
    alpha_critic: float = 0.1,
    gamma: float = 0.95,
    seed: Optional[int] = None,
    env_factory: Callable[..., GridworldEnv] = GridworldEnv,
) -> Dict[str, np.ndarray]:
    """
    Tabular linear Actor-Critic using one-hot state features.

    This is equivalent to linear function approximation with one-hot
    features over the 16 grid states.

    Actor:
        softmax policy over state-action preferences H[s, a]

    Critic:
        tabular state-value estimate V[s]

    TD(0) error:
        delta = r + gamma * V[s'] - V[s]

    Actor update:
        H[s, :] <- H[s, :] + alpha_actor * delta * grad log pi(a|s)

    Critic update:
        V[s] <- V[s] + alpha_critic * delta
    """
    rng = np.random.default_rng(seed)
    env = env_factory(seed=seed)

    actor_preferences = np.zeros((env.num_states, env.num_actions), dtype=float)
    critic_values = np.zeros(env.num_states, dtype=float)

    episode_rewards = np.zeros(episodes, dtype=float)
    episode_steps = np.zeros(episodes, dtype=int)
    reached_goal = np.zeros(episodes, dtype=bool)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0

        while True:
            probs = softmax(actor_preferences[state])
            action = int(rng.choice(env.num_actions, p=probs))

            result = env.step(action)

            next_state = result.next_state
            reward = result.reward
            done = result.done

            value = critic_values[state]
            next_value = 0.0 if done else critic_values[next_state]

            delta = reward + gamma * next_value - value

            critic_values[state] += alpha_critic * delta

            grad_log_policy = -probs
            grad_log_policy[action] += 1.0

            actor_preferences[state] += alpha_actor * delta * grad_log_policy

            total_reward += reward

            if done:
                episode_rewards[episode] = total_reward
                episode_steps[episode] = result.info["steps"]
                reached_goal[episode] = result.info["reached_goal"]
                break

            state = next_state

    return {
        "actor_preferences": actor_preferences,
        "critic_values": critic_values,
        "policy": actor_policy(actor_preferences, env),
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "reached_goal": reached_goal,
    }
