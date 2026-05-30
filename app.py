from __future__ import annotations

import pandas as pd
import streamlit as st

from rescue_nav_rl.dashboard import (
    DEFAULT_AGENT_CONFIGS,
    build_scenario,
    evaluate_policy,
    rank_agents,
    recommended_agent,
    run_agent_benchmark,
)
from rescue_nav_rl.visualization import (
    action_sequence_text,
    describe_policy_rollout,
    plot_learning_curve,
    plot_policy_map,
)


st.set_page_config(
    page_title="Rescue Route Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
.stApp {
    background: #060a14;
    color: #f8fafc;
}

.block-container {
    padding-top: 4.2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

[data-testid="stSidebar"] {
    background: #080d18;
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}

h1, h2, h3 {
    color: #f8fafc;
    letter-spacing: -0.02em;
}

p {
    color: #cbd5e1;
}

.badge {
    display: inline-block;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.10);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #bae6fd;
    font-size: 0.74rem;
    margin-right: 0.38rem;
}

.badge-warning {
    background: rgba(250, 204, 21, 0.10);
    border-color: rgba(250, 204, 21, 0.25);
    color: #fde68a;
}

.kpi-row {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin: 0.5rem 0 0.25rem 0;
}

.kpi {
    background: rgba(2, 6, 23, 0.58);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 999px;
    padding: 0.42rem 0.72rem;
    font-size: 0.82rem;
    color: #cbd5e1;
}

.kpi b {
    color: #f8fafc;
}

.recommend-block {
    border-left: 4px solid #facc15;
    padding-left: 0.9rem;
    margin-bottom: 0.75rem;
}

.muted {
    color: #94a3b8;
    font-size: 0.9rem;
}

div.stButton > button {
    background: linear-gradient(90deg, #facc15, #fb923c);
    color: #111827;
    border: 0;
    border-radius: 12px;
    font-weight: 850;
    height: 2.6rem;
}

div.stButton > button:hover {
    filter: brightness(1.08);
    color: #020617;
    border: 0;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(148, 163, 184, 0.22);
    background: rgba(8, 13, 24, 0.34);
}
</style>
""",
    unsafe_allow_html=True,
)


DEFAULT_CONFIG = {
    "scenario_type": "mixed",
    "map_size": 10,
    "water_density": 0.10,
    "fire_density": 0.08,
    "slip_prob": 0.10,
    "scenario_seed": 7,
    "selected_agents": ["Q-learning", "SARSA", "SARSA(lambda)", "Actor-Critic"],
    "episodes": 400,
    "rollout_count": 20,
    "train_seed": 42,
}


if "mission_config" not in st.session_state:
    st.session_state.mission_config = DEFAULT_CONFIG.copy()

if "mission_result" not in st.session_state:
    st.session_state.mission_result = None


@st.cache_data(show_spinner=False)
def run_benchmark_cached(
    width: int,
    height: int,
    water_density: float,
    fire_density: float,
    slip_prob: float,
    scenario_type: str,
    scenario_seed: int,
    train_seed: int,
    rollout_count: int,
    episodes: int,
    selected_agents: tuple[str, ...],
):
    env_template = build_scenario(
        width=width,
        height=height,
        water_density=water_density,
        fire_density=fire_density,
        slip_prob=slip_prob,
        scenario_type=scenario_type,
        seed=scenario_seed,
    )

    results = run_agent_benchmark(
        env_template=env_template,
        selected_agents=list(selected_agents),
        train_seed=train_seed,
        rollout_count=rollout_count,
        episodes=episodes,
    )

    return env_template, results


def format_leaderboard(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    columns = [
        "Rank",
        "Agent",
        "Mission Score",
        "Success Rate",
        "Avg Reward",
        "Avg Steps",
        "Water Visits",
        "Fire Visits",
    ]

    df = df[columns].copy()
    df["Mission Score"] = df["Mission Score"].map(lambda x: f"{x:.1f}")
    df["Success Rate"] = df["Success Rate"].map(lambda x: f"{x:.0%}")
    df["Avg Reward"] = df["Avg Reward"].map(lambda x: f"{x:.1f}")
    df["Avg Steps"] = df["Avg Steps"].map(lambda x: f"{x:.1f}")
    df["Water Visits"] = df["Water Visits"].map(lambda x: f"{x:.2f}")
    df["Fire Visits"] = df["Fire Visits"].map(lambda x: f"{x:.2f}")

    return df


def run_current_mission() -> None:
    cfg = st.session_state.mission_config

    if not cfg["selected_agents"]:
        st.warning("Select at least one agent.")
        return

    with st.spinner("Training agents and evaluating deployment routes..."):
        env_template, results = run_benchmark_cached(
            width=cfg["map_size"],
            height=cfg["map_size"],
            water_density=cfg["water_density"],
            fire_density=cfg["fire_density"],
            slip_prob=cfg["slip_prob"],
            scenario_type=cfg["scenario_type"],
            scenario_seed=int(cfg["scenario_seed"]),
            train_seed=int(cfg["train_seed"]),
            rollout_count=int(cfg["rollout_count"]),
            episodes=int(cfg["episodes"]),
            selected_agents=tuple(cfg["selected_agents"]),
        )

    st.session_state.mission_result = {
        "env_template": env_template,
        "results": results,
    }


def render_config_summary(cfg: dict) -> None:
    st.markdown(
        f"""
<div class="kpi-row">
    <div class="kpi">Incident <b>{cfg["scenario_type"]}</b></div>
    <div class="kpi">Map <b>{cfg["map_size"]}×{cfg["map_size"]}</b></div>
    <div class="kpi">Water <b>{cfg["water_density"]:.2f}</b></div>
    <div class="kpi">Fire <b>{cfg["fire_density"]:.2f}</b></div>
    <div class="kpi">Uncertainty <b>{cfg["slip_prob"]:.2f}</b></div>
    <div class="kpi">Agents <b>{len(cfg["selected_agents"])}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Sidebar Mission Setup
# -----------------------------

st.sidebar.markdown("## Mission Setup")

cfg = st.session_state.mission_config

scenario_type = st.sidebar.selectbox(
    "Incident",
    ["mixed", "flood", "wildfire", "corridor"],
    index=["mixed", "flood", "wildfire", "corridor"].index(cfg["scenario_type"]),
)

map_size = st.sidebar.selectbox(
    "Map size",
    [6, 8, 10, 12],
    index=[6, 8, 10, 12].index(cfg["map_size"]),
)

water_density = st.sidebar.slider(
    "Water hazard density",
    0.00,
    0.30,
    float(cfg["water_density"]),
    step=0.01,
)

fire_density = st.sidebar.slider(
    "Fire hazard density",
    0.00,
    0.30,
    float(cfg["fire_density"]),
    step=0.01,
)

slip_prob = st.sidebar.slider(
    "Movement uncertainty",
    0.00,
    0.25,
    float(cfg["slip_prob"]),
    step=0.01,
)

scenario_seed = st.sidebar.number_input(
    "Scenario seed",
    min_value=0,
    max_value=9999,
    value=int(cfg["scenario_seed"]),
)

selected_agents = st.sidebar.multiselect(
    "Agents",
    list(DEFAULT_AGENT_CONFIGS.keys()),
    default=cfg["selected_agents"],
)

episodes = st.sidebar.selectbox(
    "Training budget",
    [200, 400, 600, 800],
    index=[200, 400, 600, 800].index(cfg["episodes"]),
)

rollout_count = st.sidebar.selectbox(
    "Evaluation rollouts",
    [10, 20, 30, 50],
    index=[10, 20, 30, 50].index(cfg["rollout_count"]),
)

apply_and_deploy = st.sidebar.button("Apply and deploy", use_container_width=True)

current_sidebar_config = {
    "scenario_type": scenario_type,
    "map_size": int(map_size),
    "water_density": float(water_density),
    "fire_density": float(fire_density),
    "slip_prob": float(slip_prob),
    "scenario_seed": int(scenario_seed),
    "selected_agents": list(selected_agents),
    "episodes": int(episodes),
    "rollout_count": int(rollout_count),
    "train_seed": int(cfg["train_seed"]),
}

if apply_and_deploy:
    st.session_state.mission_config = current_sidebar_config
    run_current_mission()
    st.rerun()


# -----------------------------
# Header
# -----------------------------

top_left, top_right = st.columns([0.76, 0.24], gap="large")

with top_left:
    st.title("🛰️ Rescue Route Intelligence")
    st.caption(
        "Risk-aware RL simulator for evacuation routing under water, fire, and movement uncertainty."
    )
    st.markdown(
        """
<span class="badge">scenario generator</span>
<span class="badge">RL route learning</span>
<span class="badge badge-warning">hazard-aware deployment</span>
""",
        unsafe_allow_html=True,
    )

with top_right:
    st.write("")
    st.write("")
    deploy_now = st.button("Deploy current mission", use_container_width=True)

if deploy_now:
    st.session_state.mission_config = current_sidebar_config
    run_current_mission()
    st.rerun()


render_config_summary(st.session_state.mission_config)


# -----------------------------
# Main Content
# -----------------------------

mission_result = st.session_state.mission_result

if mission_result is None:
    cfg = st.session_state.mission_config

    preview_env = build_scenario(
        width=cfg["map_size"],
        height=cfg["map_size"],
        water_density=cfg["water_density"],
        fire_density=cfg["fire_density"],
        slip_prob=cfg["slip_prob"],
        scenario_type=cfg["scenario_type"],
        seed=int(cfg["scenario_seed"]),
    )

    left, right = st.columns([0.90, 1.10], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### Scenario Preview")
            st.pyplot(
                plot_policy_map(
                    env=preview_env,
                    title="Generated Disaster Scenario",
                    show_policy=False,
                    compact=True,
                ),
                use_container_width=True,
            )

    with right:
        with st.container(border=True):
            st.markdown("### Mission Briefing")
            st.markdown(
                """
Generate a stochastic disaster map, train multiple RL agents,
and recommend the safest deployable route based on success rate,
reward efficiency, route length, and hazard exposure.
"""
            )
            st.markdown(
                """
<div class="kpi-row">
    <div class="kpi"><b>S</b> base</div>
    <div class="kpi"><b>G</b> shelter</div>
    <div class="kpi"><b>≈</b> water hazard</div>
    <div class="kpi"><b>▲</b> fire hazard</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("Press **Deploy current mission** to train and evaluate agents.")

    st.stop()


env_template = mission_result["env_template"]
results = mission_result["results"]

ranked_rows = rank_agents(results)
leader = recommended_agent(results)
leader_payload = results[leader]

leader_rollout = evaluate_policy(
    leader_payload["train_result"]["policy"],
    env_template=env_template,
    rollout_seed=12345,
)

leader_metrics = describe_policy_rollout(leader_rollout)
leader_eval = leader_payload["evaluation"]

left, right = st.columns([0.94, 1.06], gap="large")

with left:
    with st.container(border=True):
        st.markdown(f"### Deployment Route — {leader}")
        st.pyplot(
            plot_policy_map(
                env=env_template,
                policy=leader_payload["train_result"]["policy"],
                trajectory=leader_rollout["states"],
                title="",
                show_policy=True,
                compact=True,
            ),
            use_container_width=True,
        )

with right:
    with st.container(border=True):
        st.markdown(f"### Recommended agent: {leader}")
        st.markdown(
            """
Selected by mission score: success rate first, then reward,
fewer hazard visits, and shorter routes.
"""
        )

        st.markdown(
            f"""
<div class="kpi-row">
    <div class="kpi">Replay <b>{leader_metrics["status"]}</b></div>
    <div class="kpi">Reward <b>{leader_metrics["total_reward"]}</b></div>
    <div class="kpi">Steps <b>{leader_metrics["steps"]}</b></div>
    <div class="kpi">Hazards <b>{int(leader_rollout["water_visits"] + leader_rollout["fire_visits"])}</b></div>
</div>
<div class="kpi-row">
    <div class="kpi">Eval success <b>{leader_eval["success_rate"]:.0%}</b></div>
    <div class="kpi">Avg reward <b>{leader_eval["avg_reward"]:.1f}</b></div>
    <div class="kpi">Avg steps <b>{leader_eval["avg_steps"]:.1f}</b></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### Agent Leaderboard")
        st.dataframe(
            format_leaderboard(ranked_rows),
            use_container_width=True,
            hide_index=True,
            height=245,
        )


# -----------------------------
# Details
# -----------------------------

st.markdown("")

tab_replay, tab_learning, tab_config = st.tabs(
    ["Replay", "Learning Curves", "Technical Config"]
)

with tab_replay:
    replay_agent = st.selectbox(
        "Replay agent",
        list(results.keys()),
        index=list(results.keys()).index(leader),
    )

    replay_seed = st.slider("Replay seed", 0, 9999, 12345)

    replay_payload = results[replay_agent]
    replay_rollout = evaluate_policy(
        replay_payload["train_result"]["policy"],
        env_template=env_template,
        rollout_seed=replay_seed,
    )

    step = st.slider(
        "Replay step",
        0,
        max(0, len(replay_rollout["states"]) - 1),
        len(replay_rollout["states"]) - 1,
    )

    st.pyplot(
        plot_policy_map(
            env=env_template,
            policy=replay_payload["train_result"]["policy"],
            trajectory=replay_rollout["states"][: step + 1],
            title=f"{replay_agent} Replay",
            show_policy=True,
            compact=True,
        ),
        use_container_width=True,
    )

    st.markdown(
        f"""
<p class="muted">
<b>Actions:</b> {action_sequence_text(replay_rollout["actions"][:step])}
</p>
<p class="muted">
<b>Total reward:</b> {replay_rollout["total_reward"]:.1f} |
<b>Steps:</b> {replay_rollout["steps"]} |
<b>Water:</b> {replay_rollout["water_visits"]} |
<b>Fire:</b> {replay_rollout["fire_visits"]}
</p>
""",
        unsafe_allow_html=True,
    )

with tab_learning:
    curve_agent = st.selectbox("Training curve", list(results.keys()))

    st.pyplot(
        plot_learning_curve(
            results[curve_agent]["train_result"]["episode_rewards"],
            title=f"{curve_agent} Training Reward",
        ),
        use_container_width=True,
    )

with tab_config:
    config_rows = []

    for agent_name in results:
        config_rows.append(
            {
                "Agent": agent_name,
                "Profile": DEFAULT_AGENT_CONFIGS[agent_name]["profile"],
                "Parameters": str(DEFAULT_AGENT_CONFIGS[agent_name]["params"]),
            }
        )

    st.dataframe(
        pd.DataFrame(config_rows),
        use_container_width=True,
        hide_index=True,
    )