import argparse
import inspect
import os
import pickle
import logging
import sys
from pathlib import Path
import numpy as np
from tqdm import trange

# from citylearn.citylearn import CityLearnEnv
from environment.citylearn_env import SocFeasibleCityLearnEnv
from citylearn.utilities import write_json

# ----------------------------- Logger Setup -----------------------------
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def setup_logging(log_filepath):
    if not LOGGER.hasHandlers():
        handler = logging.FileHandler(log_filepath, mode='w')
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)

# ------------------------- Simulation Loop ------------------------------
def simulate(schema, simulation_id, deterministic=False, static=False, save_episode_agent=None, agent_filepath=None, recalculate_reward=None):
    ROOT = Path(__file__).resolve().parent.parent
    result_dir = ROOT / "results" / "envs"
    agent_dir = ROOT / "results" / "agents"
    log_dir = ROOT / "results" / "logs"
    table_dir = ROOT / "results" / "kpi_table"

    for d in [result_dir, agent_dir, log_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_filepath = log_dir / f"{simulation_id}.log"
    setup_logging(log_filepath)

    # env = CityLearnEnv(schema)
    env = SocFeasibleCityLearnEnv(schema)
    agents = env.load_agent() if not agent_filepath else pickle.load(open(agent_filepath, 'rb'))
    if deterministic:
        agents.deterministic_start_time_step = agents.time_step

    for episode in trange(env.schema['episodes'], desc="Simulating Episodes"):
        acc_reward = 0.0
        observations = env.reset()

        while not env.done:
            actions = agents.select_actions(observations)
            next_obs, reward, _, _ = env.step(actions)

            if recalculate_reward:
                socs = [b.observations['electrical_storage_soc'] for b in env.buildings]
                env.reward_function.kwargs['electrical_storage_soc'] = np.array(socs)
                reward = env.reward_function.calculate()

            acc_reward += reward[0]

            if not static:
                agents.add_to_buffer(observations, actions, reward, next_obs, done=env.done)

            observations = next_obs

            LOGGER.debug(f"Time step: {env.time_step}/{env.time_steps - 1}, Episode: {episode}/{env.schema['episodes'] - 1}, Actions: {actions}, Rewards: {reward}")

        metrics = env.evaluate()
        LOGGER.info(f"Episode {episode} cumulative reward: {acc_reward:.4f}")
        LOGGER.info(f"Metrics: {metrics}")

        print(f"\nEpisode {episode} Summary:")
        print(f"{'Metric':<30} {'Value':<10}")
        print("-" * 40)
        print(f"[DEBUG] type(metrics): {type(metrics)}, value: {metrics}")
        for key, value in metrics.items():
            print(f"{key:<30} {value:<10.9f}")

        write_kpis_to_table(metrics, episode, simulation_id, table_dir)

        with open(result_dir / f"{simulation_id}_episode_{episode}.pkl", 'wb') as f:
            pickle.dump(env, f)

        if save_episode_agent and ((episode + 1) % save_episode_agent == 0):
            with open(agent_dir / f"{simulation_id}_agent_episode_{episode}.pkl", 'wb') as f:
                pickle.dump(agents, f)

    # Final agent save
    with open(agent_dir / f"{simulation_id}_agent_final.pkl", 'wb') as f:
        pickle.dump(agents, f)

# -------------------------- KPI Writer ------------------------------
def write_kpis_to_table(kpis, episode, simulation_id, table_dir):
    filepath = table_dir / f"TABLE_{simulation_id}.txt"
    record = {'episode': episode}
    record.update(kpis)

    max_len = max(len(k) for k in record.keys())
    row_format = f"{{:<{max_len}}} | {{}}\n"

    with open(filepath, 'a') as file:
        file.write(row_format.format("KPI", "Value"))
        file.write("-" * (max_len + 10) + "\n")
        for key, value in record.items():
            file.write(row_format.format(key, value))

# ---------------------------- CLI Entrypoint ---------------------------
def main():
    parser = argparse.ArgumentParser(prog='citylearn_simulator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('schema', type=str)
    parser.add_argument('simulation_id', type=str)
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--save_episode_agent', type=int, default=None)
    parser.add_argument('--agent_filepath', type=str, default=None)
    parser.add_argument('--recalculate_reward', action='store_true')

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(simulate)
    kwargs = {k: v for k, v in vars(args).items() if k in arg_spec.args}
    print("Simulation Parameters:\n", kwargs)
    simulate(**kwargs)

if __name__ == '__main__':
    sys.exit(main())
