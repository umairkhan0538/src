"""
Zero-shot evaluation harness.

Loads a pickled PortfolioTD3 (or compatible TD3) agent and evaluates its
frozen actor on a list of target buildings, with no gradient updates and
no replay insertion.

Computes CityLearn cost-function KPIs via `env.evaluate()` and writes
per-building results to a CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from citylearn.citylearn import CityLearnEnv


def make_eval_schema(schema_template: dict, building_id: int, data_root: str, agent_attributes: dict) -> dict:
    """Build a single-building evaluation schema that constructs a vanilla RBC agent (we will swap the agent in)."""
    schema = copy.deepcopy(schema_template)
    schema['central_agent'] = True
    schema['root_directory'] = data_root
    schema['buildings'] = {k: v for k, v in schema['buildings'].items() if k == f'Building_{building_id}'}
    schema['agent'] = {
        'type': 'agents.td3.TD3',
        'attributes': dict(agent_attributes),
    }
    return schema


def evaluate_agent_on_building(
    agent,
    building_id: int,
    schema_template: dict,
    data_root: str,
    deterministic: bool = True,
):
    """
    Run the frozen agent on a single-building env, returning cumulative reward
    and per-step reward trace.
    """
    agent_attributes = {
        'hidden_dimension': agent.hidden_dimension,
        'discount': agent.discount,
        'tau': agent.tau,
        'lr_c': agent.lr_c,
        'lr_a': agent.lr_a,
        'alpha': agent.alpha,
        'batch_size': agent.batch_size,
        'replay_buffer_capacity': agent.replay_buffer_capacity,
        'start_training_time_step': agent.start_training_time_step,
        'end_exploration_time_step': agent.end_exploration_time_step,
        'deterministic_start_time_step': agent.deterministic_start_time_step,
        'action_scaling_coef': agent.action_scaling_coefficient,
        'reward_scaling': agent.reward_scaling,
        'update_per_time_step': agent.update_per_time_step,
        'seed': getattr(agent, 'seed', 0),
    }
    # Build a fresh env for this building
    env = CityLearnEnv(make_eval_schema(schema_template, building_id, data_root, agent_attributes))
    # Re-point the agent to the new env WITHOUT recreating networks
    if hasattr(agent, 'switch_building'):
        agent.switch_building(env, building_id)
    else:
        # Plain TD3
        agent.observation_names = env.observation_names
        agent.observation_space = env.observation_space
        agent.action_space = env.action_space
        agent.building_information = env.unwrapped.get_building_information()
        agent.encoders = agent.set_encoders()
    # Freeze the actor
    if hasattr(agent, 'freeze'):
        agent.freeze()
    for p in agent.actor.parameters():
        p.requires_grad = False
    agent.actor.eval()

    # Force deterministic policy (post-exploration path)
    saved_end = agent.end_exploration_time_step
    if deterministic:
        agent.end_exploration_time_step = 0

    obs = env.reset()
    cumulative_reward = 0.0
    reward_trace = []
    while not env.done:
        actions = agent.select_actions(obs)
        next_obs, reward, done, info = env.step(actions)
        cumulative_reward += float(reward[0])
        reward_trace.append(float(reward[0]))
        obs = next_obs

    # Restore exploration cutoff
    agent.end_exploration_time_step = saved_end
    # Unfreeze
    for p in agent.actor.parameters():
        p.requires_grad = True
    agent.actor.train()

    # CityLearn KPIs
    kpis = env.evaluate()  # returns price, emission, grid ratios
    # Detailed per-building timeseries for cost/emission/load
    # net_electricity_consumption is on env (cumulative list)
    return {
        'building_id': building_id,
        'cumulative_reward': float(cumulative_reward),
        'mean_reward': float(np.mean(reward_trace)) if reward_trace else 0.0,
        'std_reward': float(np.std(reward_trace)) if reward_trace else 0.0,
        'kpis': {  # kpis is a tuple of (price, emission, grid) ratios
            'price_cost_ratio': float(kpis[0]),
            'emission_cost_ratio': float(kpis[1]),
            'grid_cost_ratio': float(kpis[2]),
        },
        'net_electricity_consumption_total': float(np.sum(env.net_electricity_consumption)),
        'net_electricity_consumption_cost_total': float(np.sum(env.net_electricity_consumption_price)),
        'net_electricity_consumption_emission_total': float(np.sum(env.net_electricity_consumption_emission)),
        'reward_trace_length': len(reward_trace),
    }


def run_zero_shot(
    agent_pickle_path: str,
    target_buildings: List[int],
    schema_template_path: str,
    output_csv_path: str,
    data_root: str,
):
    """Load agent, evaluate on each target building, write CSV."""
    with open(agent_pickle_path, 'rb') as f:
        agent = pickle.load(f)
    with open(schema_template_path) as f:
        schema_template = json.load(f)

    rows = []
    for bid in target_buildings:
        print(f"Evaluating on building {bid} ...")
        res = evaluate_agent_on_building(agent, bid, schema_template, data_root)
        rows.append(res)
        print(f"  cumulative_reward={res['cumulative_reward']:.4f}  price={res['kpis']['price_cost_ratio']:.4f}  emission={res['kpis']['emission_cost_ratio']:.4f}  grid={res['kpis']['grid_cost_ratio']:.4f}")

    # Write CSV
    if rows:
        keys = ['building_id', 'cumulative_reward', 'mean_reward', 'std_reward',
                'kpis.price_cost_ratio', 'kpis.emission_cost_ratio', 'kpis.grid_cost_ratio',
                'net_electricity_consumption_total',
                'net_electricity_consumption_cost_total',
                'net_electricity_consumption_emission_total',
                'reward_trace_length']
        with open(output_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in rows:
                w.writerow([
                    r['building_id'],
                    f"{r['cumulative_reward']:.6f}",
                    f"{r['mean_reward']:.6f}",
                    f"{r['std_reward']:.6f}",
                    f"{r['kpis']['price_cost_ratio']:.6f}",
                    f"{r['kpis']['emission_cost_ratio']:.6f}",
                    f"{r['kpis']['grid_cost_ratio']:.6f}",
                    f"{r['net_electricity_consumption_total']:.6f}",
                    f"{r['net_electricity_consumption_cost_total']:.6f}",
                    f"{r['net_electricity_consumption_emission_total']:.6f}",
                    r['reward_trace_length'],
                ])
        print(f"Results written to {output_csv_path}")
    return rows
