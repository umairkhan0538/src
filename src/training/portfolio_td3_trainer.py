"""
PortfolioTD3Trainer
===================

Drives the unified portfolio TD3 experiment:

    for cycle in 0..num_cycles-1:
        for building_id in building_order[cycle]:
            env = build_env(building_id)
            portfolio_td3.switch_building(env, building_id)
            obs = env.reset()
            for t in 0..episode_steps-1:
                a = portfolio_td3.select_actions(obs)   # advances portfolio_td3.time_step
                ns, r, done, _ = env.step(a)
                portfolio_td3.push_transition(
                    state=encoded_obs,
                    action=a[0],
                    reward=r[0],
                    next_state=encoded_ns,
                    done=bool(done),
                    building_id=building_id,
                    cycle=cycle,
                    episode_step=t,
                    global_step=portfolio_td3.time_step,
                )
                n_updates = portfolio_td3.learn()
                obs = ns

The TD3's existing `select_actions` path calls `add_to_buffer` internally,
which still pushes to the legacy 5-tuple buffer if we don't intervene. To
keep the persistence guarantee that ALL transitions land in the SHARED
portfolio buffer, we override the agent's `replay_buffer` to be the
`PortfolioReplayBuffer` instance BEFORE the first step. The legacy
`add_to_buffer` will then push to the portfolio buffer (with no metadata),
and we additionally call `push_transition` with metadata so that every
transition also carries source tags. We do both because:
  - the legacy `add_to_buffer` still works (gradient updates happen in it),
  - the portfolio buffer receives every transition with full metadata.

This avoids any risk that the legacy 5-tuple path silently swallows
transitions, and the metadata-rich `push_transition` is the audit trail.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

# Make sure src/ is importable when this file is run directly.
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from citylearn.citylearn import CityLearnEnv

from agents.portfolio_td3 import PortfolioTD3
from agents.portfolio_replay_buffer import PortfolioReplayBuffer


def build_env_for_building(schema_template: dict, building_id: int, data_root: str) -> CityLearnEnv:
    """Build a single-building env for the requested building id."""
    schema = copy.deepcopy(schema_template)
    schema['central_agent'] = True
    schema['root_directory'] = data_root
    schema['buildings'] = {k: v for k, v in schema['buildings'].items() if k == f'Building_{building_id}'}
    schema['agent'] = {
        'type': 'agents.td3.TD3',
        'attributes': dict(schema.get('drlagent_attributes', {})),
    }
    return CityLearnEnv(schema)


def make_building_order(medoids: List[int], num_cycles: int, order: str) -> List[List[int]]:
    """Return a list of length num_cycles; each entry is the list of buildings for that cycle."""
    if order == 'alternating':
        # A B A B ...
        a, b = medoids[0], medoids[1]
        return [[a, b] if c % 2 == 0 else [b, a] for c in range(num_cycles)]
    if order == 'B1B5':
        return [list(medoids) for _ in range(num_cycles)]
    if order == 'B5B1':
        return [list(reversed(medoids)) for _ in range(num_cycles)]
    raise ValueError(f"Unknown order: {order}")


def encode_obs(agent, building_index: int, raw_obs):
    """Return the encoded observation numpy array for a building index."""
    enc = agent.get_encoded_observations(building_index, list(raw_obs))
    return np.asarray(enc, dtype=float)


def train_unified(
    schema_template: dict,
    medoids: List[int],
    num_cycles: int,
    steps_per_building_episode: int,
    order: str,
    seed: int,
    output_dir: Path,
    tag: str,
    batch_size: int = 512,
    update_per_time_step: int = 2,
    init_steps_per_building: int = None,
    clear_buffer_at_building_switch: bool = False,
):
    """
    Train ONE persistent PortfolioTD3 on a medoid portfolio.

    The TD3 instance is created exactly once (on the first building's env),
    then re-pointed to subsequent buildings via `switch_building`.
    Networks, optimizers, replay buffer, and global timestep persist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{tag}_train.log"
    logger = _make_logger(log_path)

    # Seed everything for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_root = schema_template['root_directory']

    building_order = make_building_order(medoids, num_cycles, order)
    logger.info(f"Building order ({order}): {building_order[:5]}... (showing first 5 cycles)")
    logger.info(f"Total cycles: {num_cycles}, steps/building: {steps_per_building_episode}")

    # Build first env and create the agent
    first_building = medoids[0]
    env = build_env_for_building(schema_template, first_building, data_root)
    logger.info(f"Initial env: {first_building}, time_steps={env.time_steps}")

    agent_attributes = dict(schema_template['drlagent_attributes'])
    agent_attributes['seed'] = seed

    # CRITICAL: We must NOT use TD3(env=..., **attrs) directly because
    # the repo's TD3.__init__ calls super().__init__(*args, **kwargs) without
    # the 4 required positional args (observation_names, observation_space,
    # action_space, building_information) that Agent.__init__ needs in
    # citylearn 1.4.4. The canonical way to build a TD3 is via
    # env.load_agent(), which uses the correct 1.4.4 path.
    env.schema['agent'] = {'type': 'agents.td3.TD3', 'attributes': dict(agent_attributes)}
    base_agent = env.load_agent()
    from agents.portfolio_td3 import PortfolioTD3
    agent = PortfolioTD3.from_base_td3(base_agent, building_id=first_building)
    logger.info(f"Agent created: time_step={agent.time_step}, obs_dim={agent.observation_dimension[0]}")

    # Snapshot network/optimizer/replay buffer IDs
    initial_ids = _snapshot_ids(agent)
    logger.info(f"Initial IDs: {initial_ids}")

    # Optionally clear the portfolio buffer (not used for primary Contribution 2)
    if clear_buffer_at_building_switch:
        agent.replay_buffer = PortfolioReplayBuffer(agent.replay_buffer_capacity)

    # Init steps for smoke tests (override the default 8760)
    if init_steps_per_building is None:
        init_steps_per_building = steps_per_building_episode

    total_steps = 0
    composition_log_path = output_dir / f"{tag}_replay_composition.csv"
    composition_rows: List[dict] = []
    _write_composition_header(composition_log_path)

    for cycle, buildings in enumerate(building_order):
        for building_id in buildings:
            logger.info(f"=== Cycle {cycle} | Building {building_id} | t={agent.time_step} ===")

            # Verify network persistence at the boundary
            cur_ids = _snapshot_ids(agent)
            for k in initial_ids:
                if cur_ids[k] != initial_ids[k]:
                    msg = f"IDENTITY LOSS at {k}: was {initial_ids[k]} now {cur_ids[k]}"
                    logger.error(msg)
                    raise RuntimeError(msg)

            # If building is new to the env chain, re-point the agent
            if agent.current_building_id != building_id:
                env.close() if hasattr(env, "close") else None
                env = build_env_for_building(schema_template, building_id, data_root)
                agent.switch_building(env, building_id)
                logger.info(f"Switched to building {building_id}; buffer now has {len(agent.replay_buffer)} transitions")
                if clear_buffer_at_building_switch:
                    agent.replay_buffer = PortfolioReplayBuffer(agent.replay_buffer_capacity)

            obs = env.reset()
            # The PortfolioTD3's switch_building already advances time_step via env.reset?
            # No -- env.reset() resets env.time_step, NOT agent.time_step. The agent's
            # time_step is only advanced by select_actions. So no reset of agent.time_step
            # occurs here, which is exactly what we want.

            for t in range(init_steps_per_building):
                # 1) Select actions -- this also advances agent.time_step
                actions = agent.select_actions(obs)
                # 2) Step env
                next_obs, reward, done, info = env.step(actions)
                # 3) Push transition with metadata (encoded obs).
                #    This is the ONLY push path. The portfolio buffer is the
                #    single source of truth.
                building_idx = 0  # central agent with one building
                s_enc = encode_obs(agent, building_idx, obs[0])
                ns_enc = encode_obs(agent, building_idx, next_obs[0])
                agent.push_transition(
                    state=s_enc,
                    action=np.asarray(actions[0], dtype=float),
                    reward=float(reward[0]),
                    next_state=ns_enc,
                    done=bool(done),
                    building_id=building_id,
                    cycle=cycle,
                    episode_step=t,
                    global_step=agent.time_step,
                )
                # 4) Run the verbatim TD3 gradient update. The portfolio
                #    buffer.sample() returns the 5-tuple in the exact format
                #    TD3 expects, so the existing update math is preserved.
                agent.learn()
                obs = next_obs
                total_steps += 1

                if total_steps % 1000 == 0 or t == init_steps_per_building - 1:
                    summary = agent.replay_buffer.summary()
                    logger.info(
                        f"  step {total_steps} | t={t} | bid={building_id} | "
                        f"buffer={summary['total_transitions']}/{summary['capacity']} | "
                        f"B1={summary['source_counts'].get(1,0)} B5={summary['source_counts'].get(5,0)}"
                    )
                    _append_composition_row(composition_log_path, {
                        "global_step": total_steps,
                        "td3_time_step": agent.time_step,
                        "cycle": cycle,
                        "building_id": building_id,
                        "episode_step": t,
                        "replay_size": summary["total_transitions"],
                        "b1_count": summary["source_counts"].get(1, 0),
                        "b5_count": summary["source_counts"].get(5, 0),
                        "b1_fraction": summary["source_fractions"].get(1, 0.0),
                        "b5_fraction": summary["source_fractions"].get(5, 0.0),
                        "param_version": agent._param_version,
                        "actor_id": initial_ids["actor"],
                        "c1_id": initial_ids["c1"],
                        "c2_id": initial_ids["c2"],
                        "ta_id": initial_ids["ta"],
                        "tc1_id": initial_ids["tc1"],
                        "tc2_id": initial_ids["tc2"],
                        "ao_id": initial_ids["ao"],
                        "co1_id": initial_ids["co1"],
                        "co2_id": initial_ids["co2"],
                        "rb_id": initial_ids["rb"],
                    })

            if done:
                obs = env.reset()

    # Save final agent (PortfolioTD3 includes PortfolioReplayBuffer)
    final_path = output_dir / f"{tag}_agent_final.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(agent, f)
    logger.info(f"Final agent saved to {final_path}")

    return agent, final_path


# ----------------------------------------------------------------- utilities

def _make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"portfolio_td3_trainer_{log_path.name}")
    logger.setLevel(logging.DEBUG)
    # Remove pre-existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    h = logging.FileHandler(log_path, mode='w')
    h.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logger.addHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    return logger


def _snapshot_ids(agent) -> dict:
    return {
        'actor': id(agent.actor),
        'c1': id(agent.critic_1),
        'c2': id(agent.critic_2),
        'ta': id(agent.target_actor),
        'tc1': id(agent.target_critic_1),
        'tc2': id(agent.target_critic_2),
        'ao': id(agent.actor_optimizer),
        'co1': id(agent.critic_1_optimizer),
        'co2': id(agent.critic_2_optimizer),
        'rb': id(agent.replay_buffer),
    }


def _write_composition_header(path: Path) -> None:
    fieldnames = [
        "global_step", "td3_time_step", "cycle", "building_id", "episode_step",
        "replay_size", "b1_count", "b5_count", "b1_fraction", "b5_fraction",
        "param_version",
        "actor_id", "c1_id", "c2_id", "ta_id", "tc1_id", "tc2_id",
        "ao_id", "co1_id", "co2_id", "rb_id",
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def _append_composition_row(path: Path, row: dict) -> None:
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writerow(row)
