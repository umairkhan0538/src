"""
Schema builder, experiment runner, and smoke test for Contribution 2.

Usage:
    python -m training.run_experiment smoke        # tiny diagnostic
    python -m training.run_experiment unified      # full 525,600-step run
    python -m training.run_experiment b1_only
    python -m training.run_experiment b5_only
    python -m training.run_experiment seq_b1b5      # sequential B1->B5 (clear buffer at switch)
    python -m training.run_experiment seq_b5b1      # sequential B5->B1 (clear buffer at switch)
    python -m training.run_experiment unified_b5first
    python -m training.run_experiment eval_only    # zero-shot eval only
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from citylearn.citylearn import CityLearnEnv

from training.portfolio_td3_trainer import (
    train_unified, build_env_for_building, make_building_order, encode_obs
)
from evaluation.zero_shot import run_zero_shot

DATA_ROOT = '/usr/local/lib/python3.11/dist-packages/citylearn/data/citylearn_challenge_2022_phase_all'


def make_full_schema_template() -> dict:
    """Load the bundled 2022 schema, set root_directory, central_agent=True, include ALL 17 buildings (the trainer will pick one building per env)."""
    with open(f'{DATA_ROOT}/schema.json') as f:
        schema = json.load(f)
    schema['root_directory'] = DATA_ROOT
    schema['central_agent'] = True
    # DO NOT restrict include_buildings here; the trainer builds per-building envs.
    # Copy the verified TD3 hyperparameters (from settings.json) into the schema so
    # the trainer can use them as defaults.
    schema['drlagent_attributes'] = {
        'hidden_dimension': [512, 512],
        'discount': 0.95,
        'tau': 0.002,
        'lr_c': 0.0003,
        'lr_a': 0.0001,
        'alpha': 0.5,
        'batch_size': 512,
        'replay_buffer_capacity': 1000000,
        'start_training_time_step': 3671,
        'end_exploration_time_step': 3671,
        'deterministic_start_time_step': 36710,
        'action_scaling_coef': 0.5,
        'reward_scaling': 1,
        'update_per_time_step': 2,
        'policy_noise': 0.05,
        'noise_clip': 0.15,
        'policy_freq': 3,
        'seed': 101,
    }
    return schema


def run_smoke():
    """Tiny diagnostic: 50 steps on B1, 50 steps on B5, then zero-shot on B12, B15."""
    schema = make_full_schema_template()
    schema['drlagent_attributes']['batch_size'] = 16  # smaller for the tiny buffer
    schema['drlagent_attributes']['start_training_time_step'] = 5
    schema['drlagent_attributes']['end_exploration_time_step'] = 5
    schema['drlagent_attributes']['deterministic_start_time_step'] = 5
    schema['drlagent_attributes']['seed'] = 42

    output_dir = Path('/tmp/contrib2_smoke')
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = 'smoke'
    print('--- SMOKE: unified B1->B5 alternating, 50 steps each, seed 42 ---')
    agent, pkl_path = train_unified(
        schema_template=schema,
        medoids=[1, 5],
        num_cycles=1,                # tiny diagnostic: 1 cycle = 2 episodes
        steps_per_building_episode=50,
        order='alternating',
        seed=42,
        output_dir=output_dir,
        tag=tag,
        init_steps_per_building=50,
    )

    print('\n--- After training, replay summary ---')
    print(json.dumps(agent.replay_buffer.summary(), indent=2, default=str))

    # Verify TD3 identity
    print('\n--- TD3 identity check ---')
    for k, v in [
        ('actor', agent.actor),
        ('c1', agent.critic_1), ('c2', agent.critic_2),
        ('ta', agent.target_actor),
        ('tc1', agent.target_critic_1), ('tc2', agent.target_critic_2),
        ('ao', agent.actor_optimizer),
        ('co1', agent.critic_1_optimizer), ('co2', agent.critic_2_optimizer),
        ('rb', agent.replay_buffer),
    ]:
        print(f'  {k:6s}  id={id(v)}')

    # Test mixed-batch sampling
    print('\n--- Mixed batch sampling test ---')
    s, a, r, ns, d, meta = agent.replay_buffer.sample_with_meta(8)
    print(f'  sample shape: state={s.shape}, action={a.shape}, meta={meta.shape}')
    print(f'  meta columns (building_id, cycle, episode_step, global_step): {meta.tolist()}')
    unique_bids = np.unique(meta[:, 0])
    print(f'  unique building_ids in batch: {unique_bids.tolist()}')
    if len(unique_bids) >= 2:
        print('  PASS: mixed batch contains BOTH B1 and B5')
    else:
        print('  WARN: batch contains only one source; expected both')

    # Zero-shot eval on B12 and B15
    print('\n--- Zero-shot evaluation on B12, B15 ---')
    rows = run_zero_shot(
        agent_pickle_path=str(pkl_path),
        target_buildings=[12, 15],
        schema_template_path=str(_save_schema_template(schema, output_dir)),
        output_csv_path=str(output_dir / f"{tag}_eval.csv"),
        data_root=DATA_ROOT,
    )
    print('Zero-shot rows:')
    for r in rows:
        print(f'  B{r["building_id"]}: cum_r={r["cumulative_reward"]:.4f}  '
              f'price={r["kpis"]["price_cost_ratio"]:.4f}  emission={r["kpis"]["emission_cost_ratio"]:.4f}  '
              f'grid={r["kpis"]["grid_cost_ratio"]:.4f}')

    # Compare against baseline (B1 only, same 50 steps)
    print('\n--- Baseline: B1 only, 100 steps ---')
    agent_b1, pkl_b1 = train_unified(
        schema_template=schema,
        medoids=[1],
        num_cycles=1,
        steps_per_building_episode=100,
        order='B1B5',
        seed=42,
        output_dir=output_dir,
        tag='b1_only_smoke',
        init_steps_per_building=100,
    )
    print('B1-only replay:', agent_b1.replay_buffer.summary()['source_counts'])

    # Check pickle round-trip
    print('\n--- Pickle round-trip test ---')
    import pickle
    p = output_dir / f"{tag}_agent_final.pkl"
    with open(p, 'rb') as f:
        loaded = pickle.load(f)
    print(f'  loaded time_step={loaded.time_step}, '
          f'buffer_size={len(loaded.replay_buffer)}, '
          f'actor_id same={id(loaded.actor) == id(agent.actor)}')

    return agent


def _save_schema_template(schema, output_dir):
    p = output_dir / 'schema_template.json'
    with open(p, 'w') as f:
        json.dump(schema, f, indent=2)
    return p


def run_full(args):
    """Run the full 525,600-step unified experiment."""
    schema = make_full_schema_template()
    schema['drlagent_attributes']['seed'] = args.seed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == 'unified':
        agent, pkl = train_unified(
            schema_template=schema,
            medoids=[1, 5],
            num_cycles=args.cycles,
            steps_per_building_episode=args.steps,
            order='alternating',
            seed=args.seed,
            output_dir=output_dir,
            tag=f"unified_seed{args.seed}",
        )
    elif args.method == 'unified_b5first':
        agent, pkl = train_unified(
            schema_template=schema,
            medoids=[5, 1],
            num_cycles=args.cycles,
            steps_per_building_episode=args.steps,
            order='alternating',
            seed=args.seed,
            output_dir=output_dir,
            tag=f"unified_b5first_seed{args.seed}",
        )
    elif args.method == 'b1_only':
        agent, pkl = train_unified(
            schema_template=schema,
            medoids=[1],
            num_cycles=args.cycles,
            steps_per_building_episode=args.steps,
            order='B1B5',
            seed=args.seed,
            output_dir=output_dir,
            tag=f"b1only_seed{args.seed}",
        )
    elif args.method == 'b5_only':
        agent, pkl = train_unified(
            schema_template=schema,
            medoids=[5],
            num_cycles=args.cycles,
            steps_per_building_episode=args.steps,
            order='B1B5',
            seed=args.seed,
            output_dir=output_dir,
            tag=f"b5only_seed{args.seed}",
        )
    elif args.method == 'seq_b1b5':
        # Sequential: B1 first for half the budget, then B5, with the BUFFER
        # CLEARED at the switch (so this isolates parameter transfer, not shared replay).
        agent, pkl = train_unified(
            schema_template=schema,
            medoids=[1, 5],
            num_cycles=args.cycles,
            steps_per_building_episode=args.steps,
            order='B1B5',  # all B1 then all B5
            seed=args.seed,
            output_dir=output_dir,
            tag=f"seq_b1b5_seed{args.seed}",
            clear_buffer_at_building_switch=True,
        )
    elif args.method == 'seq_b5b1':
        agent, pkl = train_unified(
            schema_template=schema,
            medoids=[5, 1],
            num_cycles=args.cycles,
            steps_per_building_episode=args.steps,
            order='B5B1',
            seed=args.seed,
            output_dir=output_dir,
            tag=f"seq_b5b1_seed{args.seed}",
            clear_buffer_at_building_switch=True,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save schema template
    _save_schema_template(schema, output_dir)

    # Zero-shot eval
    rows = run_zero_shot(
        agent_pickle_path=str(pkl),
        target_buildings=[12, 15],
        schema_template_path=str(output_dir / 'schema_template.json'),
        output_csv_path=str(output_dir / f"{args.method}_seed{args.seed}_eval.csv"),
        data_root=DATA_ROOT,
    )
    return agent, rows


def run_eval_only(args):
    """Load a pre-trained agent pickle and run zero-shot on B12 and B15."""
    rows = run_zero_shot(
        agent_pickle_path=args.agent_pickle,
        target_buildings=[12, 15],
        schema_template_path=args.schema_template,
        output_csv_path=args.eval_output,
        data_root=DATA_ROOT,
    )
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_smoke = sub.add_parser('smoke', help='Tiny diagnostic: 50 steps B1 + 50 steps B5')

    p_full = sub.add_parser('full', help='Full training run')
    p_full.add_argument('--method', choices=['unified', 'unified_b5first', 'b1_only', 'b5_only', 'seq_b1b5', 'seq_b5b1'], required=True)
    p_full.add_argument('--cycles', type=int, default=30)
    p_full.add_argument('--steps', type=int, default=8760)
    p_full.add_argument('--seed', type=int, default=101)
    p_full.add_argument('--output-dir', type=str, default='/tmp/contrib2_full')

    p_eval = sub.add_parser('eval_only', help='Run zero-shot eval on a pre-trained agent')
    p_eval.add_argument('--agent-pickle', type=str, required=True)
    p_eval.add_argument('--schema-template', type=str, required=True)
    p_eval.add_argument('--eval-output', type=str, required=True)

    args = parser.parse_args()
    if args.cmd == 'smoke':
        run_smoke()
    elif args.cmd == 'full':
        run_full(args)
    elif args.cmd == 'eval_only':
        run_eval_only(args)
