"""
Run all 6 ablations for Contribution 2 with a fixed training budget, then
zero-shot evaluation on B12 and B15 for each.

All runs use:
  - 2 cycles × 4000 steps/building = 16000 env steps total
  - Production hyperparameters from settings.json:
      batch_size=512
      start_training_time_step=3671
      end_exploration_time_step=3671
      discount=0.95, tau=0.002, lr_c=0.0003, lr_a=0.0001
      policy_noise=0.05, noise_clip=0.15, policy_freq=3
      hidden=[512,512], action_scaling=0.5, replay_capacity=1000000
  - Seed 101

The 6 ablations are:
  A. B1-only           (medoids=[1])
  B. B5-only           (medoids=[5])
  C. Sequential B1->B5 with CLEARED buffer at switch (isolates parameter transfer)
  D. Sequential B5->B1 with CLEARED buffer at switch
  E. Unified shared replay, B1-first alternating (the primary Contribution 2)
  F. Unified shared replay, B5-first alternating (order sensitivity)
"""

from __future__ import annotations

import json
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from training.run_experiment import make_full_schema_template, DATA_ROOT
from training.portfolio_td3_trainer import train_unified
from evaluation.zero_shot import run_zero_shot


ABLATIONS = [
    ('A_b1_only',          {'medoids': [1],   'order': 'B1B5',     'clear_buffer_at_building_switch': False, 'num_cycles': 4, 'init_steps_per_building': 4000}),
    ('B_b5_only',          {'medoids': [5],   'order': 'B1B5',     'clear_buffer_at_building_switch': False, 'num_cycles': 4, 'init_steps_per_building': 4000}),
    ('C_seq_b1b5',         {'medoids': [1,5], 'order': 'B1B5',     'clear_buffer_at_building_switch': True,  'num_cycles': 2, 'init_steps_per_building': 8000}),
    ('D_seq_b5b1',         {'medoids': [5,1], 'order': 'B5B1',     'clear_buffer_at_building_switch': True,  'num_cycles': 2, 'init_steps_per_building': 8000}),
    ('E_unified_alt',      {'medoids': [1,5], 'order': 'alternating','clear_buffer_at_building_switch': False, 'num_cycles': 2, 'init_steps_per_building': 8000}),
    ('F_unified_alt_b5',   {'medoids': [5,1], 'order': 'alternating','clear_buffer_at_building_switch': False, 'num_cycles': 2, 'init_steps_per_building': 8000}),
]


def main():
    out_root = Path('/home/user/src/src/results_ablations')
    out_root.mkdir(parents=True, exist_ok=True)

    schema = make_full_schema_template()
    schema['drlagent_attributes']['seed'] = 101
    # PRODUCTION hyperparameters
    schema['drlagent_attributes']['batch_size'] = 512
    schema['drlagent_attributes']['start_training_time_step'] = 3671
    schema['drlagent_attributes']['end_exploration_time_step'] = 3671
    schema['drlagent_attributes']['deterministic_start_time_step'] = 3671

    with open(out_root / 'schema_template.json', 'w') as f:
        json.dump(schema, f, indent=2)

    all_results = []
    grand_t0 = time.time()
    for tag, params in ABLATIONS:
        print(f'\n{"="*70}\n=== Ablation {tag}  params={params}\n{"="*70}')
        t0 = time.time()
        agent, pkl = train_unified(
            schema_template=schema,
            medoids=params['medoids'],
            num_cycles=params['num_cycles'],
            steps_per_building_episode=params.get('steps_per_building_episode', 4000),
            order=params['order'],
            seed=101,
            output_dir=out_root,
            tag=tag,
            init_steps_per_building=params['init_steps_per_building'],
            clear_buffer_at_building_switch=params.get('clear_buffer_at_building_switch', False),
        )
        train_time = time.time() - t0
        print(f'  Training time: {train_time:.1f}s')
        print(f'  Buffer: {len(agent.replay_buffer)}')
        print(f'  Source counts: {agent.replay_buffer.source_counts()}')
        print(f'  param_version: {agent._param_version}')

        # Zero-shot eval on B12, B15
        eval_path = out_root / f'{tag}_eval.csv'
        rows = run_zero_shot(
            agent_pickle_path=str(pkl),
            target_buildings=[12, 15],
            schema_template_path=str(out_root / 'schema_template.json'),
            output_csv_path=str(eval_path),
            data_root=DATA_ROOT,
        )
        for r in rows:
            print(f'  B{r["building_id"]}: cum_r={r["cumulative_reward"]:.4f}  '
                  f'price={r["kpis"]["price_cost_ratio"]:.4f}  '
                  f'emission={r["kpis"]["emission_cost_ratio"]:.4f}  '
                  f'grid={r["kpis"]["grid_cost_ratio"]:.4f}  '
                  f'net_elec={r["net_electricity_consumption_total"]:.2f}')
            all_results.append({
                'ablation': tag,
                **params,
                'train_time_s': train_time,
                'param_version': agent._param_version,
                'buffer_size': len(agent.replay_buffer),
                'b1_count': agent.replay_buffer.source_counts().get(1, 0),
                'b5_count': agent.replay_buffer.source_counts().get(5, 0),
                'b12_count': agent.replay_buffer.source_counts().get(12, 0),
                'b15_count': agent.replay_buffer.source_counts().get(15, 0),
                **r,
            })

        # Save running results
        with open(out_root / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'  Results updated in {out_root / "all_results.json"}')

    grand_total = time.time() - grand_t0
    print(f'\n{"="*70}\nALL ABLATIONS COMPLETE in {grand_total/60:.1f} min\n{"="*70}')

    # Final summary table
    print(f'\n{"Ablation":<20s} {"Target":<8s} {"Cum reward":>12s} {"Price":>8s} {"Emission":>10s} {"Grid":>8s} {"NetElec":>10s}')
    for r in all_results:
        print(f'{r["ablation"]:<20s} B{r["building_id"]:<7d} {r["cumulative_reward"]:>12.4f} '
              f'{r["kpis"]["price_cost_ratio"]:>8.4f} {r["kpis"]["emission_cost_ratio"]:>10.4f} '
              f'{r["kpis"]["grid_cost_ratio"]:>8.4f} {r["net_electricity_consumption_total"]:>10.2f}')


if __name__ == '__main__':
    main()
