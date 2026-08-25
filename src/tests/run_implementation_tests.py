"""
Implementation tests for Contribution 2.

Tests 1-15 (per spec):
 1. B1 transitions enter the shared replay.
 2. B5 transitions enter the SAME replay.
 3. B1 transitions remain after B5 starts.
 4. actor object persists.
 5. critic objects persist.
 6. target networks persist.
 7. optimizers persist.
 8. replay buffer persists.
 9. TD3 global timestep continues monotonically.
10. B1 and B5 observation/action shapes are compatible.
11. mixed replay sampling works.
12. B1-first works.
13. B5-first works.
14. final actor can be frozen.
15. frozen actor can perform zero-shot inference.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from training.run_experiment import make_full_schema_template, run_smoke, DATA_ROOT
from training.portfolio_td3_trainer import (
    train_unified, build_env_for_building, encode_obs
)
from agents.portfolio_td3 import PortfolioTD3
from agents.portfolio_replay_buffer import PortfolioReplayBuffer

PASS = '\033[32mPASS\033[0m'
FAIL = '\033[31mFAIL\033[0m'
results = []

def report(name, ok, detail=''):
    tag = PASS if ok else FAIL
    print(f'  [{tag}] {name}  {detail}')
    results.append((name, ok, detail))


def make_smoke_schema():
    schema = make_full_schema_template()
    schema['drlagent_attributes']['batch_size'] = 16
    schema['drlagent_attributes']['start_training_time_step'] = 5
    schema['drlagent_attributes']['end_exploration_time_step'] = 5
    schema['drlagent_attributes']['deterministic_start_time_step'] = 5
    schema['drlagent_attributes']['seed'] = 1234
    return schema


def test_1_to_15():
    schema = make_smoke_schema()
    output_dir = Path('/tmp/contrib2_tests')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 12: B1-first alternating (run 60 steps on B1 then 60 on B5)
    print('--- Test 12: B1-first alternating ---')
    agent, pkl = train_unified(
        schema_template=schema,
        medoids=[1, 5],
        num_cycles=1,
        steps_per_building_episode=60,
        order='alternating',
        seed=1234,
        output_dir=output_dir,
        tag='test_b1first',
        init_steps_per_building=60,
    )

    summary = agent.replay_buffer.summary()
    # Test 1: B1 transitions in shared replay
    report('T1: B1 transitions in shared replay',
           summary['source_counts'].get(1, 0) > 0,
           f"B1={summary['source_counts'].get(1, 0)}")

    # Test 2: B5 transitions in SAME replay
    report('T2: B5 transitions in same replay',
           summary['source_counts'].get(5, 0) > 0,
           f"B5={summary['source_counts'].get(5, 0)}")

    # Test 3: B1 transitions remain after B5 starts
    report('T3: B1 transitions remain after B5 starts',
           summary['source_counts'].get(1, 0) == 60 and summary['source_counts'].get(5, 0) == 60,
           f"B1={summary['source_counts'].get(1, 0)}, B5={summary['source_counts'].get(5, 0)}")

    # Test 4-8: object persistence across building switch WITHIN one run.
    # We can't easily capture IDs mid-run from the train_unified return value,
    # so we manually drive a 2-building run and check IDs at the boundary.
    schema_p = make_smoke_schema()
    env_p1 = build_env_for_building(schema_p, 1, DATA_ROOT)
    env_p1.schema['agent']['attributes'] = dict(schema_p['drlagent_attributes'])
    env_p1.schema['agent']['attributes']['seed'] = 1234
    base_p = env_p1.load_agent()
    agent_p = PortfolioTD3.from_base_td3(base_p, building_id=1)
    pre_ids = {k: id(getattr(agent_p, k)) for k in
               ['actor','critic_1','critic_2','target_actor','target_critic_1','target_critic_2',
                'actor_optimizer','critic_1_optimizer','critic_2_optimizer','replay_buffer']}
    # Run 5 steps on B1
    obs = env_p1.reset()
    for _ in range(5):
        a = agent_p.select_actions(obs)
        ns, r, d, i = env_p1.step(a)
        agent_p.push_transition(encode_obs(agent_p,0,obs[0]), a[0], r[0], encode_obs(agent_p,0,ns[0]), d, 1, 0, 0, agent_p.time_step)
        obs = ns
    # Switch to B5 env
    env_p5 = build_env_for_building(schema_p, 5, DATA_ROOT)
    agent_p.switch_building(env_p5, 5)
    obs = env_p5.reset()
    for _ in range(5):
        a = agent_p.select_actions(obs)
        ns, r, d, i = env_p5.step(a)
        agent_p.push_transition(encode_obs(agent_p,0,obs[0]), a[0], r[0], encode_obs(agent_p,0,ns[0]), d, 5, 0, 0, agent_p.time_step)
        obs = ns
    post_ids = {k: id(getattr(agent_p, k)) for k in pre_ids}
    same = all(pre_ids[k] == post_ids[k] for k in pre_ids)
    report('T4-T8: actor/critics/targets/optimizers/replay persist across B1->B5',
           same,
           'all 10 IDs identical' if same else f'DIFF: {[(k, pre_ids[k], post_ids[k]) for k in pre_ids if pre_ids[k] != post_ids[k]]}')

    # Test 9: TD3 time_step continues monotonically
    expected_ts = 60 + 60  # 60 B1 + 60 B5
    report('T9: TD3 time_step continues monotonically',
           agent.time_step == expected_ts,
           f'time_step={agent.time_step} (expected {expected_ts})')

    # Test 10: B1 and B5 obs/action shapes are compatible
    env1 = build_env_for_building(schema, 1, DATA_ROOT)
    env5 = build_env_for_building(schema, 5, DATA_ROOT)
    obs_shape_match = env1.observation_space[0].shape == env5.observation_space[0].shape
    act_shape_match = env1.action_space[0].shape == env5.action_space[0].shape
    report('T10: B1 and B5 observation/action shapes are compatible',
           obs_shape_match and act_shape_match,
           f'obs={env1.observation_space[0].shape} act={env1.action_space[0].shape}')

    # Test 11: mixed replay sampling
    s, a, r, ns, d, meta = agent.replay_buffer.sample_with_meta(16)
    unique_bids = np.unique(meta[:, 0])
    report('T11: mixed replay sampling works',
           s.shape[0] == 16 and len(unique_bids) >= 2,
           f'sample_shape={s.shape} unique_bids={unique_bids.tolist()}')

    # Test 14: final actor can be frozen
    try:
        agent.freeze()
        frozen = all(not p.requires_grad for p in agent.actor.parameters()) and not agent.actor.training
        report('T14: final actor can be frozen', frozen, 'requires_grad=False for all params')
    except Exception as e:
        report('T14: final actor can be frozen', False, str(e))

    # Test 15: frozen actor can perform zero-shot inference
    # Re-point the existing agent to a B5 env (it was on B5 at end of B1-first test).
    # We make a fresh copy to avoid perturbing the original.
    import pickle
    agent.freeze()
    env_test = build_env_for_building(schema, 5, DATA_ROOT)
    agent.switch_building(env_test, 5)
    obs = env_test.reset()
    n_steps = 10
    for _ in range(n_steps):
        actions = agent.select_actions(obs)
        next_obs, r, done, info = env_test.step(actions)
        obs = next_obs
    report('T15: frozen actor can perform zero-shot inference',
           env_test.time_step >= n_steps,
           f'env.time_step={env_test.time_step}')

    # Test 13: B5-first works
    print('\n--- Test 13: B5-first alternating ---')
    schema_b5 = make_smoke_schema()
    agent_b5, pkl_b5 = train_unified(
        schema_template=schema_b5, medoids=[5, 1], num_cycles=1,
        steps_per_building_episode=60, order='alternating', seed=1234,
        output_dir=output_dir, tag='test_b5first', init_steps_per_building=60,
    )
    summary_b5 = agent_b5.replay_buffer.summary()
    report('T13: B5-first works',
           summary_b5['source_counts'].get(5, 0) == 60 and summary_b5['source_counts'].get(1, 0) == 60,
           f"B5={summary_b5['source_counts'].get(5, 0)}, B1={summary_b5['source_counts'].get(1, 0)}")

    # Test 3 again on b5-first run: B5 transitions should remain after B1
    report('T3b: B5 transitions remain after B1 starts (B5-first)',
           summary_b5['source_counts'].get(5, 0) == 60,
           f"B5={summary_b5['source_counts'].get(5, 0)}")

    # Additional sanity: verify that learning actually occurred.
    # Compare a 200-step run (where training fires after step ~16) against
    # a 10-step run (no training yet because buffer < batch_size).
    schema_long = make_smoke_schema()
    agent_long, _ = train_unified(
        schema_template=schema_long, medoids=[1], num_cycles=1,
        steps_per_building_episode=200, order='B1B5', seed=1234,
        output_dir=output_dir, tag='test_long', init_steps_per_building=200,
    )
    schema_short = make_smoke_schema()
    agent_short, _ = train_unified(
        schema_template=schema_short, medoids=[1], num_cycles=1,
        steps_per_building_episode=10, order='B1B5', seed=1234,
        output_dir=output_dir, tag='test_short', init_steps_per_building=10,
    )
    sum_long = float(agent_long.actor.linear1.weight.sum().item())
    sum_short = float(agent_short.actor.linear1.weight.sum().item())
    report('T16 (extra): actor params changed (learning occurred)',
           sum_long != sum_short and agent_long._param_version > 0 and agent_short._param_version == 0,
           f'long={sum_long:.4f} (ver={agent_long._param_version}) short={sum_short:.4f} (ver={agent_short._param_version})')

    # Test 17 (extra): replay buffer is FIFO ring with correct position tracking
    cap = agent.replay_buffer.capacity
    pos = agent.replay_buffer.position
    fill = len(agent.replay_buffer)
    expected_pos = 120 % cap
    report('T17 (extra): buffer position is correct',
           pos == expected_pos and fill == 120,
           f'pos={pos} (expected {expected_pos}) fill={fill}')

    return results


if __name__ == '__main__':
    print('Running implementation tests for Contribution 2...\n')
    results = test_1_to_15()
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f'\n{"="*50}')
    print(f'SUMMARY: {n_pass}/{n_total} tests passed')
    if n_pass < n_total:
        print('FAILED tests:')
        for name, ok, detail in results:
            if not ok:
                print(f'  - {name}: {detail}')
        sys.exit(1)
    sys.exit(0)
