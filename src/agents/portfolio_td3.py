"""
PortfolioTD3
=============

A thin wrapper around the repo's `agents.td3.TD3` that:
  1. Replaces the existing 5-tuple `ReplayBuffer` with a `PortfolioReplayBuffer`
     so that source metadata (building_id, cycle, episode_step, global_step)
     is preserved alongside every transition.
  2. Exposes a `switch_environment(env)` helper that re-points all
     environment-dependent attributes (env, observation_space, action_space,
     observation_names, building_information, encoders) to a new
     single-building env WITHOUT recreating the actor, twin critics, target
     networks, optimizers, or the replay buffer.
  3. Exposes a `learn(n_updates)` method that performs the same gradient
     updates the existing TD3 performs inside `add_to_buffer` (verbatim
     copy), but reading from the portfolio buffer instead of the
     5-tuple buffer. The existing `add_to_buffer` path remains intact for
     backward compatibility.
  4. Exposes `freeze()` for zero-shot evaluation.

Existing TD3 algorithm is preserved bit-for-bit. The wrapper is additive.
"""

from __future__ import annotations

import copy
import numpy as np
import torch

from citylearn.agents.rlc import RLC
from agents.td3 import TD3
from agents.portfolio_replay_buffer import PortfolioReplayBuffer


class PortfolioTD3(TD3):
    """
    Persistent single TD3 that owns a PortfolioReplayBuffer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace the existing 5-tuple buffer with the portfolio buffer.
        self.replay_buffer = PortfolioReplayBuffer(self.replay_buffer_capacity)

        # Stable identifier for the persistence verification tests.
        self.instance_id = id(self)
        self._param_version = 0

        # Track which building is the current source of transitions.
        self.current_building_id = None

    @classmethod
    def from_base_td3(cls, base_agent, building_id: int):
        """
        Build a PortfolioTD3 by wrapping an already-constructed TD3 agent.
        The base_agent must have been created via env.load_agent() so all
        parent-class state (networks, optimizers, encoders, encoders,
        observation_space, action_space, etc.) is already initialised.

        Returns a PortfolioTD3 instance that shares the base agent's
        state via __class__ reassignment, replacing only the replay buffer
        and adding the new methods.
        """
        if not isinstance(base_agent, TD3):
            raise TypeError(f"base_agent must be a TD3, got {type(base_agent)}")
        # Replace the class on the existing instance. All instance state
        # (networks, optimizers, env-pointers) is preserved.
        base_agent.__class__ = cls
        # Swap the 5-tuple buffer for a PortfolioReplayBuffer.
        capacity = int(base_agent.replay_buffer_capacity)
        base_agent.replay_buffer = PortfolioReplayBuffer(capacity)
        base_agent.instance_id = id(base_agent)
        base_agent._param_version = 0
        base_agent.current_building_id = int(building_id)
        return base_agent

    # -------------------------------------------------------- switch_building
    def switch_building(self, env, building_id: int) -> None:
        """
        Re-point all environment-dependent attributes of the agent to `env`.
        The networks, optimizers, replay buffer, and global timestep are
        NOT touched. The encoders are rebuilt from the new env's
        observation names.

        After calling this method, the agent can run a new episode on
        the new building; existing state continues to flow.
        """
        # Re-attach to the new env using the upstream Agent setters.
        # We cannot simply assign self.env (private attribute). The upstream
        # RLC's `set_encoders` reads from self.observation_names, so we must
        # update those first.
        self.observation_names = env.observation_names
        # CityLearnEnv in 1.4.4 does not have `action_names`. The env keeps
        # the active action names per-building; we leave action_names as it
        # was set during construction (the existing TD3 never reads it in
        # add_to_buffer or learn, so this is safe).
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.building_information = env.unwrapped.get_building_information()
        self.encoders = self.set_encoders()
        # Note: we do NOT call self.reset() (which would zero time_step).
        # We also do NOT touch the networks, optimizers, or replay buffer.
        self.current_building_id = int(building_id)

    # -------------------------------------------------------- learn (verbatim TD3 update)
    def learn(self, n_updates: int = None) -> int:
        """
        Verbatim copy of the gradient-update block from TD3.add_to_buffer,
        but reading from the portfolio buffer. The existing TD3 update
        mechanics are preserved bit-for-bit.

        Returns the number of critic update iterations performed.
        """
        if n_updates is None:
            n_updates = self.update_per_time_step

        if not (self.time_step >= self.start_training_time_step
                and self.batch_size <= len(self.replay_buffer)):
            return 0

        critic_steps = 0
        for _ in range(n_updates):
            o, a, r, n, d = self.replay_buffer.sample(self.batch_size)

            tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
            o = tensor(o).to(self.device)
            n = tensor(n).to(self.device)
            a = tensor(a).to(self.device)
            r = tensor(r).unsqueeze(1).to(self.device)
            d = tensor(d).unsqueeze(1).to(self.device)

            with torch.no_grad():
                noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_actions = self.target_actor.forward(n) + noise
                for j, space in enumerate(self.action_space):
                    next_actions[:, j] = next_actions[:, j].clamp(float(space.low[0]), float(space.high[0]))
                target_Q1 = self.target_critic_1.forward(n, next_actions)
                target_Q2 = self.target_critic_2.forward(n, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                Qprime = r + (1 - d) * self.discount * target_Q

            current_Q1 = self.critic_1.forward(o, a)
            current_Q2 = self.critic_2.forward(o, a)
            critic_loss_1 = self.critic_criterion(current_Q1, Qprime)
            critic_loss_2 = self.critic_criterion(current_Q2, Qprime)

            self.critic_1_optimizer.zero_grad()
            critic_loss_1.backward()
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_loss_2.backward()
            self.critic_2_optimizer.step()

            critic_steps += 1

            if self.time_step % self.policy_freq == 0:
                policy_loss = -self.critic_1.forward(o, self.actor.forward(o)).mean()
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
                for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
                for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        if critic_steps > 0:
            self._param_version += 1
        return critic_steps

    # -------------------------------------------------------- push (extended)
    def push_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        building_id: int,
        cycle: int,
        episode_step: int,
        global_step: int,
    ) -> None:
        """Push a single transition with source metadata into the portfolio buffer."""
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            building_id=int(building_id),
            cycle=int(cycle),
            episode_step=int(episode_step),
            global_step=int(global_step),
        )

    # -------------------------------------------------------- freeze
    def freeze(self) -> None:
        """
        Freeze the actor for zero-shot evaluation. Sets the actor to eval mode
        and disables gradient computation for all policy parameters.
        """
        for p in self.actor.parameters():
            p.requires_grad = False
        self.actor.eval()
