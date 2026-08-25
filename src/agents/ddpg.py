import os
import sys
from pathlib import Path
from typing import List
import numpy as np

print(f"USING {os.path.join(*Path(os.path.dirname(__file__)).absolute().parts[0:])}, {os.path.basename(__file__)}")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except (ModuleNotFoundError, ImportError):
    raise ImportError("Please install torch and torchvision: pip install torch torchvision")

from citylearn.agents.rlc import RLC
from citylearn.preprocessing import Encoder, RemoveFeature
from agents.rl import CriticNetwork, ReplayBuffer, ActorNetwork

class DDPG(RLC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.critic_criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(int(self.replay_buffer_capacity))

        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.set_networks()

    def add_to_buffer(self, observations: List[List[float]], actions: List[List[float]], rewards: List[float], next_observations: List[List[float]], done: bool):
        buffer_ready = len(self.replay_buffer) >= self.batch_size
        training_started = self.time_step >= self.start_training_time_step

        for i, (obs, act, rew, next_obs) in enumerate(zip(observations, actions, rewards, next_observations)):
            obs_enc = np.array(self.get_encoded_observations(i, obs), dtype=float)
            next_obs_enc = np.array(self.get_encoded_observations(i, next_obs), dtype=float)
            self.replay_buffer.push(obs_enc, act, rew, next_obs_enc, done)

        if training_started and buffer_ready:
            for _ in range(self.update_per_time_step):
                obs, act, rew, next_obs, dones = self.replay_buffer.sample(self.batch_size)
                tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                obs = tensor(obs).to(self.device)
                next_obs = tensor(next_obs).to(self.device)
                act = tensor(act).to(self.device)
                rew = tensor(rew).unsqueeze(1).to(self.device)
                dones = tensor(dones).unsqueeze(1).to(self.device)

                # Critic update
                q_vals = self.critic(obs, act)
                next_actions = self.target_actor(next_obs).detach()
                next_q = self.target_critic(next_obs, next_actions)
                q_target = rew + self.discount * next_q
                critic_loss = self.critic_criterion(q_vals, q_target)

                # Actor update
                policy_loss = -self.critic(obs, self.actor(obs)).mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Soft update of target networks
                for t_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    t_param.data.copy_(param.data * self.tau + t_param.data * (1.0 - self.tau))

                for t_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                    t_param.data.copy_(param.data * self.tau + t_param.data * (1.0 - self.tau))

    def select_actions(self, observations: List[List[float]]):
        if self.time_step <= self.end_exploration_time_step:
            actions = self.get_exploration_actions(observations)
        else:
            actions = self.get_post_exploration_actions(observations)

        self.actions = actions
        self.next_time_step()
        return actions

    def get_post_exploration_actions(self, observations: List[List[float]]) -> List[List[float]]:
        self.actor.eval()
        actions = []

        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor = torch.FloatTensor(self.get_encoded_observations(i, obs)).unsqueeze(0).to(self.device)
                action = self.actor.sample(obs_tensor)
                actions.append(list(action.cpu().numpy()[0]))

        self.actor.train()
        return actions

    def get_exploration_actions(self, observations: List[List[float]]) -> List[List[float]]:
        actions = []

        for n, o, i, d in zip(self.observation_names, observations, self.building_information, self.action_dimension):
            soc = o[n.index('electrical_storage_soc')]
            hour = o[n.index('hour')]
            capacity = i['electrical_storage_capacity']

            if 9 <= hour <= 12:
                a = [2.0 / capacity for _ in range(d)]
            elif (hour >= 18 or hour < 9) and soc > 0.25:
                a = [-2.0 / capacity for _ in range(d)]
            else:
                a = [0.0 for _ in range(d)]

            actions.append(a)

        return actions

    def get_encoded_observations(self, index: int, observations: List[float]) -> List[float]:
        encoded = [j for j in np.hstack(self.encoders[index] * np.array(observations, dtype=float)) if j is not None]
        return np.array(encoded, dtype=float).tolist()

    def set_networks(self, internal_observation_count: int = 0):
        obs_dim = self.observation_dimension[0] + internal_observation_count
        act_dim = self.action_dimension[0]

        self.critic = CriticNetwork(obs_dim, self.action_dimension, self.hidden_dimension).to(self.device)
        self.target_critic = CriticNetwork(obs_dim, self.action_dimension, self.hidden_dimension).to(self.device)

        self.actor = ActorNetwork(obs_dim, self.action_dimension, self.action_space, self.action_scaling_coefficient, self.hidden_dimension).to(self.device)
        self.target_actor = ActorNetwork(obs_dim, self.action_dimension, self.action_space, self.action_scaling_coefficient, self.hidden_dimension).to(self.device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.004)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    def set_encoders(self) -> List[List[Encoder]]:
        encoders = super().set_encoders()

        for i, o in enumerate(self.observation_names):
            for j, name in enumerate(o):
                if name == 'net_electricity_consumption':
                    encoders[i][j] = RemoveFeature()

        return encoders
