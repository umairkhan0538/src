import os
from pathlib import Path
from typing import List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except (ModuleNotFoundError, ImportError) as e:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

from citylearn.agents.rlc import RLC
from citylearn.preprocessing import Encoder, RemoveFeature
from agents.rl import CriticNetwork, ReplayBuffer, ActorNetwork

class TD3(RLC):
    def __init__(self, *args, **kwargs):
        r"""Initialize :class:`TD3`.

        Parameters
        ----------
        *args : tuple
            `RLC` positional arguments.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)

        self.hidden_dimension = kwargs.get('hidden_dimension', [400, 300])
        self.discount = kwargs.get('discount', 0.93)
        self.tau = kwargs.get('tau', 0.05)
        self.lr_c = kwargs.get('lr_c', 0.002)
        self.lr_a = kwargs.get('lr_a', 0.002)
        self.alpha = kwargs.get('alpha', 0.5)
        self.batch_size = kwargs.get('batch_size', 256)
        self.replay_buffer_capacity = int(kwargs.get('replay_buffer_capacity', 1e6))
        self.start_training_time_step = kwargs.get('start_training_time_step', 3671)
        self.end_exploration_time_step = kwargs.get('end_exploration_time_step', 3671)
        self.deterministic_start_time_step = kwargs.get('deterministic_start_time_step', 36710)
        self.action_scaling_coefficient = kwargs.get('action_scaling_coef', 0.5)
        self.reward_scaling = kwargs.get('reward_scaling', 5.0)
        self.update_per_time_step = kwargs.get('update_per_time_step', 2)
        self.seed = kwargs.get('seed', 0)

        self.critic_criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = ReplayBuffer(int(self.replay_buffer_capacity))
        self.critic_1 = None
        self.critic_2 = None
        self.target_critic_1 = None
        self.target_critic_2 = None
        self.actor = None
        self.target_actor = None
        self.actor_optimizer = None
        self.critic_1_optimizer = None
        self.critic_2_optimizer = None
        self.policy_noise = 0.01
        self.noise_clip = 0.5
        self.policy_freq = 2 

        self.set_networks()

    def add_to_buffer(self, observations: List[List[float]], actions: List[List[float]], reward: List[float], next_observations: List[List[float]], done: bool):
        r"""Update replay buffer.

        Parameters
        ----------
        observations : List[List[float]]
            Previous time step observations.
        actions : List[List[float]]
            Previous time step actions.
        reward : List[float]
            Current time step reward.
        next_observations : List[List[float]]
            Current time step observations.
        done : bool
            Indication that episode has ended.
        """

        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            o = np.array(self.get_encoded_observations(i, o), dtype=float)
            n = np.array(self.get_encoded_observations(i, n), dtype=float)
            self.replay_buffer.push(o, a, r, n, done)
            if self.time_step >= self.start_training_time_step and self.batch_size <= len(self.replay_buffer):
                for _ in range(self.update_per_time_step):
                    o, a, r, n, d = self.replay_buffer.sample(self.batch_size)

                    tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                    o = tensor(o).to(self.device)
                    n = tensor(n).to(self.device)
                    a = tensor(a).to(self.device)
                    r = tensor(r).unsqueeze(1).to(self.device)
                    d = tensor(d).unsqueeze(1).to(self.device)

                    with torch.no_grad():
                        # Select action according to policy and add clipped noise
                        noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                        next_actions = self.target_actor.forward(n) + noise
                        # Clamp actions to action_space bounds
                        for j, space in enumerate(self.action_space):
                            next_actions[:, j] = next_actions[:, j].clamp(float(space.low[0]), float(space.high[0]))
                        # Compute the target Q values
                        target_Q1 = self.target_critic_1.forward(n, next_actions)
                        target_Q2 = self.target_critic_2.forward(n, next_actions)
                        target_Q = torch.min(target_Q1, target_Q2)
                        Qprime = r + (1 - d) * self.discount * target_Q

                    # Get current Q estimates
                    current_Q1 = self.critic_1.forward(o, a)
                    current_Q2 = self.critic_2.forward(o, a)
                    # Compute critic loss
                    critic_loss_1 = self.critic_criterion(current_Q1, Qprime)
                    critic_loss_2 = self.critic_criterion(current_Q2, Qprime)

                    # Optimize the critics
                    self.critic_1_optimizer.zero_grad()
                    critic_loss_1.backward()
                    self.critic_1_optimizer.step()

                    self.critic_2_optimizer.zero_grad()
                    critic_loss_2.backward()
                    self.critic_2_optimizer.step()

                    # Delayed policy updates
                    if self.time_step % self.policy_freq == 0:
                        # Compute actor loss
                        policy_loss = -self.critic_1.forward(o, self.actor.forward(o)).mean()
                        # Optimize the actor
                        self.actor_optimizer.zero_grad()
                        policy_loss.backward()
                        self.actor_optimizer.step()

                        # Update target networks
                        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
                        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
                        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            else:
                pass

    def select_actions(self, observations: List[List[float]]):
        r"""Provide actions for current time step.

        Will return randomly sampled actions from `action_space` if :attr:`end_exploration_time_step` >= :attr:`time_step` 
        else will use policy to sample actions.
        
        Returns
        -------
        actions: List[List[float]]
            Action values
        """

        if self.time_step <= self.end_exploration_time_step:
            actions = self.get_exploration_actions(observations)
        else:
            actions = self.get_post_exploration_actions(observations)
        self.actions = actions
        self.next_time_step()
        return actions

    def get_post_exploration_actions(self, observations: List[List[float]]) -> List[List[float]]:
        r"""Action sampling using policy, post-exploration time step"""

        actions = []
        for i, o in enumerate(observations):
            o = np.array(self.get_encoded_observations(i, o), dtype=float)
            o = torch.FloatTensor(o).unsqueeze(0).to(self.device)
            self.actor.eval()
            a = self.actor.forward(o)
            a = list(a.detach().cpu().numpy()[0])
            actions.append(a)
            self.actor.train()
        return actions

    def get_exploration_actions(self, observations: List[List[float]]) -> List[List[float]]:
        r"""Return actions based on specific conditions during exploration phase.

        Returns
        -------
        actions: List[List[float]]
            Action values.
        """
        actions = []
        for n, o, i, d in zip(self.observation_names, observations, self.building_information, self.action_dimension):
            soc = o[n.index('electrical_storage_soc')]
            hour = o[n.index('hour')]
            capacity = i['electrical_storage_capacity']
            if 9 <= hour <= 12:
                a = [2.0/capacity for _ in range(d)]
            elif (hour >= 18 or hour < 9) and soc > 0.25:
                a = [-2.0/capacity for _ in range(d)]
            else:
                a = [0.0 for _ in range(d)]
            actions.append(a)
        return actions

    def get_encoded_observations(self, index: int, observations: List[float]) -> List[float]:
        return np.array([j for j in np.hstack(self.encoders[index]*np.array(observations, dtype=float)) if j != None], dtype=float).tolist()

    def set_networks(self, internal_observation_count: int = None):
        internal_observation_count = 0 if internal_observation_count is None else internal_observation_count
        for i in range(len(self.action_dimension)):
            observation_dimension = self.observation_dimension[0] + internal_observation_count
            # Initialize two critic networks for TD3
            self.critic_1 = CriticNetwork(observation_dimension, [self.action_dimension[i]], self.hidden_dimension).to(self.device)
            self.critic_2 = CriticNetwork(observation_dimension, [self.action_dimension[i]], self.hidden_dimension).to(self.device)
            self.target_critic_1 = CriticNetwork(observation_dimension, [self.action_dimension[i]], self.hidden_dimension).to(self.device)
            self.target_critic_2 = CriticNetwork(observation_dimension, [self.action_dimension[i]], self.hidden_dimension).to(self.device)
            # Initialize actor network
            self.actor = ActorNetwork(observation_dimension, [self.action_dimension[i]], self.action_space[i], self.action_scaling_coefficient, self.hidden_dimension).to(self.device)
            self.target_actor = ActorNetwork(observation_dimension, [self.action_dimension[i]], self.action_space[i], self.action_scaling_coefficient, self.hidden_dimension).to(self.device)

            self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr_c)
            self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr_c)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_a)

            # Initialize target network parameters
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

    def set_encoders(self) -> List[List[Encoder]]:
        encoders = super().set_encoders()
        for i, o in enumerate(self.observation_names):
            for j, n in enumerate(o):
                if n == 'net_electricity_consumption':
                    encoders[i][j] = RemoveFeature()
                else:
                    pass
        return encoders
