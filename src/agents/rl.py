



import random
import numpy as np
import csv
import os
# conditional imports
try:
    import torch
    from torch.distributions import Normal
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

class ActorNetwork(nn.Module):
    def __init__(self, 
                 num_inputs, 
                 num_actions, 
                 action_space, 
                 action_scaling_coef, 
                 hidden_dim = [400,300],
                 init_w = 3e-3, 
                 log_std_min = -20, 
                 log_std_max = 2, 
                 epsilon = 1e-6):
        
        super(ActorNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        ######
        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])
        ######
        self.action = nn.Linear(hidden_dim[1], num_actions[0])
        self.action.weight.data.uniform_(-init_w, init_w)
        self.action.bias.data.uniform_(-init_w, init_w)
            
            
    def forward(self, state):
        x = self.bn1(F.relu(self.linear1(state)))
        x = self.bn2(F.relu(self.linear2(x)))
        action =torch.tanh(self.action(x))
        return action
    
    
    def sample(self, state):
        action = self.forward(state)
        return action

    def to(self, device):
        return super(ActorNetwork, self).to(device)

#random buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class RegressionBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.x = []
        self.y = []
        self.position = 0
    
    def push(self, variables, targets):
        if len(self.x) < self.capacity and len(self.x)==len(self.y):
            self.x.append(None)
            self.y.append(None)
        
        self.x[self.position] = variables
        self.y[self.position] = targets
        self.position = (self.position + 1) % self.capacity
    
    def __len__(self):
        return len(self.x)
    
class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[400,300], init_w=3e-4):
        super(CriticNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs+num_actions[0] , hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        ########

        #######
        self.Q = nn.Linear(hidden_size[1], 1)
        self.Q.weight.data.uniform_(-init_w, init_w)
        self.Q.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.Q(x)
        return x










