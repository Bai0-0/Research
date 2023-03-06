#%%
from matplotlib.pyplot import axis
import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 10)
        self.fc1.weight.data.normal_(0, 0.1) # initialization
        self.out = nn.Linear(10, act_dim)
        self.out.weight.data.normal_(0, 0.1) # initialization
    def forward(self, state):
        state = self.fc1(state)
        state = nn.functional.relu(state)
        state = self.out(state)
        action = torch.tanh(state)
        return action

        # 1. Input state vector, output 10-d vector
        # 2. Activate by ReLU
        # 4. Input 10-d vector, output action
        # 4. Activate by tanh


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(CriticNetwork, self).__init__()
        self.fc_state = nn.Linear(state_dim, 10)
        self.fc_state.weight.data.normal_(0, 0.1) # initialization
        self.fc_act = nn.Linear(act_dim, 10)
        self.fc_act.weight.data.normal_(0, 0.1) # initialization
        self.out = nn.Linear(20, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization
    def forward(self, state, act):
        s = self.fc_state(state)
        a = self.fc_act(act)
        state_action_pair = torch.cat((s, a), dim=1)
        # print(s.shape, a.shape, state_action_pair.shape)
        state_action_pair = nn.functional.relu(state_action_pair)
        Q_val = self.out(state_action_pair)
        return Q_val

        # 1. Input state and action vector, output 10-d vector respectively
        # 2. Concatenate two vectors
        # 3. Activate by ReLU
        # 4. Input 20-d vector, output Q-value

# %%
