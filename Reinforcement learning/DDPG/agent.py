#%%
import numpy as np
import torch
from model import ActorNetwork, CriticNetwork

class DDPG(object):

    def __init__(self, state_dim, act_dim, hyper_params):
        
        self.state_dim, self.act_dim, self.hyper_params = state_dim, act_dim, hyper_params

        # store (St, At, St+1, Rt), so the number of columns is (state_dim * 2 + act_dim + 1)
        self.replay_memory = np.zeros((hyper_params['memory_size'], state_dim * 2 + act_dim + 1), dtype=np.float32)

        self.pointer = 0

        # initialize learning network
        self.actor_learn = ActorNetwork(state_dim, act_dim)
        self.critic_learn = CriticNetwork(state_dim, act_dim)

        # initialize target network
        self.actor_target = ActorNetwork(state_dim, act_dim)
        self.critic_target = CriticNetwork(state_dim, act_dim)

        # set optimizer 
        self.actor_optim = torch.optim.Adam(self.actor_learn.parameters(),lr=hyper_params['lr_actor'])
        self.critic_optim = torch.optim.Adam(self.critic_learn.parameters(),lr=hyper_params['lr_critic'])
        
        # set TD loss function
        self.loss_TD = torch.nn.MSELoss()

    def predict(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # vector of 1 * state_dim
        action = self.actor_learn(state)[0].detach()
        return action

        # torch.unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.
        # Eg. state = [1, 2]
        # torch.unsqueeze(torch.FloatTensor(state), 0) = [[1, 2]]
        # self.actor_learn(state) = tensor([[0.0657]], grad_fn=<TanhBackward0>)
        # self.actor_learn(state)[0] = tensor([0.0657], grad_fn=<TanhBackward0>)
        # self.actor_learn(state)[0].detach() = tensor([0.0657])

    def learn(self):

        # 1. Soft target replacement
        # theta_target = tau * theta_learn + (1-tau) * theta_target

        tau = self.hyper_params['replacement_rate']
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-tau))')
            eval('self.actor_target.' + x + '.data.add_(tau * self.actor_learn.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-tau))')
            eval('self.critic_target.' + x + '.data.add_(tau * self.critic_learn.' + x + '.data)')

        # 2. Draw traning data from replay memory

        indices = np.random.choice(self.hyper_params['memory_size'], size=self.hyper_params['batch_size'])
        batch = self.replay_memory[indices, :]
        batch_state = torch.FloatTensor(batch[:, :self.state_dim])
        batch_act = torch.FloatTensor(batch[:, self.state_dim: self.state_dim + self.act_dim])
        batch_reward = torch.FloatTensor(batch[:, -self.state_dim - 1: -self.state_dim])
        batch_state_next = torch.FloatTensor(batch[:, -self.state_dim:])

        # 3. Train actor network

        act = self.actor_learn(batch_state)
        Q_val = self.critic_learn(batch_state, act)

        # note that we want to maximize the Q-value of the action, so loss = - Q
        loss_action = -torch.mean(Q_val) 
        self.actor_optim.zero_grad()  # initialize gradient
        loss_action.backward()        # backward propagation
        self.actor_optim.step()       # update all parameters

        # 4. The critic score of current action

        Q_learn = self.critic_learn(batch_state, batch_act)

        # 5. Estimate target Q-value: Q_target = Rt + gamma * Qt+1

        act_next = self.actor_target(batch_state_next)
        Q_val_next = self.critic_target(batch_state_next, act_next)
        Q_target = batch_reward + self.hyper_params['discount_rate'] * Q_val_next 

        # 6. Train critic network

        error_TD = self.loss_TD(Q_target, Q_learn)
        self.critic_optim.zero_grad()
        error_TD.backward()
        self.critic_optim.step()

    def store_memory(self, state, act, reward, state_next):

        data = np.hstack((state, act, [reward], state_next))
        
        # replace the old memory with new memory
        # if pointer > memory size, store from the begining
        index = self.pointer % self.hyper_params['memory_size']  
        
        self.replay_memory[index, :] = data
        self.pointer += 1





# %%
