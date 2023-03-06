#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from agent import DDPG
from env import GBMStock




def train_episode(agent, env, hyper_params):
    n_step = env.n_step
    mat_S, mat_V = env.sample(hyper_params['n_train_episodes'])
    cum_reward = []

    for episode in range(hyper_params['n_train_episodes']):

        t_list = range(n_step)
        S_list = mat_S[:, episode]
        V_list = mat_V[:, episode]
        act_list = np.zeros(n_step)
        reward_list = np.zeros(n_step)

        for t in t_list:
            state = [t_list[t], S_list[t], V_list[t]]
            act_list[t] = agent.predict(state)
            # TODO 随机性

            if t > 0:
                reward_list[t] = -np.abs(V_list[t] - act_list[t-1] * S_list[t])

            if t < n_step-1:
                state_next = [t_list[t+1], S_list[t+1], V_list[t+1]]
                agent.store_memory(state, act_list[t], reward_list[t], state_next)

            
            if agent.pointer > hyper_params['memory_size']:
                agent.learn()
            
        reward_total = np.zeros(n_step)
        for t in t_list[::-1]:
            if t < n_step-1:
                reward_total[t] = reward_list[t+1] + 0.9 * reward_total[t+1]
            
        cum_reward.append(reward_total[0])

        if episode/hyper_params['n_train_episodes']*100 % 10 == 0:
            print('Finish training %d%% episodes.' % int(episode/hyper_params['n_train_episodes']*100))

    return cum_reward

def test_episode(agent, env, n_episodes):
    n_step = env.n_step
    mat_S, mat_V = env.sample(n_episodes)
    cum_reward = []

    for episode in range(n_episodes):

        t_list = range(n_step)
        S_list = mat_S[:, episode]
        V_list = mat_V[:, episode]
        act_list = np.zeros(n_step)
        reward_list = np.zeros(n_step)

        for t in t_list:
            state = [t_list[t], S_list[t], V_list[t]]
            act_list[t] = agent.predict(state)

            if t > 0:
                reward_list[t] = -np.abs(V_list[t] - act_list[t-1] * S_list[t])
        
        reward_total = np.zeros(n_step)
        for t in t_list[::-1]:
            if t < n_step-1:
                reward_total[t] = reward_list[t+1] + 0.9 * reward_total[t+1]
            
        cum_reward.append(reward_total[0])

    return cum_reward



hyper_params = {
    'n_train_episodes': 3000,
    'n_days': 20,
    'lr_actor': 1e-3,
    'lr_critic': 1e-3,
    'discount_rate': 0.9,
    'replacement_rate':0.01,
    'memory_size': 2000,
    'batch_size': 32,
}


env = GBMStock(50, 0.05, 0.3, hyper_params['n_days'])
state_dim = 3
act_dim = 1
agent = DDPG(state_dim, act_dim, hyper_params)

train_reward = train_episode(agent, env, hyper_params)
test_reward = test_episode(agent, env, 10000)

plt.plot(range(hyper_params['n_train_episodes']), train_reward)
plt.show()
#%%

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(range(hyper_params['n_train_episodes']), train_reward)
plt.title('Training with Reward -abs(V - action * S)')
plt.ylabel('Discounted Total Reward')
plt.ylim((-550, 0))
plt.xlabel('Episodes')



plt.subplot(1, 2, 2)
plt.plot(range(10000), test_reward)
plt.title('Testing with Reward -abs(V - action * S)')
plt.ylabel('Discounted Total Reward')
plt.ylim((-550, 0))
plt.xlabel('Episodes')
plt.show()
print(np.mean(test_reward), np.std(test_reward))
# %%
