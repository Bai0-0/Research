#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from env import GBMStock

n_days = 20
env = GBMStock(50, 0.05, 0.3, n_days)
n_step = env.n_step
n_test_episode = 10000

# np.random.seed(1)

n_episode = n_test_episode
mat_S, mat_V = env.sample(n_episode)
gamma = 0.9

cum_reward = []
mat_act = np.zeros((n_episode, n_step))

for episode in range(n_episode):

    t_list = range(n_step)
    S_list = mat_S[:, episode]
    V_list = mat_V[:, episode]
    delta_list = np.zeros(n_step)
    reward_list = np.zeros(n_step)

    for t in t_list:
        if t > 0:
            dV = V_list[t] - V_list[t-1]
            dS = S_list[t] - S_list[t-1]
            delta = dV / dS
            if delta > 1:
                delta = 1
            elif delta < -1:
                delta = -1
            delta_list[t] = delta


    for t in t_list:
        if t > 0:
            reward_list[t] = -np.abs(V_list[t] - delta_list[t-1] * S_list[t])
            # reward_list[t] = -np.abs((V_list[t]- V_list[t-1]) - delta_list[t-1] * (S_list[t]-S_list[t-1]))

    reward_total = np.zeros(n_step)
    # gamma = 1
    for t in t_list[::-1]:
        if t < n_step-1:
            reward_total[t] = reward_list[t+1] + gamma * reward_total[t+1]

    cum_reward.append(reward_total[0])

    if episode/n_episode*100 % 10 == 0:
        print('Finish testing %d%% episodes.' % int(episode/n_episode*100))



plt.figure(figsize=(8, 8))

plt.plot(range(n_test_episode), cum_reward)
plt.title('Testing with Reward -abs(V - action * S)')
plt.ylabel('Discounted Total Reward')
plt.ylim((-550, 0))
plt.xlabel('Episodes')
plt.show()
print(np.mean(cum_reward), np.std(cum_reward))
# %%
