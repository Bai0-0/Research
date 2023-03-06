#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from agent import PolicyGradientAgent
from env import GBMStock, BinomialStock



def train_episode(agent, env, n_episode):

    mat_S, mat_V = env.sample(n_episode)
    cum_reward = []
    mat_act = np.zeros((n_episode, n_step))
    mat_theta = np.zeros((n_episode, 6))
    rf = 0.05
    dt = 1 / 365

    for episode in range(n_episode):
        # generate 1 episode

        t_list = range(n_step)
        S_list = mat_S[:, episode]
        V_list = mat_V[:, episode]
        act_list = np.zeros(n_step)
        reward_list = np.zeros(n_step)
        for t in t_list:
            state = [t_list[t], S_list[t], V_list[t]]
            act_list[t] = agent.predict(state)
            # if t > 1:
            #     pv = V_list[t] - act_list[t-1] * S_list[t]
            #     pre_pv = V_list[t-1] - act_list[t-2] * S_list[t-1]
            #     reward_list[t] = -np.abs(pv * np.exp(-rf * dt) - pre_pv)
            if t > 0:
                reward_list[t] = -np.abs(V_list[t] - act_list[t-1] * S_list[t])
                # reward_list[t] = -np.abs((V_list[t]- V_list[t-1]) - act_list[t-1] * (S_list[t]-S_list[t-1]))

        reward_total = np.zeros(n_step)
        for t in t_list[::-1]:
            if t < n_step-1:
                reward_total[t] = reward_list[t+1] + agent.gamma * reward_total[t+1]

        for t in t_list:
            if t < n_step-1:
                state = [t_list[t], S_list[t], V_list[t]]
                agent.learn(state, act_list[t], reward_total[t])

        agent.fast_to_slow()

        cum_reward.append(reward_total[0])
        mat_act[episode] = act_list.copy()
        mat_theta[episode] = agent.theta.reshape(6).copy()

        if episode/n_episode*100 % 10 == 0:
            print('Finish training %d%% episodes.' % int(episode/n_episode*100))


    return cum_reward, mat_act, mat_theta


def test_episode(agent, env, n_episode):
    mat_S, mat_V = env.sample(n_episode)
    cum_reward = []
    mat_act = np.zeros((n_episode, n_step))
    mat_theta = np.zeros((n_episode, 6))

    for episode in range(n_episode):
        # generate 1 episode

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
                # reward_list[t] = -np.abs((V_list[t]- V_list[t-1]) - act_list[t-1] * (S_list[t]-S_list[t-1]))

        reward_total = np.zeros(n_step)
        for t in t_list[::-1]:
            if t < n_step-1:
                reward_total[t] = reward_list[t+1] + agent.gamma * reward_total[t+1]


        cum_reward.append(reward_total[0])
        mat_act[episode] = act_list.copy()
        mat_theta[episode] = agent.theta.reshape(6).copy()

        if episode/n_episode*100 % 10 == 0:
            print('Finish testing %d%% episodes.' % int(episode/n_episode*100))


    return cum_reward, mat_act, mat_theta


np.random.seed(1)
learning_rate = 1e-09
gamma = 0.9
n_days = 20
env = GBMStock(50, 0.05, 0.3, n_days)
# env = BinomialStock(50, 0.05, 0.3, n_days)
n_step = env.n_step

agent = PolicyGradientAgent(n_step, learning_rate, gamma)
n_train_episode = 10000
n_test_episode = 10000
train_reward, train_act, train_theta = train_episode(agent, env, n_train_episode)
test_reward, test_act, test_theta = test_episode(agent, env, n_test_episode)

#%%
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(range(n_train_episode), train_reward)
plt.title('Training with Reward -abs(V - action * S)')
plt.ylabel('Discounted Total Reward')
plt.ylim((-550, 0))
plt.xlabel('Episodes')

plt.subplot(1, 2, 2)
plt.plot(range(n_test_episode), test_reward)
plt.title('Testing with Reward -abs(V - action * S)')
plt.ylabel('Discounted Total Reward')
plt.ylim((-550, 0))
plt.xlabel('Episodes')
plt.show()
print(np.mean(test_reward), np.std(test_reward))
# %%
