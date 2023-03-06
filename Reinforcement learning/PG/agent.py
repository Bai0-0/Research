#%%
import numpy as np
from scipy import stats


class PolicyGradientAgent(object):

    def __init__(self, n_step, learning_rate, gamma) -> None:
        self.n_step = n_step
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.num_state = 3
        
        self.theta = np.zeros((2 * self.num_state, 1))
        self.theta_fast = self.theta.copy()

    def feature_vec(self, state):
        return state


    def predict(self, state):

        # state = [time, stock price, call price]

        state = np.array(state).reshape((self.num_state, 1))
        feature = self.feature_vec(state)
        theta_mu = self.theta[:self.num_state].reshape((self.num_state, 1))
        theta_sigma = self.theta[self.num_state:].reshape((self.num_state, 1))

        mu = (theta_mu.T @ feature)[0, 0]
        sigma = np.exp((theta_sigma.T @ feature)[0, 0])

        action = np.random.normal(loc=mu, scale=sigma)

        return action


    def learn(self, state_curr, act_curr, reward_total):

        # state = [time, stock price, call price]

        state_idx = state_curr[0]
        state_curr = np.array(state_curr).reshape((self.num_state, 1))
        feature = self.feature_vec(state_curr)
        theta_mu = self.theta[:self.num_state].reshape((self.num_state, 1))
        theta_sigma = self.theta[self.num_state:].reshape((self.num_state, 1))

        mu = (theta_mu.T @ feature)[0, 0]
        sigma = np.exp((theta_sigma.T @ feature)[0, 0])

        gradient_ln_mu = 1 / (sigma**2) * (act_curr - mu) * feature
        gradient_ln_sigma = ((act_curr - mu)**2 / (sigma**2) - 1) * feature
        gradient_ln = (np.concatenate([gradient_ln_mu, gradient_ln_sigma])).reshape(2*self.num_state, 1)

        self.theta_fast = self.theta_fast + self.learning_rate * (self.gamma**state_idx) * reward_total * gradient_ln

    def fast_to_slow(self):
        self.theta = self.theta_fast.copy()


agent = PolicyGradientAgent(100, 0.1, 0.1)
agent.predict([0, 50, 4])
agent.learn((1, 20, 3), 0.5, 23)
agent.predict([0, 50, 4])

# %%
