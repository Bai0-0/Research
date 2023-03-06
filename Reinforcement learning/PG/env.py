#%%
from scipy.stats import norm
import numpy as np


# Create stock price series by geometric Brownian motion
class GBMStock(object):

    def __init__(self, S0, rf, sigma, n_days) -> None:
        self.S0 = S0
        self.rf = rf
        self.sigma = sigma
        self.n_step = n_days
        self.T = n_days / 365

    def sample(self, n_episode):
        self.mat_S = np.zeros((self.n_step, n_episode))
        self.mat_V = np.zeros((self.n_step, n_episode))
        self.mat_delta = np.zeros((self.n_step, n_episode))
        self.mat_S[0] = self.S0
        self.mat_V[0] = self.call_price(self.S0, self.T, self.S0)
        self.mat_delta[0] = self.delta(self.S0, self.T, self.S0)
        dt = 1 / 365

        for t in range(1, self.n_step):
            tau = (self.n_step - t) / 365
            Z = np.random.standard_normal((1, n_episode))
            self.mat_S[t] = self.mat_S[t-1] * np.exp((self.rf - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)
            self.mat_V[t] = self.call_price(self.mat_S[t], tau, self.S0)
            self.mat_delta[t] = self.delta(self.mat_S[t], tau, self.S0)

        return self.mat_S, self.mat_V

    def call_price(self, s, t, k):
        d1 = (np.log(s/k) + (self.rf+0.5*self.sigma**2)*t)/(self.sigma*np.sqrt(t))
        d2 = d1 - self.sigma * np.sqrt(t)
        return norm.cdf(d1) * s - norm.cdf(d2) * k * np.exp(-self.rf * t)

    def delta(self, S, tau, K):
        d1 = (np.log(S/K) + (self.rf+0.5*self.sigma**2)*tau)/(self.sigma*np.sqrt(tau))
        return norm.cdf(d1)

    def get_delta(self):
        # must be used after self.sample()
        return self.mat_delta


class BinomialStock(object):

    def __init__(self, S0, rf, sigma, n_step) -> None:
        self.S0 = S0
        self.K = S0
        self.rf = rf
        self.sigma = sigma
        dt = 1 / 365
        self.T = n_step * dt
        self.u = np.exp(self.sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.R = np.exp(rf * dt)
        self.n_step = n_step
        self.p = (self.R - self.d) / (self.u - self.d)

        self.stock_price = np.zeros((self.n_step+1, self.n_step+1))
        self.option_price = np.zeros((self.n_step+1, self.n_step+1))

        for j in range(self.n_step+1):
            for i in range(j+1):
                self.stock_price[i, j] = self.S0 * self.u**(j-i) * self.d**(i)

        # terminal payoff of call option
        self.option_price[:, self.n_step] = np.maximum(self.stock_price[:, self.n_step] - self.K, 0)
    
        for j in reversed(range(self.n_step)):
            for i in range(j+1):
                # option value if holds
                opt_val = (self.p) * self.option_price[i, j+1] + (1-self.p) * self.option_price[i+1, j+1]
                opt_val = np.exp(- self.rf * dt) * opt_val # discounted by risk-free rate
                self.option_price[i, j] = opt_val

    def sample(self, n_episode):

        self.mat_S = np.zeros((self.n_step, n_episode))
        self.mat_V = np.zeros((self.n_step, n_episode))

        for episode in range(n_episode):
            # we store the S0 and C0, so we need the first element of up-down vector be 1
            up_down = np.concatenate([[1], np.random.binomial(1, self.p, size=self.n_step)])
            S_list = []
            V_list = []
            i = 0
            for j in range(self.n_step):
                if up_down[j]:
                    S_list.append(self.stock_price[i, j])
                    V_list.append(self.option_price[i, j])
                else:
                    i += 1
                    S_list.append(self.stock_price[i, j])
                    V_list.append(self.option_price[i, j])
            
            self.mat_S[:, episode] = np.array(S_list)
            self.mat_V[:, episode] = np.array(V_list)
        
        return self.mat_S, self.mat_V

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = GBMStock(50, 0.05, 0.3, 10)
    stock, call = env.sample(100)
    
    plt.plot(stock)
    plt.show()

    env = BinomialStock(50, 0.05, 0.3, 10)
    stock, call = env.sample(100)

    plt.plot(call)
    plt.show()


