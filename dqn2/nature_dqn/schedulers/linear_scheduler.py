import numpy as np


class LinearScheduler(object):
    def __init__(self, eps_begin, eps_end, n_steps):
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.n_steps = n_steps

    def update(self, t):
        if t > self.n_steps :
            self.epsilon = self.eps_end
        else:
            self.epsilon = self.eps_begin - (self.eps_begin-self.eps_end) * t / self.n_steps

    def get_eps(self):
        return self.epsilon


class LinearExploration(LinearScheduler):
    def __init__(self, env, eps_begin, eps_end, n_steps):
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, n_steps)

    def get_action(self, best_action):
        bound = np.random.rand(1)
        if bound < self.epsilon:
            return self.env.action_space.sample()
        else:
            return best_action
