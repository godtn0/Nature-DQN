import numpy as np
import random
from nature_dqn.configs.config import config
from collections import deque


class ReplayBuffer(object):
    def __init__(self, size, scale):
        self.buffer_size = size
        self.idx = -1
        self.scale = scale

        self.obs    = deque()
        self.action = deque()
        self.reward = deque()
        self.done   = deque()

    def store_frame(self, obs):
        self.obs.append(obs)
        self.idx += 1

    def store_effect(self, idx, action, reward, done):
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)

    def sample(self, batch_size):
        if (self.idx+1) < batch_size:
            assert()
        idxes = np.random.choice(np.arange(self.idx), batch_size, replace=False)
        obs_batch = np.array([self._encode_obs(idx) for idx in idxes])
        act_batch = np.array([self.action[idx] for idx in idxes], dtype=np.int32).reshape(-1)
        rew_batch = np.array([self.reward[idx] for idx in idxes], dtype=np.float32).reshape(-1)
        done_batch = np.array([self.done[idx] for idx in idxes], dtype=np.float32).reshape(-1)
        next_obs_batch = np.array([self._encode_obs(idx + 1) for idx in idxes])

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch

    def _encode_obs(self, idx):
        if self.scale:
            return np.array(self.obs[idx]).astype(np.float32) / 255.0
        else:
            return np.array(self.obs[idx]).astype(np.float32)

    def encode_recent_observation(self):
        return np.expand_dims(self._encode_obs(self.idx), 0)
