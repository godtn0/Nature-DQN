import gym
import numpy as np
import numpy
from collections import deque
from gym import spaces


class AtariPreproWrapper(gym.Wrapper):
    def __init__(self, env, prepro):
        super(AtariPreproWrapper, self).__init__(env)
        self.env = env
        self.prepro = prepro
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, done, info

    def reset(self):
        self.obs = self.prepro(self.env.reset())
        return self.obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Wrapper from Berkeley's Assignment
    Takes a max pool over the last n states
    """
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


def deepmind_wrapping(env, prepro=None, skip=None, clipreward=True, atariprepro=True, maxandskip=True):
    if atariprepro:
        env = AtariPreproWrapper(env, prepro)
        print('atariprepro yes')
    if clipreward:
        env = ClipRewardEnv(env)
    if maxandskip:
        env = MaxAndSkipEnv(env, skip)

    return env
