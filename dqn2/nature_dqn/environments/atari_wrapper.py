import gym
import numpy as np
import numpy
from collections import deque
from gym import spaces


class NoopResetWrapper(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetWrapper, self).__init__(env)
        self.env = env
        self.noop_max=noop_max
        self.noop_action = 0

    def reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max+1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()

        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeWrapper(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


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


def deepmind_wrapping(env, prepro=None, skip=None, clipreward=True, atariprepro=True, maxandskip=True, episodiclife=True
                      , noopreset=True):
    if atariprepro:
        env = AtariPreproWrapper(env, prepro)
        print('atariprepro yes')
    if clipreward:
        env = ClipRewardEnv(env)
    if maxandskip:
        env = MaxAndSkipEnv(env, skip)
    if episodiclife:
        env = EpisodicLifeWrapper(env)
    if noopreset:
        env = NoopResetWrapper(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    return env
