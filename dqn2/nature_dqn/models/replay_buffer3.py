import numpy as np

class ReplayBuffer(object):

    def __init__(self, size, observation_dims, history_length):

        #initialize
        self.size = size
        self.observation_dims = observation_dims
        self.history_length = history_length
        self.idx = -1
        self.next_idx = 0
        self.num_in_buffer = 0

        #initialize buffer
        self.obs = np.empty([self.size]+ observation_dims, dtype = np.float32)
        self.reward = np.empty([self.size], dtype = np.float32)
        self.actions = np.empty([self.size], dtype = np.int32)
        self.done_mask = np.empty([self.size], dtype = np.float32)

    def store(self, obs):

        #안빼도 됨????
        self.obs[self.next_idx] = obs
        self.idx = (self.idx + 1)%(self.size)

    def store_effect(self, action, reward, done_mask):

        self.actions[self.next_idx] = action
        self.reward[self.next_idx] = reward
        self.done_mask[self.next_idx] = done_mask

        self.num_in_buffer = min(self.num_in_buffer+1, self.size)
        self.next_idx = (self.next_idx + 1) % self.size

    def sample(self, batch_size):

        if self.num_in_buffer < batch_size:
            assert('Replay buffer does not has enough data to sample')

        indices = np.random.choice(np.arange(self.num_in_buffer-1), batch_size, replace = False)

        obs_batch = np.array([self._encode_obs(idx) for idx in indices])
        action_batch = self.actions[indices]
        reward_batch = self.reward[indices]
        done_mask_batch = self.done_mask[indices]
        next_obs_batch = np.array([self._encode_obs((idx+1)%self.size) for idx in indices])

        return obs_batch, action_batch, reward_batch, done_mask_batch, next_obs_batch

    def _encode_obs(self, idx):

        end_idx = (idx + 1)%(self.size)
        start_idx = end_idx - self.history_length
        obs_list = []

        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0

        for i in range(start_idx, end_idx):
            if self.done_mask[i] == True:
                start_idx = i + 1

        missing_context = self.history_length - (end_idx - start_idx)
        if missing_context != 0:
            for _ in range(missing_context):
                obs_list.append(np.zeros(self.observation_dims))

        for i in range(start_idx, end_idx):
            obs_list.append(self.obs[i])

        return np.concatenate(obs_list, -1)

    def encode_recent_obs(self):

        return np.expand_dims(self._encode_obs(self.idx), 0)