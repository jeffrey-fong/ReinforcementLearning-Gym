import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, obs_shape, action_shape, reward_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # Initialize all the buffers
        self.obs_buffer = np.empty(shape=(capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buffer = np.empty(shape=(capacity, *obs_shape), dtype=np.float32)
        self.action_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.done_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.idx = 0

    def add(self, obs, action, reward, next_obs, done):
        if self.idx < self.capacity:
            self.obs_buffer[self.idx] = obs
            self.next_obs_buffer[self.idx] = next_obs
            self.action_buffer[self.idx] = action
            self.reward_buffer[self.idx] = reward
            self.done_buffer[self.idx] = done
            self.idx += 1
        else:
            self.obs_buffer = self.obs_buffer[1:]
            self.obs_buffer = np.append(self.obs_buffer,
                                        obs.reshape((1, obs.shape[0])),
                                        axis=0)
            self.next_obs_buffer = self.next_obs_buffer[1:]
            self.next_obs_buffer = np.append(self.next_obs_buffer,
                                        next_obs.reshape((1, next_obs.shape[0])),
                                        axis=0)
            self.action_buffer = self.action_buffer[1:]
            self.action_buffer = np.append(self.action_buffer,
                                           action.reshape((1, action.shape[0])),
                                           axis=0)
            self.reward_buffer = self.reward_buffer[1:]
            self.reward_buffer = np.append(self.reward_buffer,
                                           reward.reshape((1, reward.shape[0])),
                                           axis=0)
            self.done_buffer = self.done_buffer[1:]
            self.done_buffer = np.append(self.done_buffer,
                                         done.reshape((1, done.shape[0])),
                                         axis=0)

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.idx == self.capacity else self.idx, size=self.batch_size)
        obses = torch.as_tensor(self.obs_buffer[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obs_buffer[idxs], device=self.device)
        actions = torch.as_tensor(self.action_buffer[idxs], device=self.device)
        rewards = torch.as_tensor(self.reward_buffer[idxs], device=self.device)
        dones = torch.as_tensor(self.done_buffer[idxs], device=self.device)

        return obses, actions, rewards, next_obses, dones

    def save(self, path=None, level=None):
        if path is None:
            raise NotImplementedError
        else:
            np.save(f'{path}/obs_buffer_{level}.npy', self.obs_buffer)
            np.save(f'{path}/next_obs_buffer_{level}.npy', self.next_obs_buffer)
            np.save(f'{path}/action_buffer_{level}.npy', self.action_buffer)
            np.save(f'{path}/reward_buffer_{level}.npy', self.reward_buffer)
            np.save(f'{path}/done_buffer_{level}.npy', self.done_buffer)

    def load(self, path=None, level=None):
        self.obs_buffer = np.load(f'{path}/obs_buffer_{level}.npy')
        self.next_obs_buffer = np.load(f'{path}/next_obs_buffer_{level}.npy')
        self.action_buffer = np.load(f'{path}/action_buffer_{level}.npy')
        self.reward_buffer = np.load(f'{path}/reward_buffer_{level}.npy')
        self.done_buffer = np.load(f'{path}/done_buffer_{level}.npy')