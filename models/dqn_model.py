import numpy as np
import torch
import torch.nn as nn

from models.dense import DenseModel

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class Model(nn.Module):
    def __init__(self,
                 action_shape,
                 obs_shape,
                 action_dist='one_hot',
                 qf_shape=(1,),
                 qf_layers=2,
                 qf_hidden=64,
                 dtype=torch.float,
                 device=torch.device('cpu'),
                 q_update_tau=0.005):
        super().__init__()
        self.action_shape = action_shape
        self.action_size = np.prod(self.action_shape)
        self.obs_shape = obs_shape
        self.obs_size = np.prod(self.obs_shape)
        self.qf1_model = DenseModel(self.obs_size + self.action_size, qf_shape, qf_layers,
                                    qf_hidden, dist='normal')
        self.qf2_model = DenseModel(self.obs_size + self.action_size, qf_shape, qf_layers,
                                    qf_hidden, dist='normal')
        self.target_qf1_model = DenseModel(self.obs_size + self.action_size, qf_shape,
                                           qf_layers, qf_hidden, dist='normal')
        self.target_qf2_model = DenseModel(self.obs_size + self.action_size, qf_shape,
                                           qf_layers, qf_hidden, dist='normal')
        self.device = device
        self.dtype = dtype
        self.q_update_tau = q_update_tau

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

    def policy(self, state):
        all_actions = np.zeros((self.action_size, self.action_size))
        for i in range(all_actions.shape[0]):
            all_actions[i,i] = 1
        all_actions = all_actions.tolist()
        q_values = []

        size = state.size()[:-1] + (self.action_size,)
        q = list()
        for action in all_actions:
            action = torch.tensor(action).to(self.device).type(self.dtype)
            action = torch.zeros(size=size).to(state.device) + action
            combined_feat = torch.cat([state, action], dim=-1)
            q += [self.qf1_model(combined_feat).mean]
        q = torch.stack(q, dim=0)
        index = torch.argmax(q, dim=0)
        action = torch.zeros(size=size).to(self.device)
        lead_dim = state.dim() - 1
        assert lead_dim in (0, 1, 2)
        if lead_dim == 0:
            action[index] = 1.0
        elif lead_dim == 1:
            for i in range(state.size(0)):
                action[i, index[i].item()] = 1.0
        elif lead_dim == 2:
            for i in range(state.size(0)):
                for j in range(state.size(1)):
                    action[i, j, index[i, j].item()] = 1.0
        return action