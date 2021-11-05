import numpy as np
import torch
from tqdm import tqdm
import os
import random

from utils.parameters import getParameters, FreezeParameters
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)


class Algorithm():
    def __init__(self,
                 model,
                 train_steps=1,
                 qf_lr=5e-4,
                 discount=0.99,
                 target_update_period=1,
                 type=torch.float,
                 device='cpu',
                 save_every=5):
        self.model = model
        self.train_steps = train_steps
        self.qf_lr = qf_lr
        self.discount = discount
        self.target_update_period = target_update_period
        self.type = type
        self.device = device
        self.itr = 0
        self.writer = SummaryWriter()
        self.initOptim()

    def initOptim(self):
        self.critic_modules = [self.model.qf1_model,
                               self.model.qf2_model]
        self.critic_optimizer = torch.optim.Adam(getParameters(self.critic_modules),
                                                 lr=self.qf_lr)

    def optimize(self, model, replay_buffer):
        # self.model = model
        for i in range(self.train_steps):
            obses, actions, rewards, next_obses, dones = replay_buffer.sample()
            self.batch_size = obses.shape[0]
            qf1_loss, qf2_loss = self.critic_loss(obses, actions, rewards, next_obses, dones)
            self.critic_optimizer.zero_grad()
            (qf1_loss + 0.0 * qf2_loss).backward()
            self.critic_optimizer.step()
            self.writer.add_scalar('Loss/critic1', qf1_loss, self.itr)
            self.writer.add_scalar('Loss/critic2', qf2_loss, self.itr)
            self.itr += 1
        return self.model

    def critic_loss(self, obses, actions, rewards, next_obses, dones):
        # Sort out the times (s_t, a_{t-1}, r_t, d_t)
        obs = obses[:].to(self.device).type(self.type).detach()
        action = actions[:].to(self.device).type(self.type).detach()
        reward = rewards[:].to(self.device).type(self.type).detach()
        next_obs = next_obses[:].to(self.device).type(self.type).detach()
        done = dones[:].to(self.device).type(self.type).detach()

        if self.itr % self.target_update_period == 0:
            self.model.update_target_networks()

        feat = torch.cat([obs, action], dim=-1)
        qf1_pred = self.model.qf1_model(feat).mean
        qf2_pred = self.model.qf2_model(feat).mean

        with torch.no_grad():
            next_policy_action = self.model.policy(next_obs)
            next_feat = torch.cat([next_obs, next_policy_action.to(self.device)],
                                  dim=-1)
            # target_value = torch.min(self.model.target_qf1_model(next_feat),
            #                          self.model.target_qf2_model(next_feat))
            target_value = self.model.target_qf1_model(next_feat).mean
            q_target = reward + (1 - done.float()) * self.discount * target_value

        qf1_loss = torch.nn.functional.mse_loss(qf1_pred, q_target.type(self.type))
        qf2_loss = torch.nn.functional.mse_loss(qf2_pred, q_target.type(self.type))

        return qf1_loss, qf2_loss