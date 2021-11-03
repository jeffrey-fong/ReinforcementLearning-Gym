import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,
                 action_shape,
                 action_dist='one_hot',
                 qf_shape=(1,),
                 qf_layers=3,
                 qf_hidden=128,
                 dtype=torch.float,
                 q_update_tau=0.005):
        super().__init__()
        