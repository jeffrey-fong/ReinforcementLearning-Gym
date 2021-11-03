import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

class DenseModel(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int,
                 hidden_size: int, dist='no_dist', activation=nn.ReLU):
        super().__init__()
        self.feature_size = feature_size
        self.output_shape = output_shape
        self.layers = layers
        self.hidden_size = hidden_size
        self.dist = dist
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self.feature_size, self.hidden_size)]
        model += [self.activation]
        for i in range(self.layers - 1):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation]
        model += [nn.Linear(self.hidden_size, int(np.prod(self.output_shape)))]
        return nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x)
        if self.dist == 'no_dist':
            return output
        elif self.dist == 'categorical':
            return td.one_hot_categorical.OneHotCategorical(logits=x)