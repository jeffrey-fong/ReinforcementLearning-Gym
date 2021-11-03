import torch
import torch.nn as nn
import torch.distributions as td

class DenseModel(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int,
                 hidden_size: int, dist='normal'):
        super().__init__()
        self.feature_size = feature_size
        self.output_shape = output_shape
        self.layers = layers
        self.hidden_size = hidden_size
        self.dist = dist
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self.feature_size, )]