from typing import Iterable
from torch.nn import Module


def getParameters(modules: Iterable[Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        self.modules = modules
        self.param_states = [p.requires_grad for p in getParameters(self.modules)]

    def __enter__(self):
        for param in getParameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(getParameters(self.modules)):
            param.requires_grad = self.param_states[i]