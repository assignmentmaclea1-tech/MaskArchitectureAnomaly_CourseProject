import torch
from torch import nn, optim
from torch.nn import functional as F

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Modifica per la segmentazione 4D (ignora l'espansione 2D)
        return logits / self.temperature
