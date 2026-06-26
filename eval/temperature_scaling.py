import torch
from torch import nn, optim
from torch.nn import functional as F

'''
"""
Questo modulo implementa la Temperature Scaling, una tecnica di calibrazione usata per migliorare le prestazioni del classifier.
La classe incapsula un modello già addestrato e introduce un parametro scalare apprendibile (temperature) che viene 
usato per ridimensionare i logits in output.
Nel forward, il modello produce i logits e questi vengono divisi per la temperatura, modificando la “confidenza” 
delle predizioni senza alterare le classi previste.
Nel caso della segmentazione, la scaling viene applicata direttamente ai logits 
pixel-wise senza necessità di adattamenti aggiuntivi.
"""
'''

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
