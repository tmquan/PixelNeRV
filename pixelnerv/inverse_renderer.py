import torch.nn as nn

class PixelNeRVFrontToBackInverseRenderer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
       
        
    def forward(self, inputs):
        volumes = self.model(inputs)
        return volumes
        