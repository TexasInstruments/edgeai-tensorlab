import torch
from torch import nn 
class SEModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32, out_channels= 16,kernel_size=3,),
            nn.Hardsigmoid()
        )
    
    def forward(self,x):
        return torch.mul(self.sequence(x),x)

class InstaModule(nn.Module):
    def __init__(self,preDefinedLayer:nn.Module) -> None:
        super().__init__()
        self.model=preDefinedLayer
    def forward(self,x):
        return self.model(x)