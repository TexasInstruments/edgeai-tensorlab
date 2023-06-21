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


class SEModule1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.SiLU(),
            nn.Conv2d(in_channels= 32, out_channels= 16,kernel_size=3,),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        return torch.mul(self.sequence(x),x)


class InstaModule(nn.Module):
    def __init__(self,preDefinedLayer:nn.Module) -> None:
        super().__init__()
        self.model=preDefinedLayer

    def forward(self,x):
        return self.model(x)


class Focus(nn.Module):
    def forward(self,x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x


class ConvBNRModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding) -> None:
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=nn.ReLU()
    
    def forward(self,x,*args):
        return self.act(self.bn(self.conv(x)))

