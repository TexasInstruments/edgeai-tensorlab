import torch
from torch import rand,nn,fx
import torchvision
from torchvision import models
from replacer import replaceAndExpot,graphPatternReplacer
from utils import exportAndSimplifyOnnx
main_model= models.mobilenet_v3_small()
dummy= rand(1,3,224,224)

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

exportAndSimplifyOnnx(main_model,dummy,'mbv3smallbefore.onnx')
# print(main_model)
traced_model= fx.symbolic_trace(main_model)
pattern_model= SEModule()
replaceModel=nn.Identity()
afterSE2I=graphPatternReplacer(traced_model,pattern_model,replaceModel)
print(afterSE2I)
afterSE2I.graph.print_tabular()
print(dict(afterSE2I.named_modules())['features.4.block.0.2'])
# afterSE2I.graph.print_tabular()
pattern_model= InstaModule(nn.Hardswish())
replaceModel=nn.ReLU()
afterHS2ReLU=graphPatternReplacer(afterSE2I,pattern_model,replaceModel)
pattern_model=nn.ReLU(inplace=True)
afterReLUT2ReLUF=graphPatternReplacer(afterHS2ReLU,pattern_model,nn.ReLU())
pattern_model=nn.Dropout(inplace=True)
afterDOT2DOF=graphPatternReplacer(afterReLUT2ReLUF,pattern_model,nn.Dropout())
exportAndSimplifyOnnx(afterDOT2DOF,dummy,'mbv3smallafter.onnx')

dix={1:2,2:3,3:4,4:5}
dix.pop(1)