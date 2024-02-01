#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

from torch import nn
from torch.fx import symbolic_trace 
import copy
import math
import torch

from edgeai_torchmodelopt import xmodelopt


class AddModule(nn.Module):
    def forward(self, x, y):
        return x + y

class MultModule(nn.Module):
    def forward(self, x, y):
        return x * y

import operator

#replacing a add function node with a mul function node (Default Way)
def replace_add_function(m:nn.Module):
    traced_m =  symbolic_trace(m)
    pattern_m = symbolic_trace(AddModule())

    #finding the nodes having target as operator.add or '+' operator 
    matches= xmodelopt.surgery.replacer.straight_chain_searcher(traced_m,pattern_m)

    #replace using call_function
    for start,end in matches:
        with traced_m.graph.inserting_after(start):
            new_node= traced_m.graph.call_function(operator.mul,start.args,start.kwargs) 
            start.replace_all_uses_with(new_node)
        traced_m.graph.erase_node(start)

    traced_m.recompile()
    return m

#replacing a add function node with a mul function node (using a wrapper module)
def replace_add_module(m:nn.Module):
    traced_m =  symbolic_trace(m)
    pattern_m = symbolic_trace(AddModule())
    replace_m = MultModule()
    
    #finding the nodes having target as operator.add or '+' operator 
    matches= xmodelopt.surgery.replacer.straight_chain_searcher(traced_m,pattern_m)
    main_modules=dict(traced_m.named_modules())

    # replace using call_module
    i=0
    for start,end in matches:
        replacement=copy.deepcopy(replace_m)
        traced_m.add_module(f'replace{i}',replacement)
        with traced_m.graph.inserting_before(start):
            new_node= traced_m.graph.call_module(f'replace{i}',start.args,start.kwargs)
            start.replace_all_uses_with(new_node)
        traced_m.graph.erase_node(start)
        main_modules.update({f'replace{i}':replacement})
        i+=1
    traced_m.recompile()
    return m

#replacing conv2d module with kernel size 5X5 with two conv2d modules with kernel size 3X3 along with a normalization
def replace_conv2d(m:nn.Module):
    traced_m =  symbolic_trace(m)
    pattern_m= xmodelopt.surgery.custom_module.InstaModule(nn.Conv2d(3,16,5))
    replace_m= nn.Sequential(
                                nn.Conv2d(5,8,3),
                                nn.BatchNorm2d(8),
                                nn.Conv2d(8,20,1)
                             )
    traced_pattern= symbolic_trace(pattern_m)
    matches= xmodelopt.surgery.replacer.straight_chain_searcher(traced_m,traced_pattern)
    main_modules=dict(traced_m.named_modules())
    #filter Out the proper matched one and replace all filtered matches
    for start,end in matches:
        if main_modules[start.target].kernel_size==5:
            replacement=copy.deepcopy(replace_m)
            #after adjustment
            main_in_channels = main_modules[start.target].in_channels
            main_out_channels = main_modules[end.target].out_channels
            replacement[0].in_channels=main_in_channels
            replacement[2].out_channels=main_out_channels
            replacement[2].in_channels=replacement[1].num_features=replacement[0].out_channels=math.floor(main_in_channels*(8/5))
            parent_name,name= xmodelopt.surgery.replacer._get_parent_name(start.target)
            main_modules[parent_name].__setattr__(name,replacement)
            main_modules.update({start.target,replacement})
    traced_m.recompile()
    return traced_m

class CustomModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.acts = nn.Sequential(
            nn.ReLU(),
            nn.ReLU6(),
            nn.ReLU(),
            nn.ReLU6(),
            nn.ReLU(),
            nn.ReLU6(),
        )
    def forward(self,x):
        return torch.relu(self.acts(x))
    
pattern_type= [nn.ReLU,nn.ReLU6,torch.relu]
model = symbolic_trace(CustomModule())
print(model)
matches = xmodelopt.surgery.v2.replacer.straight_type_chain_searcher(model,pattern_type)
print(len(matches))

xmodelopt.surgery.v2.replacer._replace_all_matches(model,matches,nn.Identity())
print(model)