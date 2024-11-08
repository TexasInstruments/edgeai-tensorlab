# Copyright (c) 2018-2021, Texas Instruments
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

import copy
import torch
from torch.fx import symbolic_trace, GraphModule
from torch import nn
from mmcv.cnn import bricks
import types
import edgeai_torchmodelopt
from torch.fx import GraphModule
from torch.fx.passes.utils.source_matcher_utils import SourcePartition


def wrap_fn_for_bbox_head(fn, module:nn.Module, *args, **kwargs):
    if hasattr(module,'new_bbox_head'):
        module.new_bbox_head = fn(module.new_bbox_head,  *args, **kwargs)
    else:
        new_bbox_head = fn(module, *args, **kwargs)
        if new_bbox_head is not module:    
            module.add_module('new_bbox_head',new_bbox_head)
            def new_forward(self, x: tuple[torch.Tensor]) -> tuple[list]:
                return self.new_bbox_head(x)
            module.forward = types.MethodType(new_forward, module)
            if isinstance(new_bbox_head,GraphModule):
                params = dict(module.named_parameters())
                for key in params:
                    if key.startswith("new_bbox_head."):
                        continue
                    split = key.rsplit('.',1)
                    if len(split) == 1:
                        param_name = split[0]
                        delattr(module,param_name)
                    else:
                        parent_module, param_name = split
                        main_module = parent_module.split('.',1)[0]
                        if hasattr(module, main_module):
                            delattr(module,main_module)
        else:
            module = new_bbox_head
    return module    


def get_input(model, cfg, batch_size=None, to_export=False):
    image_size = None
    if hasattr(cfg, 'image_size'):
        image_size = cfg.image_size
    elif hasattr(cfg, 'img_scale'):
        image_size = cfg.img_scale
    elif hasattr(cfg, 'input_size'):
        image_size = cfg.input_size
    if image_size is None:
        image_size = 512
    if not isinstance(image_size, (list,tuple)):
        image_size = (image_size, image_size)
    
    batch_size = batch_size or cfg.train_dataloader.batch_size
    x = torch.rand(batch_size, 3, *image_size)
    x = x.to(device='cpu' if to_export else next(model.parameters()).device)
    example_inputs = [x]
    example_kwargs = {}
    return example_inputs, example_kwargs

def get_replacement_dict(model_surgery_version, cfg):
    from mmdet.models.backbones.csp_darknet import Focus, FocusLite
    if hasattr(cfg,'convert_to_lite_model') : 
        convert_to_lite_model_args = copy.deepcopy(cfg.convert_to_lite_model)
        convert_to_lite_model_args.pop('model_surgery', None)
    else:
        convert_to_lite_model_args = dict()

    if model_surgery_version == 1:
        replacement_dict = copy.deepcopy(edgeai_torchmodelopt.xmodelopt.surgery.v1.get_replacement_dict_default(**convert_to_lite_model_args))
        replacements_ext = {
            'mmdet_focus_to_focus_lite':{Focus:[FocusLite, 'in_channels', 'out_channels', 'kernel_size', 'stride']},
            'mmdet_swish_to_relu':{ bricks.Swish:[nn.ReLU]},
            'mmdet_maxpool2d_to_sequential_maxpool2d':{nn.MaxPool2d:[replace_maxpool2d]}
        }
        replacement_dict.update(replacements_ext)
    
    elif model_surgery_version == 2:
        replacement_dict = copy.deepcopy(edgeai_torchmodelopt.xmodelopt.surgery.v2.get_replacement_flag_dict_default())
        replacements_ext = {
            'mmdet_focus_to_focus_lite':{'focus':replace_focus_with_focus_lite},
            'mmdet_swish_to_relu':{bricks.Swish():nn.ReLU}, # as this swish is not a nn Module it gets traced through in symbolic traced in v2
            'mmdet_maxpool2d_to_sequential_maxpool2d':{'MaxPool2d':replace_maxpool2d_k_size_gt_3}
        }
        replacement_dict.update(replacements_ext)
    
    elif model_surgery_version == 3:
        replacement_dict = copy.deepcopy(edgeai_torchmodelopt.xmodelopt.surgery.v3.get_replacement_flag_dict_default())
        replacements_ext = {
            'mmdet_focus_to_focus_lite':{Focus:gen_func_for_focus},
            'mmdet_swish_to_relu':{bricks.Swish:nn.ReLU},
            'mmdet_maxpool2d_to_sequential_maxpool2d':{nn.MaxPool2d:gen_func_for_maxpool_k_size_gt_3}
        }
        replacement_dict.update(replacements_ext)
    else:
        replacement_dict = {}

    return replacement_dict


def replace_maxpool2d(m):
    from mmdet.models.backbones.csp_darknet import SequentialMaxPool2d
    if m.kernel_size > 3:
        new_m = SequentialMaxPool2d(m.kernel_size, m.stride)
    else:
        new_m = m
    #
    return new_m

def gen_func_for_maxpool_k_size_gt_3(main_module: GraphModule, partition: SourcePartition, aten_graph: bool = True):
    if partition.source != nn.MaxPool2d:
        return None
    
    if aten_graph:
        if partition.output_nodes[0].target != torch.ops.aten.max_pool2d.default:
            return None
        maxpool_node = partition.output_nodes[0]
        kernel_size = maxpool_node.args[1]
        module = None
    else:
        modules = dict(main_module.named_modules())
        module = modules.get(partition.output_nodes[0].target,None)
        assert isinstance(module,nn.MaxPool2d)
        kernel_size = module.kernel_size
    if isinstance(kernel_size,(tuple,list)):
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
    module = module or nn.MaxPool2d(kernel_size)
    return replace_maxpool2d(module) if kernel_size > 3 else None


def gen_func_for_focus(main_module: GraphModule, partition: SourcePartition, aten_graph: bool = True):
    from mmdet.models.backbones.csp_darknet import Focus, FocusLite
    if partition.source != Focus:
        return None
    
    if aten_graph:
        conv_node = None
        for node in partition.nodes:
            if node.op != 'call_function':
                continue
            if node.target == torch.ops.aten.conv2d.default:
                conv_node = node
                break
        if conv_node is None:
            return None
        params = dict(main_module.named_parameters())
        weight = params.get(conv_node.args[1].target)
        assert len(weight.shape) == 4
        out_channels, in_channels, *kernel_size = list(weight.shape)
        if len(conv_node.args) >= 4:
            stride = conv_node.args[3]
        module = None
    else:
        modules = dict(main_module.named_modules())
        for node in partition.nodes:
            if node.op != 'call_module':
                continue
            module = modules.get(node.target,None)
            if isinstance(module,nn.Conv2d):
                break
        if not isinstance(module,nn.Conv2d):
            return None
        kernel_size = module.kernel_size 
        stride = module.stride 
        in_channels = module.in_channels
        out_channels = module.out_channels
    if isinstance(kernel_size,(tuple,list)):
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
    if isinstance(stride,(tuple,list)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    
    return FocusLite(in_channels//4, out_channels,kernel_size,stride)

def replace_focus_with_focus_lite(model:nn.Module, verbose_mode=False, **kwargs):
    from mmdet.models.backbones.csp_darknet import Focus, FocusLite
    traced_m=symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    pattern_m= Focus(3, 12, 3, 1)
    traced_pattern= symbolic_trace(pattern_m)
    matches= edgeai_torchmodelopt.surgery.v2.replacer.straight_chain_searcher(traced_m,traced_pattern)
    modules = dict(traced_m.named_modules())
    num_focus = 0
    for start,end in matches:
        node = start
        found_conv = False
        while node != end:
            if node.op == 'call_module' and isinstance((module := modules[node.target]),nn.Conv2d):
                found_conv = True
            if found_conv:
                break
            node = node.next
        if not found_conv:
            continue
        
        if isinstance(module.kernel_size, tuple) and len(module.kernel_size)==2:
            assert module.kernel_size[0] == module.kernel_size[1], "In Focus module, kernel_size must be a square" 
            kernel_size = module.kernel_size[0]
        else:
            kernel_size = module.kernel_size
        
        device = next(iter(module.parameters())).device
        replacement = FocusLite(module.in_channels//4, module.out_channels, kernel_size, module.stride)
        replacement =replacement.to(device)
        traced_m.add_module(f'replaced_focus_{num_focus}',replacement)
        args = (start.args[0],)
        with traced_m.graph.inserting_before(start):
            new_node= traced_m.graph.call_module(f'replaced_focus_{num_focus}',args,{})
            end.replace_all_uses_with(new_node)
        num_focus += 1
    
    edgeai_torchmodelopt.surgery.v2.replacer._remove_hanging_nodes(traced_m)
    traced_m.graph.lint()
    traced_m.recompile()
    if verbose_mode:
        print('Focus',len(matches))
    return traced_m

def replace_maxpool2d_k_size_gt_3(model:nn.Module, verbose_mode=False, **kwargs):
    
    traced_model = symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    modules=dict(traced_model.named_modules())
    
    no_of_pool=0
    for node in traced_model.graph.nodes:
        if node.op == 'call_module' and isinstance((module := modules[node.target]),nn.MaxPool2d):
            #for call module pool
                if module.kernel_size >4:
                    k_size=module.kernel_size 
                    stride=module.stride
                    padding=module.padding
                    replacement= nn.Sequential()
                    while k_size > 3:
                        # if k_size % 2 ==0: replacement.append(pool_class(kernel_size=2,stride=1,padding=(0,0,1,1)))
                        if k_size % 2 == 0:
                            replacement.append(nn.MaxPool2d(kernel_size=2, stride=1, padding=(1,1)))
                        else: replacement.append(nn.MaxPool2d(kernel_size=3,stride=1,padding=1))
                        k_size-=2
                    # replacement.append(pool_class(kernel_size=k_size,stride=stride,padding=1 if padding %2 !=0 else (0,0,1,1)))
                    replacement.append(
                        nn.MaxPool2d(kernel_size=k_size, stride=stride, padding=1 if padding % 2 != 0 else (1,1)))
                    edgeai_torchmodelopt.surgery.v2.replacer._replace_pattern(traced_model,node,node,replacement,no_of_pool)
                    no_of_pool+=1
        
        if node.target == nn.functional.max_pool2d:
            #for functional pool
            k_size=node.args[1]
            stride=node.kwargs['stride']
            padding=node.kwargs['padding']
            replacement= nn.Sequential()
            if k_size>4:
                while k_size > 3:
                    # if k_size % 2 ==0: replacement.append(pool_class(kernel_size=2,stride=1,padding=(0,0,1,1)))
                    if k_size % 2 == 0:
                        replacement.append(nn.MaxPool2d(kernel_size=2, stride=1, padding=(1,1)))
                    else: replacement.append(nn.MaxPool2d(kernel_size=3,stride=1,padding=1))
                    k_size-=2
                # replacement.append(pool_class(kernel_size=k_size,stride=stride,padding=1 if padding %2 !=0 else (0,0,1,1)))
                replacement.append(
                    nn.MaxPool2d(kernel_size=k_size, stride=stride, padding=1 if padding % 2 != 0 else (1,1)))
                new_node_name=f'replaced_maxpool2d_{no_of_pool}'
                traced_model.add_submodule(new_node_name,replacement)
                args=(node.args[0],)
                with traced_model.graph.inserting_before(node):
                    new_node=traced_model.graph.call_module(new_node_name,args,{})
                    node.replace_all_uses_with(new_node)
                traced_model.graph.erase_node(node)
        
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print(f'maxpool2d',no_of_pool)
    return traced_model
