# Copyright (c) 2018-2025, Texas Instruments
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


import torch
import onnx_graphsurgeon as gs
from . import utils
import warnings


class BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        pass


def add_batchnorm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 5, f'{node.name} with operator {node.op} should have 5 input, but got {len(node.inputs)}'
    assert all(isinstance(x, gs.Constant) for x in node.inputs[1:]), f'node {node.name} with operator {node.op} should have weight, bias, running_mean, running_var as constant but got {[type(x).__name__ for x in node.inputs[1:]]}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, utils.Buffer, utils.Buffer]
    inp, weight, bias, mean, var = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    momentum = 1 - node.attrs.get('momentum', 0.9)
    training_mode = node.attrs.get('training_mode', 0) == 1
    # if training_mode:
    #     raise NotImplementedError(f'node {node.name} with operator {node.op} in training mode is not implemented yet')
    num_outputs = len(node.outputs)
    for i in range(-1, -num_outputs-1, -1):
        i = i+num_outputs
        out = node.outputs[i]
        if i<1:
            continue
        if out.outputs:
            raise NotImplementedError(f'node {node.name} with operator {node.op} has multiple outputs which are used further in the model, but not implemented yet')
        node.outputs.remove(out)
        node.attrs['training_mode'] = 0
        if out in state.graph.outputs:
            state.graph.outputs.remove(out)
            warnings.warn(f'{out.name} output is removed from the graph because it is not used in model and needed to be removed for {node.name}({node.op}) ')
    if len(node.outputs) != 1:
        raise NotImplementedError(f'node {node.name} with operator {node.op} has multiple outputs which are used further in the model, but not implemented yet')
    
    module = BatchNorm(num_features=node.inputs[1].values.shape[0], eps=epsilon, momentum=momentum)
    module.weight = getattr(torch_module, weight.target)
    module.bias = getattr(torch_module, bias.target)
    module.running_mean = getattr(torch_module, mean.target)
    module.running_var = getattr(torch_module, var.target)
    module.training = training_mode
    torch_module.add_module(node.name, module)
    torch_nodes[node.name] = torch_graph.call_module(node.name,(inp,))

class InstanceNorm(torch.nn.modules.instancenorm._InstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() not in (3,4,5):
            raise ValueError(
                f"expected 2D, 3D, 4D or 5D input (got {input.dim()}D input)"
            )
    def _get_no_batch_dim(self):
        pass
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        feature_dim = 1
        if input.size(feature_dim) != self.num_features:
            if self.affine:
                raise ValueError(
                    f"expected input's size at dim={feature_dim} to match num_features"
                    f" ({self.num_features}), but got: {input.size(feature_dim)}."
                )
            else:
                warnings.warn(
                    f"input's size at dim={feature_dim} does not match num_features. "
                    "You can silence this warning by not passing in num_features, "
                    "which is not used because affine=False"
                )

        return self._apply_instance_norm(input)
def add_instance_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 input, but got {len(node.inputs)}'
    assert all(isinstance(x, gs.Constant) for x in node.inputs[1:]), f'node {node.name} with operator {node.op} should have weight, bias as constant but got {[type(x).__name__ for x in node.inputs[1:]]}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]
    inp, weight, bias = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    module = InstanceNorm(num_features=node.inputs[1].values.shape[0], eps=epsilon,)
    module.training =False
    module.weight = getattr(torch_module, weight.target)
    module.bias = getattr(torch_module, bias.target)
    torch_module.add_module(node.name, module)
    torch_nodes[node.name] = torch_graph.call_module(node.name,(inp,))

def add_layer_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 2<=len(node.inputs) <= 3, f'{node.name} with operator {node.op} should have 3 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    epsilon = node.attrs.get('epsilon', 1e-5)
    axis = node.attrs.get('axis', -1)
    num_outputs = len(node.outputs)
    for i in range(-1, -num_outputs-1, -1):
        i = i+num_outputs
        out = node.outputs[i]
        if i<1:
            continue
        if out.outputs:
            raise NotImplementedError(f'node {node.name} with operator {node.op} has multiple outputs which are used further in the model, but not implemented yet')
        node.outputs.remove(out)
        if out in state.graph.outputs:
            state.graph.outputs.remove(out)
            warnings.warn(f'{out.name} output is removed from the graph because it is not used in model and needed to be removed for {node.name}({node.op}) ')
    if len(node.outputs) != 1:
        raise NotImplementedError(f'node {node.name} with operator {node.op} has multiple outputs which are used further in the model, but not implemented yet')
    bias  = len(node.inputs) == 3
    if not bias:
        args.append(None)
    kwargs = dict(
        # num_groups = num_groups,
        eps = epsilon,
        bias = bias

    )
    inp, weight, bias = args
    module = torch.nn.LayerNorm(normalized_shape=node.inputs[1].values.shape[0], **kwargs)
    module.weight = getattr(torch_module, weight.target)
    if bias:
        module.bias = getattr(torch_module, bias.target)
    torch_module.add_module(node.name, module)
    torch_nodes[node.name] = torch_graph.call_module(node.name,(inp,))

def add_group_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 2<=len(node.inputs) <= 3, f'{node.name} with operator {node.op} should have 2 to 3 input, but got {len(node.inputs)}'
    assert all(isinstance(x, gs.Constant) for x in node.inputs[1:]), f'node {node.name} with operator {node.op} should have weight, bias as constant but got {[type(x).__name__ for x in node.inputs[1:]]}'
    types = [torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]
    inp, weight, bias = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    num_groups = node.attrs.get('num_groups')
    epsilon = node.attrs.get('epsilon', 1e-5)
    kwargs = dict(
        eps = epsilon,
    )

    module = torch.nn.GroupNorm(num_groups=num_groups, num_channels=node.inputs[1].values.shape[0], **kwargs)
    module.weight = getattr(torch_module, weight.target)
    module.bias = getattr(torch_module, bias.target)
    torch_module.add_module(node.name, module)
    torch_nodes[node.name] = torch_graph.call_module(node.name,(inp,))

def add_lp_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_mean_variance_norm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")
    
