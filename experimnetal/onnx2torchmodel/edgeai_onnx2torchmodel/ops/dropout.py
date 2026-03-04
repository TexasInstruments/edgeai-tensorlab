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

def torch_dropout(x, ratio=0.5, training=True,):
    return torch.nn.functional.dropout(x, p=ratio, training=training)

torch.nn.Dropout()

def add_dropout_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
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
    module = torch.nn.Dropout()
    torch_module.add_module(node.name, module)
    if state.graph.opset>=12:
        assert 1<= len(node.inputs) <= 3, f'{node.name} with operator {node.op} should have 1 to 3 inputs in opset (12 and above), but got {len(node.inputs)}'
        types = [torch.nn.Parameter, list, list]
        args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
        seed = node.attrs.get('seed', None)
        if len(args) == 1:
            args.append(0.5)
        if len(args) == 2:
            args.append(False)
        inp, ratio, training_mode = args
        if len(node.inputs)<=2:
            assert isinstance(node.inputs[1], gs.Constant), f'node {node.name} with operator {node.op} should have ratio as constant but got {type(node.inputs[1])}'
        if len(node.inputs) == 3:
            assert isinstance(node.inputs[2], gs.Constant), f'node {node.name} with operator {node.op} should have training_mode as constant but got {type(node.inputs[1])}'
        module.p = ratio
        module.training = training_mode
        torch_nodes[node.name] = torch_graph.call_module(node.name, (inp,),  name=node.name)
        torch_nodes[node.name].meta['seed'] = seed
    elif  state.graph.opset>=7:
        assert len(node.inputs)== 1 , f'{node.name} with operator {node.op} should have 1 input in opset (7 to 11), but got {len(node.inputs)}'
        types = [torch.nn.Parameter]
        args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
        ratio = node.attrs.get('ratio', 0.5)
        module.p = ratio
        torch_nodes[node.name] = torch_graph.call_module(node.name, (inp,),  name=node.name)
        torch_nodes[node.name].meta['seed'] = seed
    else:
        assert len(node.inputs)== 1 , f'{node.name} with operator {node.op} should have 1 input in opset (7 to 11), but got {len(node.inputs)}'
        types = [torch.nn.Parameter]
        args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
        ratio = node.attrs.get('ratio', 0.5)
        training_mode = node.attrs.get('is_test', 0) == 0
        module.p = ratio
        module.training = training_mode
        torch_nodes[node.name] = torch_graph.call_module(node.name, (inp,),  name=node.name)
        torch_nodes[node.name].meta['seed'] = seed