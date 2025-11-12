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
from operator import getitem
from onnx import TensorProto
import numpy as np
import copy

class Buffer(torch.Tensor):
    pass

class WrappedModule(torch.nn.Module):
    def __init__(self, node_name, onnx_op, parent_module:torch.nn.Module, func, args=None, kwargs=None,):
        super().__init__()
        self.node_name = node_name
        self.onnx_op = onnx_op
        self.func = func
        args = args or []
        args = list(args)
        kwargs = kwargs or {}
        self.args = args.copy()
        for i, arg in enumerate(self.args):
            if isinstance(arg, torch.fx.Node) and arg.op == 'get_attr':
                param = getattr(parent_module, arg.target)
                if arg.target in list(parent_module.state_dict().keys()):
                    if isinstance(param, torch.nn.Parameter):
                        param = torch.nn.Parameter(param.data, param.requires_grad)
                        self.register_parameter(arg.target, param)
                    else:
                        self.register_buffer(arg.target, param)
                else:
                    setattr(self, arg.target, param)
        self.variable_indices = [i for i, arg in enumerate(args) if isinstance(arg, torch.fx.Node) and arg.op != 'get_attr']
        self.args = [a.target if isinstance(a, torch.fx.Node) else copy.deepcopy(a) for a in self.args]
        for i,arg in enumerate(self.args):
            if isinstance(arg, str):
                continue
            self.args[i] = f'arg_{i}'
            if isinstance(arg, torch.Tensor):
                self.register_buffer(self.args[i], arg)
            else:
                setattr(self, self.args[i], arg)
        self.kwargs = kwargs

    def forward(self, *args,):
        temp_args = self.args.copy()
        if temp_args:
            assert len(args) == len(self.variable_indices), f"Expected {len(self.variable_indices)} arguments but got {len(args)}"
            var_arg_counter = 0
            for i, arg in enumerate(temp_args):  
                if i in self.variable_indices:
                    temp_args[i] = args[var_arg_counter]
                    var_arg_counter+=1
                else:
                    if isinstance(arg, str) and hasattr(self, arg):
                        temp = getattr(self, arg) 
                        if hasattr(temp, 'copy'):
                            temp = temp.copy()
                        temp_args[i] = temp
                    else:
                        temp_args[i] = arg
            return self.func(*temp_args, **self.kwargs)
        else:
            return self.func(*args, **self.kwargs)
    
    def __repr__(self):
        return f"WrappedModule(node_name={self.node_name}, op={self.onnx_op},func={self.func.__name__}, args = {[getattr(self,a) if hasattr(self, a) else a for a in self.args]},  kwargs="+ r'{' + ', '.join([f'{k} = {v}' for k, v in self.kwargs.items()]) + r'})'

def get_input_from_node(node:gs.Node, inp:gs.Variable|gs.Constant, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module, attr_type:type=None, **kwargs):
    if inp.name in torch_nodes and  torch_nodes[inp.name].op != 'get_attr':
        return torch_nodes[inp.name]
    if isinstance(inp, gs.Constant):
        name = inp.name + f"_{len([k for k,v in torch_nodes.items() if v.op =='get_attr'])}"
        values=inp.values
        temp = gs.Constant(name, values, inp.data_location)
        i = node.inputs.index(inp)
        node.inputs[i] = temp
        inp = temp
        attr_type = attr_type or torch.Tensor
        if attr_type not in (list, tuple, set, torch.Tensor, torch.nn.Parameter, Buffer):
            raise ValueError(f"Unsupported type {attr_type}")
        val = inp.values
        if attr_type == list:
            val = val.tolist()
        elif attr_type in (tuple, set):
            val = attr_type(val)
        if attr_type in (torch.Tensor, torch.nn.Parameter, Buffer):
            val = torch.from_numpy(val)
        try:
            if inp.dtype not in (np.float32,np.float16, np.float64,np.complex64, np.complex128) and attr_type in (torch.nn.Parameter, Buffer):
                attr_type = torch.Tensor
            if attr_type in (list, tuple, set):
                setattr(torch_module, inp.name, val)
            elif attr_type in  (torch.Tensor, Buffer):
                torch_module.register_buffer(inp.name, val,)
            elif attr_type == torch.nn.Parameter:
                val = attr_type(val)
                torch_module.register_parameter(inp.name, val)
        except Exception as e:
            raise ValueError(f"Failed to register {inp.name} in the root module with Exception {e}. # Fix it")
        torch_nodes[inp.name] = torch_graph.get_attr(inp.name)
        return torch_nodes[inp.name]
    if inp.shape is None and inp.dtype is None and len(inp.inputs) == 0:
        return None
    if len(inp.inputs) == 0:
        # graph input case 
        # already handled in get_torch_graph_module
        raise ValueError (f"Graph input {inp.name} not handled")
    inp_node = inp.inputs[0]
    if len(inp_node.outputs) == 1:
        return torch_nodes[inp_node.name]
    else:
        index = inp_node.outputs.index(inp)
        torch_nodes[inp.name] = torch_graph.call_function(getitem, tuple([torch_nodes[inp_node.name], index]), name=inp.name)
        return torch_nodes[inp.name]


onnx_2_torch_type_mapping = {
    TensorProto.FLOAT16 : torch.float16,
    TensorProto.FLOAT : torch.float32,
    TensorProto.DOUBLE : torch.float64,
    TensorProto.INT8 : torch.int8,
    TensorProto.UINT8 : torch.uint8,
    TensorProto.INT16 : torch.int16,
    TensorProto.UINT16 : torch.uint16,
    TensorProto.INT32 : torch.int32,
    TensorProto.UINT32 : torch.uint32,
    TensorProto.INT64 : torch.int64,
    TensorProto.UINT64 : torch.uint64,
    TensorProto.BOOL : torch.bool,
    TensorProto.BFLOAT16: torch.bfloat16,
    TensorProto.COMPLEX64: torch.complex64,
    TensorProto.COMPLEX128: torch.complex128,
}
np_2_torch_type_mapping = {
    np.float16 : torch.float16,
    np.float32 : torch.float32,
    np.float64 : torch.float64,
    np.int8 : torch.int8,
    np.uint8 : torch.uint8,
    np.int16 : torch.int16,
    np.uint16 : torch.uint16,
    np.int32 : torch.int32,
    np.uint32 : torch.uint32,
    np.int64 : torch.int64,
    np.uint64 : torch.uint64,
    np.bool_ : torch.bool,
    np.complex64 : torch.complex64,
    np.complex128 : torch.complex128,
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('int8'): torch.int8,
    np.dtype('uint8'): torch.uint8,
    np.dtype('int16'): torch.int16,
    np.dtype('uint16'): torch.uint16,
    np.dtype('int32'): torch.int32,
    np.dtype('uint32'): torch.uint32,   
    np.dtype('int64'): torch.int64,
    np.dtype('uint64'): torch.uint64,
    np.dtype('bool'): torch.bool,
    np.dtype('complex64'): torch.complex64,
    np.dtype('complex128'): torch.complex128,
    None:None
}