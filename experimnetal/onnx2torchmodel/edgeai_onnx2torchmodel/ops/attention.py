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


import onnx_graphsurgeon as gs
import torch
from . import utils



def torch_attention(q,k,v, attn_mask=None, past_key=None, past_val=None, non_pad_kv_seqlen=None, is_casual=0, kv_num_heads=None,q_num_heads=None, qk_matmul_output_mode=0, scale=1, softcap=0, softmax_precision=None, num_outputs=1):
    assert len(q.shape) in [3, 4], f'q should be 3D or 4D, but got {q.shape}'
    assert len(k.shape) in [3, 4], f'k should be 3D or 4D, but got {q.shape}'
    assert len(v.shape) in [3, 4], f'v should be 3D or 4D, but got {q.shape}'
    k_changed = v_changed = False
    input_dim = len(q.shape)
    def to_4d(x, num_heads):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
            x = x.permute(0, 2, 1, 3)
        return x
    
    if len(q.shape) == 3:
        assert q_num_heads is not None, f'q_num_heads should not be None for #D input  '
        q = to_4d(q, q_num_heads)

    if len(k.shape) == 3:
        assert kv_num_heads is not None, f'kv_num_heads should not be None for #D input  '
        k_changed =True
        k = to_4d(k, kv_num_heads)
        if past_key:
            past_key = to_4d(past_key, kv_num_heads)
    if len(v.shape) == 3:
        assert kv_num_heads is not None, f'kv_num_heads should not be None for #D input  '
        v_changed = True
        v = to_4d(v, kv_num_heads)
        if past_val:
            past_val = to_4d(past_val, kv_num_heads)
    
    q_num_heads = q_num_heads or q.shape[-3]
    kv_num_heads = kv_num_heads or k.shape[-3]
    assert (past_key is None and past_val is None) or (past_key is not None and past_val is not None), f'past_key and past_val should be both None or both not None, but past_key is {past_key} and past_val is {past_val}'
    if non_pad_kv_seqlen is not None and past_key is not None:
        raise ValueError(f'non_pad_kv_seqlen can not be used together with past_key')
    #TODO add support for non_pad_kv_seqlen
    if non_pad_kv_seqlen:
        raise NotImplementedError(f'usage of non_pad_kv_seqlen is not implemented')
    if past_key is not None:
        k = torch.concat([past_key, k], dim=-2)
        v = torch.concat([past_val, v], dim=-2)
    
    q = q*torch.sqrt(scale)
    k_ = k*torch.sqrt(scale)
    k_ = k_.transpose(-2,-1)
    qk = torch.matmul(q,k_)
    if attn_mask is not None:
        #TODO attention mask padding and shape checking
        qk = qk + attn_mask
    if is_casual:
        qk = torch.tril(qk)
    if qk_matmul_output_mode > 0:
        qk_matmul_output = qk
    qk = torch.clip(qk, softcap)
    if qk_matmul_output_mode > 1:  
        qk_matmul_output = qk
    qk = torch.softmax(qk,dim = -1)
    if softmax_precision:
        qk = torch.round(qk, decimals=softmax_precision)
    if qk_matmul_output_mode > 2:
        qk_matmul_output = qk
    
    if qk_matmul_output_mode<0 or qk_matmul_output_mode>3:
        raise ValueError(f'Invalid qk_matmul_output_mode: {qk_matmul_output_mode} should be between 0 and 3')
    
    qkv = torch.matmul(qk, v)
    
    def to_3d(x):
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)
        return x
    
    if input_dim == 3:
        qkv = to_3d(qkv)
    if k_changed:
        k = to_3d(k)
    if v_changed:
        v = to_3d(v)
    
    if num_outputs == 1:
        return qkv
    if num_outputs == 2:
        return qkv, qk_matmul_output
    if past_key is not None and num_outputs> 2:
        return qkv, k, v, qk_matmul_output
    


def add_attention_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert 7>= len(node.inputs) >= 3, f'{node.name} with operator {node.op} should have between 3 and 7 inputs, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    kwargs = dict(node.attrs)
    kwargs['num_outputs'] = len(node.outputs)
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_attention, args, kwargs)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_attention, tuple(args),  kwargs, name=node.name)