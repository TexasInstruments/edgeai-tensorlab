import torch
import onnx_graphsurgeon as gs
from operator import getitem
from . import utils

def torch_gemm(a:torch.Tensor, b:torch.Tensor, c:torch.Tensor, alpha=1, beta=1, transA=False, transB=False):
    # return torch.mm
    if transA: 
        a = a.transpose(-2,-1)
    if transB:
        b = b.transpose(-2, -1)
    y = torch.matmul(a,b)
    if alpha != 1:
        y = alpha*y
    if beta != 1:
        y = y + beta*c
    else:
        y = y + c
    return y

def add_gemm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.0)
    beta = node.attrs.get('beta', 1.0)
    transA = node.attrs.get('transA', 0)==1
    transB = node.attrs.get('transB', 0)==1
    torch_nodes[node.name] = torch_graph.call_function(torch_gemm, tuple(args), dict(alpha=alpha, beta=beta, transA=transA, transB=transB), name=node.name)
    