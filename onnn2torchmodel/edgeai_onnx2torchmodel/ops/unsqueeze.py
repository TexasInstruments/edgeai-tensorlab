import torch
import onnx_graphsurgeon as gs
from . import utils

def torch_unsqueeze(x, dim):
    if isinstance(dim, (list, tuple)):
        if len(dim) == 1:
            return torch.unsqueeze(x, dim[0])
        shape =list(x.shape)
        output_rank = len(shape)+len(dim)
        output_shape = [0]*output_rank
        for d in dim:
            output_shape[d] = 1
        j=0
        for i in range(output_rank):
            if output_shape[i] == 1:
                continue
            output_shape[i] = shape[j]
            j+=1
        return torch.reshape(x, output_shape)
    else:
        return torch.unsqueeze(x, dim)

def add_unsqueeze_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<= len(node.inputs) <= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter,list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict()
    if 'axes' in node.attrs:
        kwargs['dim'] = node.attrs['axes'] 

    torch_nodes[node.name] = torch_graph.call_function(torch_unsqueeze, tuple(args),  kwargs, name=node.name)