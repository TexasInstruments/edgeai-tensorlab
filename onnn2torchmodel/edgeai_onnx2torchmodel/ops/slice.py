import torch
import onnx_graphsurgeon as gs
from . import utils
from operator import getitem

def torch_slice(x, starts, ends, axes, steps=None):
    if steps is None:
        steps = torch.ones_like(torch.tensor(axes))
    slices = [[None, None, None] for _ in range(x.ndim)]
    for i, axis in enumerate(axes):
        slices[axis][0] = starts[i]
        slices[axis][1] = ends[i]
        slices[axis][2] = steps[i]
    for i, slc in enumerate(slices):
        if slc[0] == None:
            continue
        slc_ = [slice(None, None, None) for _ in range(x.ndim)]
        slc_[i] = slice(*slc)
        x = getitem(x, tuple(slc_))
    return x

def add_slice_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 4<=len(node.inputs)<=5, f'{node.name} with operator {node.op} should have 4 or 5 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, list, list, list, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict()
    if 'starts' in node.attrs:
        starts = node.attrs.get('starts')
        kwargs['starts'] = starts
    if 'ends' in node.attrs:
        ends = node.attrs.get('ends')
        kwargs['ends'] = ends
    if 'axes' in node.attrs:
        axes = node.attrs.get('axes')
        kwargs['axes'] = axes
    if 'steps' in node.attrs:
        steps = node.attrs.get('steps')
        kwargs['steps'] = steps
    torch_nodes[node.name] = torch_graph.call_function(torch_slice, tuple(args),  kwargs, name=node.name)
