from typing import Any
import onnx_graphsurgeon as gs
import torch
from ops import utils, basic_ops_2_func_dict, custom_add_2_torch_graph

class State:
    def __init__(self):
        self.data = {}
    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except:
            if name in self.data: return self.data[name]
    def __setattr__(self, name: str, value: Any) -> None:
        try:
            super().__setattr__(name, value)
        except:
            if hasattr(self, 'data'): 
                self.data[name] = value
    
def get_torch_graph_module(graph:gs.Graph):
    inputs = list(graph.inputs)
    
    op_2_func_dict = basic_ops_2_func_dict.copy()
    op_2_func_dict.update(custom_add_2_torch_graph)
    
    state = State()
    state.graph = graph
    torch_graph = torch.fx.Graph()
    root_module = torch.nn.Module()
    torch_nodes = {}
    for inp in inputs:
        torch_nodes[inp.name] = torch_graph.placeholder(inp.name)
    
    for node in graph.nodes:
        func = op_2_func_dict[node.op]
        func(state, node, torch_graph, torch_nodes, root_module)
    
    outputs = list(graph.outputs)
    if len(outputs) == 1:
        output = utils.get_input_from_node(outputs[0], torch_graph, torch_nodes, root_module)
        torch_nodes['outputs'] = torch_graph.output(output)
    else:
        torch_outputs = [] 
        for out in outputs:
            torch_outputs.append(utils.get_input_from_node(out, torch_graph, torch_nodes, root_module))
        torch_nodes['outputs'] = torch_graph.output(tuple(torch_outputs))
    
    
    return torch.fx.GraphModule(root_module, torch_graph)