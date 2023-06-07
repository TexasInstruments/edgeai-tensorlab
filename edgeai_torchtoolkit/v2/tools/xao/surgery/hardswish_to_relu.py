import torch, onnx
from torch import nn, onnx as tonnx,rand
from torchvision.models import  mobilenet_v3_small
from torch.fx import symbolic_trace, GraphModule, replace_pattern, Node
from typing import Any,Dict,Tuple

def replace_node_module(node: torch.fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert (isinstance(node.target, str))
    *parent, name = node.target.rsplit('.', 1)
    parent_name, name =( parent[0] if parent else ''), name
    setattr(modules[parent_name], name, new_module)
    

def replaceHardSwishToReLU(model:nn.Module):
    tracedModel : GraphModule = symbolic_trace(model)
    # tracedModel.print_readable()
    nodeModules=dict(tracedModel.named_modules())
    noOFHS=0
    for node in tracedModel.graph.nodes:
        # arguments=node.args.deepcopy()
        if node.op == 'call_module' :
            module=nodeModules[node.target]
            if type(module) in [nn.Hardswish,nn.ReLU]:
                newModel=nn.ReLU(inplace=False)
                module=newModel
                replace_node_module(node,nodeModules,newModel)
            elif type(module) in [nn.Dropout]:
                newModel=nn.Dropout(inplace=False)
                module=newModel
                replace_node_module(node,nodeModules,newModel) 
            # newNodeName=node.target
            # noOFHS+=1
            # newModel=nn.ReLU(inplace=False)
            # tracedModel.add_submodule(str(node)+'_relu', newModel)
            # nodeModules[newNodeName]=newModel
            # with tracedModel.graph.inserting_before(node):
            #     newLayer =tracedModel.graph.call_module(module_name=newNodeName,args=node.args[0:1],kwargs={})
            # print(nodeModules[node.target])
            # node.replace_all_uses_with(newLayer)
            # tracedModel.delete_all_unused_submodules()
            # tracedModel.graph.erase_node(node)

        elif node.op == 'call_function' and node.target in [torch.nn.functional.hardswish]:
            print('Hard Swish function')
            with tracedModel.graph.inserting_after(node):
                new_node = tracedModel.graph.call_function(torch.nn.functional.relu, node.args, node.kwargs)
                node.replace_all_uses_with(new_node)
            # Remove the old node from the graph
            tracedModel.graph.erase_node(node)
    tracedModel.graph.lint()
    tracedModel.recompile()
    # tracedModel.print_readable( )
    return (tracedModel)