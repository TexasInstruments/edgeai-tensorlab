
import torch
import onnx
import onnx_graphsurgeon as gs

import os
import types

from ...xmodelopt.utils.helper_functions import get_attr, get_module


def create_gs_node(model, fx_node:torch.fx.Node, fx_2_gs_dict:dict[str,gs.Tensor]):
    """Creates a GraphSurgeon node from a PyTorch FX node.
    
    This function converts a PyTorch FX node to a GraphSurgeon (gs) node, which is used in 
    the ONNX graph construction process. It handles different types of FX nodes including 
    'get_attr', 'placeholder', and operation nodes.
    
    Args:
        model: The PyTorch model that contains the FX node.
        fx_node (torch.fx.Node): The PyTorch FX node to convert.
        fx_2_gs_dict (dict[str, gs.Tensor]): Dictionary mapping FX node names to their 
            corresponding GraphSurgeon tensors.
            
    Returns:
        gs.Node or None: The created GraphSurgeon node, or None for special node types 
        like 'get_attr' and 'placeholder' that only update the dictionary.
    """
    if fx_node.op == 'get_attr':
        attr = get_attr(model, fx_node.target)
        if isinstance(attr, torch.Tensor):
            attr = attr.detach().cpu().numpy()
            attr = gs.Constant(fx_node.name, attr)
        fx_2_gs_dict[fx_node.name] = attr
        return 
    
    if fx_node.op == 'placeholder':
        val = gs.Variable(fx_node.name, )
        val.shape = getattr(fx_node.meta.get('val', None),'shape', ['NA'])
        fx_2_gs_dict[fx_node.name] = val
        return
    
    def _flatten_types(args):
        """Recursively extracts types from nested data structures.
        
        This helper function recursively traverses nested collections (lists, tuples, sets, dicts)
        and collects the types of all leaf elements.
        
        Args:
            args: A potentially nested collection of objects.
            
        Returns:
            list: A list of types found in the nested structure, or a single type for 
            non-collection inputs.
        """
        result = []
        if isinstance(args, (list, tuple, set)):
            for arg in args:
                types_ = _flatten_types(arg)
                if isinstance(types_, list):
                    result.extend(types_)
                else:
                    result.append(types_)
        elif isinstance(args, dict):
            for arg in args.values():
                types_ = _flatten_types(arg)
                if isinstance(types_, list):
                    result.extend(types_)
                else:
                    result.append(types_)
        else:
            return type(args)
        return result
    
    def get_inputs(args, name=f'{fx_node.name}_arg'):
        """Converts PyTorch FX node arguments to GraphSurgeon inputs.
        
        This helper function processes node arguments and converts FX nodes to their
        corresponding GraphSurgeon tensors from the provided dictionary.
        It handles nested structures and builds appropriate named inputs.
        
        Args:
            args: The arguments to process, which can be FX nodes, collections, or other values.
            name (str): Base name to use for generated variables.
            
        Returns:
            dict or object: A dictionary of named inputs if the args contain FX nodes,
            otherwise the original args if they don't need conversion.
        """
        if isinstance(args, (list, tuple, set, dict)):
            types_ = _flatten_types(args)
            if torch.fx.Node in  types_:
                results =  {}
                if isinstance(args, dict):
                    for k, v in args.items():
                        results[name + f'_{k}'] = get_inputs(v, name + f'_{k}')
                else:
                    for i, arg in enumerate(args):
                        results[name + f'_{i}'] = get_inputs(arg, name + f'_{i}')
                final_result = {}
                for k, v in results.items():
                    if isinstance(v, dict):
                        final_result.update(v)
                    else:
                        final_result[k] = v
                return final_result
            else:
                return args
        if isinstance(args, torch.fx.Node):
            return fx_2_gs_dict[args.name]
        return args
    
    inputs = get_inputs(fx_node.args)
    assert isinstance(inputs, dict)
    attrs = {}
    for name, inp in inputs.items():
        if not isinstance(inp, (gs.Constant, gs.Variable)):
            inputs[name] = gs.Variable(name,dtype=None, shape=None)
            attrs[name] = str(inp)
    inputs = [v for k,v in inputs.items() if k not in attrs]
    attrs.update(**{k:str(v) for k,v in fx_node.kwargs.items()})
    
    out = gs.Variable(f'{fx_node}',None, )
    out.shape = getattr(fx_node.meta.get('val', None),'shape', None)
    target = fx_node.target
    if fx_node.op == 'call_module':
        module = get_module(model, fx_node.target)
        if target.startswith('activation_post_process'):
            attrs_to_check = ['dtype', 'ch_axis', 'eps','fixed_range','is_dynamic','has_customized_qrange',
                        'min_val', 'max_val', 'power2_scale', 'qscheme','quant_min','quant_max','range_max',
                        'reduce_range','symmetric','is_per_channel','observer_enabled', 'fake_quantized_enabled',
                        'scale','zero_point']
        else:
            attrs_to_check = dir(module)
            attrs_to_check = [attr for attr in attrs_to_check if not attr.startswith('_')]
            attrs_to_check = [attr for attr in attrs_to_check if not isinstance(getattr(module, attr), types.MethodType)]
            
        vals = {}
        for attr in attrs_to_check:
            val = 'NA'
            if hasattr(module, attr):
                val = getattr(module, attr)
            elif hasattr(module, 'activation_post_process'):
                m = module.activation_post_process
                if hasattr(m, attr):
                    val = getattr(m,attr)
            if val == 'NA':
                continue
            vals[attr]=val
        attrs.update(**vals)
        target = str(type(module))
        if fx_node.args[0].name in fx_2_gs_dict:
            out.shape = list(fx_2_gs_dict[fx_node.args[0].name].shape).copy()
    else:
        split = str(target).rsplit('.', 1)
        if len(split) == 2:
            target =  '_'.join(split)
    for name, attr in attrs.items():
        if isinstance(attr, torch.Tensor):
            attr = attr.detach().cpu().numpy()
            if len(attr.flatten()) == 0:
                attr = str(attr)
            elif attr.ndim == 0:
                attr = attr.tolist()
            attrs[name] = attr
        else:
            attrs[name] = str(attr)
    fx_2_gs_dict[fx_node.name] = out
    return gs.Node(str(target), fx_node.name, attrs, inputs, [out] )
    

def save_torch_graph_as_onnx(model:torch.fx.GraphModule, path):
    """Converts a PyTorch GraphModule to ONNX format and saves it to disk.
    
    This function takes a PyTorch FX GraphModule and converts it to an ONNX model
    using the ONNX GraphSurgeon library. It traverses the PyTorch graph, creates
    equivalent GraphSurgeon nodes and tensors, and exports the resulting graph
    to an ONNX file.
    
    Args:
        model (torch.fx.GraphModule): The PyTorch GraphModule to convert to ONNX.
        path (str): The file path where the ONNX model should be saved. If the path
            doesn't end with '.onnx', the extension will be added automatically.
            If the path is a directory, 'torch_model.onnx' will be appended.
            
    Returns:
        None: The function saves the ONNX model to disk but doesn't return a value.
    """
    nodes = list(model.graph.nodes)
    if not path.endswith('.onnx'):
        if os.path.isdir(path):
            path += 'torch_model'
        path += '.onnx'
    fx_2_gs_tensor_dict = {}
    gs_nodes = [create_gs_node(model, node, fx_2_gs_tensor_dict) for node in nodes]
    inputs = []
    outputs = []
    for node in nodes:
        if node.op == 'placeholder':
            inputs.append(fx_2_gs_tensor_dict[node.name])
        if node.op == 'output':
            outputs.extend([fx_2_gs_tensor_dict[o.name] for o in node._input_nodes])
    gs_nodes.pop()
    # nodes = []
    graph = gs.Graph([node for node in gs_nodes if node], inputs, outputs, doc_string=str(model.graph), )
    for v in graph.tensors().values():
        if v.shape is None:
            v.shape = ['NA']
        if isinstance(v, gs.Variable):
            v.dtype = 'float32'
    graph.toposort()
    onnx_model = gs.export_onnx(graph)
    onnx.save(onnx_model, path)