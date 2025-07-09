import torch
import os
from itertools import chain
import pydot 
from torch import fx

try:
    from torch.fx.node import _format_arg
    from torch.fx.graph import _parse_stack_trace
    from torch.fx.passes.graph_drawer import FxGraphDrawer, _WEIGHT_TEMPLATE
except:
    print('WARNING: graph drawing functions for Pytorch models are not available - please update Pytorch to a more recent version to enable.')


class CustomFxGraphDrawer(FxGraphDrawer):
    # Note: Source code for this method is copied from pytorch github(https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/graph_drawer.py#L208) 
    # and modified as per requirement
    def _get_node_label(
            self,
            module: torch.fx.GraphModule,
            node: torch.fx.Node,
            skip_node_names_in_args: bool,
            parse_stack_trace: bool,
        ) -> str:
        def _get_str_for_args_kwargs(arg):
            if isinstance(arg, tuple):
                prefix, suffix = r"|args=(\l", r",\n)\l"
                arg_strs_list = [_format_arg(a, max_list_len=8) for a in arg]
            elif isinstance(arg, dict):
                prefix, suffix = r"|kwargs={\l", r",\n}\l"
                arg_strs_list = [
                    f"{k}: {_format_arg(v, max_list_len=8)}"
                    for k, v in arg.items()
                ]
            else:  # Fall back to nothing in unexpected case.
                return ""

            # Strip out node names if requested.
            if skip_node_names_in_args:
                arg_strs_list = [a for a in arg_strs_list if "%" not in a]
            if len(arg_strs_list) == 0:
                return ""
            arg_strs = prefix + r",\n".join(arg_strs_list) + suffix
            if len(arg_strs_list) == 1:
                arg_strs = arg_strs.replace(r"\l", "").replace(r"\n", "")
            return arg_strs.replace("{", r"\{").replace("}", r"\}")


        label = "{" + f"name=%{node.name}|op_code={node.op}\n|"

        if node.op == "call_module":
            leaf_module = self._get_leaf_node(module, node)
            label += str(node.target) + r"\n|"
            label += str(leaf_module) + r"\n|"
            extra = ""
            if hasattr(leaf_module, "__constants__"):
                extra = r"\n".join(
                    [f"{c}: {getattr(leaf_module, c)}" for c in leaf_module.__constants__]  # type: ignore[union-attr]
                )
            label += extra + r"\n"
        else:
            label += f"|target={self._typename(node.target)}" + r"\n"
        if len(node.args) > 0:
            label += _get_str_for_args_kwargs(node.args)
        if len(node.kwargs) > 0:
            label += _get_str_for_args_kwargs(node.kwargs)
        label += f"|num_users={len(node.users)}" + r"\n"

        tensor_meta = node.meta.get('tensor_meta')
        label += self._tensor_meta_to_label(tensor_meta)

        # for original fx graph
        # print buf=buf0, n_origin=6
        buf_meta = node.meta.get('buf_meta', None)
        if buf_meta is not None:
            label += f"|buf={buf_meta.name}" + r"\n"
            label += f"|n_origin={buf_meta.n_origin}" + r"\n"

        # for original fx graph
        # print file:lineno code
        if parse_stack_trace and node.stack_trace is not None:
            parsed_stack_trace = _parse_stack_trace(node.stack_trace)
            fname = self._shorten_file_name(parsed_stack_trace.file)
            label += f"|file={fname}:{parsed_stack_trace.lineno} {parsed_stack_trace.code}" + r"\n"


        return label + "}"
    
    # Note: Source code for this method is copied from pytorch github(https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/graph_drawer.py#L348) 
    # and modified as per requirement
    def _to_dot(
        self,
        graph_module: torch.fx.GraphModule,
        name: str,
        ignore_getattr: bool,
        ignore_parameters_and_buffers: bool,
        skip_node_names_in_args: bool,
        parse_stack_trace: bool,
    ) -> pydot.Dot:
        """
        Actual interface to visualize a fx.Graph. Note that it takes in the GraphModule instead of the Graph.
        If ignore_parameters_and_buffers is True, the parameters and buffers
        created with the module will not be added as nodes and edges.
        """

        # "TB" means top-to-bottom rank direction in layout
        dot_graph = pydot.Dot(name, rankdir="TB")


        buf_name_to_subgraph = {}

        for node in graph_module.graph.nodes:
            if ignore_getattr and node.op == "get_attr":
                continue

            style = self._get_node_style(node)
            node_name = str(node.name).replace('.','_')
            dot_node = pydot.Node(
                node_name, label=self._get_node_label(graph_module, node, skip_node_names_in_args, parse_stack_trace), **style
            )

            current_graph = dot_graph

            buf_meta = node.meta.get('buf_meta', None)
            if buf_meta is not None and buf_meta.n_origin > 1:
                buf_name = buf_meta.name
                if buf_name not in buf_name_to_subgraph:
                    buf_name_to_subgraph[buf_name] = pydot.Cluster(buf_name, label=buf_name)
                current_graph = buf_name_to_subgraph.get(buf_name)

            current_graph.add_node(dot_node)

            def get_module_params_or_buffers():
                for pname, ptensor in chain(
                    leaf_module.named_parameters(), leaf_module.named_buffers()
                ):
                    pname1 = node_name + "_" + pname # originally '.' instead of '_' causing error in dot program
                    pname1 = pname1.replace('.','_')
                    label1 = (
                        pname1 + "|op_code=get_" + "parameter"
                        if isinstance(ptensor, torch.nn.Parameter)
                        else "buffer" + r"\l"
                    )
                    dot_w_node = pydot.Node(
                        pname1,
                        label="{" + label1 + self._get_tensor_label(ptensor) + "}",
                        **_WEIGHT_TEMPLATE,
                    )
                    dot_graph.add_node(dot_w_node)
                    dot_graph.add_edge(pydot.Edge(pname1, node_name))

            if node.op == "call_module":
                leaf_module = self._get_leaf_node(graph_module, node)

                if not ignore_parameters_and_buffers and not isinstance(leaf_module, torch.fx.GraphModule):
                    get_module_params_or_buffers()

        for subgraph in buf_name_to_subgraph.values():
            subgraph.set('color', 'royalblue')
            subgraph.set('penwidth', '2')
            dot_graph.add_subgraph(subgraph)

        for node in graph_module.graph.nodes:
            if ignore_getattr and node.op == "get_attr":
                continue
            node_name = str(node.name).replace('.','_')
            for user in node.users:
                dot_graph.add_edge(pydot.Edge(node_name, str(user.name).replace('.','_')))

        return dot_graph


class CustomPT2EGraphDrawer(FxGraphDrawer):
    def __init__(self, graph_module: fx.GraphModule, name: str,ignore_getattr: bool = False, ignore_parameters_and_buffers: bool = False, skip_node_names_in_args: bool = True, parse_stack_trace: bool = False, get_shape_inference: bool = True, dot_graph_shape: str | None = None):
        super().__init__(graph_module, name, ignore_getattr, ignore_parameters_and_buffers, skip_node_names_in_args, parse_stack_trace, dot_graph_shape)
        self._dot_graphs = {
                name: self._to_dot(
                    graph_module, name, ignore_getattr, ignore_parameters_and_buffers, skip_node_names_in_args, parse_stack_trace, get_shape_inference
                )
            }
        
    # Note: Source code for this method is copied from pytorch github(https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/graph_drawer.py#L208) 
    # and modified as per requirement
    def _get_node_label(
            self,
            module: torch.fx.GraphModule,
            node: torch.fx.Node,
            skip_node_names_in_args: bool,
            parse_stack_trace: bool,
        ) -> str:
        def _get_str_for_args_kwargs(arg):
            if isinstance(arg, tuple):
                prefix, suffix = r"|args=(\l", r",\n)\l"
                arg_strs_list = [_format_arg(a, max_list_len=8) for a in arg]
            elif isinstance(arg, dict):
                prefix, suffix = r"|kwargs={\l", r",\n}\l"
                arg_strs_list = [
                    f"{k}: {_format_arg(v, max_list_len=8)}"
                    for k, v in arg.items()
                ]
            else:  # Fall back to nothing in unexpected case.
                return ""

            # Strip out node names if requested.
            if skip_node_names_in_args:
                arg_strs_list = [a for a in arg_strs_list if "%" not in a]
            if len(arg_strs_list) == 0:
                return ""
            arg_strs = prefix + r",\n".join(arg_strs_list) + suffix
            if len(arg_strs_list) == 1:
                arg_strs = arg_strs.replace(r"\l", "").replace(r"\n", "")
            return arg_strs.replace("{", r"\{").replace("}", r"\}")


        label = "{" + f"name=%{node.name}|op_code={node.op}\n|"

        if node.op == "call_module":
            leaf_module = self._get_leaf_node(module, node)
            label += str(node.target) + r"\n|"
            label += str(leaf_module) + r"\n|"
            extra = ""
            if hasattr(leaf_module, "__constants__"):
                extra = r"\n".join(
                    [f"{c}: {getattr(leaf_module, c)}" for c in leaf_module.__constants__]  # type: ignore[union-attr]
                )
            label += extra + r"\n"
        else:
            label += f"|target={self._typename(node.target)}" + r"\n"
        if len(node.args) > 0:
            label += _get_str_for_args_kwargs(node.args)
        if len(node.kwargs) > 0:
            label += _get_str_for_args_kwargs(node.kwargs)
        label += f"|num_users={len(node.users)}" + r"\n"

        tensor_meta = node.meta.get('tensor_meta')
        label += self._tensor_meta_to_label(tensor_meta)

        # for original fx graph
        # print buf=buf0, n_origin=6
        buf_meta = node.meta.get('buf_meta', None)
        if buf_meta is not None:
            label += f"|buf={buf_meta.name}" + r"\n"
            label += f"|n_origin={buf_meta.n_origin}" + r"\n"

        # for original fx graph
        # print file:lineno code
        if parse_stack_trace and node.stack_trace is not None:
            parsed_stack_trace = _parse_stack_trace(node.stack_trace)
            fname = self._shorten_file_name(parsed_stack_trace.file)
            label += f"|file={fname}:{parsed_stack_trace.lineno} {parsed_stack_trace.code}" + r"\n"


        return label + "}"
    
    # Note: Source code for this method is copied from pytorch github(https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/graph_drawer.py#L348) 
    # and modified as per requirement
    def _to_dot(
        self,
        graph_module: torch.fx.GraphModule,
        name: str,
        ignore_getattr: bool,
        ignore_parameters_and_buffers: bool,
        skip_node_names_in_args: bool,
        parse_stack_trace: bool,
        get_shape_inference: bool = True
    ) -> pydot.Dot:
        """
        Actual interface to visualize a fx.Graph. Note that it takes in the GraphModule instead of the Graph.
        If ignore_parameters_and_buffers is True, the parameters and buffers
        created with the module will not be added as nodes and edges.
        """

        # "TB" means top-to-bottom rank direction in layout
        dot_graph = pydot.Dot(name, rankdir="TB")


        buf_name_to_subgraph = {}
        char_pairs = (
            ('[',r'\['),(']',r'\]'),
            ('(',r'\('),(')',r'\)'),
            ('{',r'\{'),('}',r'\}'),
            ('\"',r'\"'),('\'',r'\''),
            )
        
        params_and_buffer = dict(chain(graph_module.named_parameters(), graph_module.named_buffers()))
        
        for node in graph_module.graph.nodes:
            if ignore_getattr and node.op == "get_attr":
                continue
            
            node_name = node.name.replace('.','_')
            style = self._get_node_style(node)
            if node.op == 'get_attr':
                tooltip =  str(params_and_buffer[node.target])
                
            else:
                tooltip = ''
            dot_node = pydot.Node(
                node_name, label=self._get_node_label(graph_module, node, skip_node_names_in_args, parse_stack_trace), tooltip = '{'+tooltip+'}', **style
            )

            current_graph = dot_graph

            buf_meta = node.meta.get('buf_meta', None)
            if buf_meta is not None and buf_meta.n_origin > 1:
                buf_name = buf_meta.name
                if buf_name not in buf_name_to_subgraph:
                    buf_name_to_subgraph[buf_name] = pydot.Cluster(buf_name, label=buf_name)
                current_graph = buf_name_to_subgraph.get(buf_name)

            current_graph.add_node(dot_node)

            def get_module_params_or_buffers():
                for pname, ptensor in chain(
                    leaf_module.named_parameters(), leaf_module.named_buffers()
                ):
                    pname1 = node_name + "_" + pname # originally '.' instead of '_'
                    pname1 = pname1.replace('.','_')
                    label1 = (
                        pname1 + "|op_code=get_" + "parameter"
                        if isinstance(ptensor, torch.nn.Parameter)
                        else "buffer"+ f'_{pname}'  + r"|"
                    )
                    tooltip = (
                        str(ptensor)
                    )
                    dot_w_node = pydot.Node(
                        pname1,
                        label="{" + label1 + self._get_tensor_label(ptensor) + "}", tooltip = '{' + tooltip +'}',
                        **_WEIGHT_TEMPLATE,
                    )
                    if get_shape_inference:
                        if 'example_value' in node.meta:
                            value = node.meta['example_value']
                        elif 'val' in node.meta:
                            value = node.meta['val']
                        
                        if hasattr(value,'shape') and len(value.shape):
                            shape= list(value.shape) 
                        elif isinstance(value, (list, tuple)) :
                            shape = []
                            for val in value:
                                if hasattr(val,'shape') and len(val.shape) != 0:
                                    shape.append(list(val.shape))
                                else: shape.append(val)
                        else:
                            shape = value
                        label = str (shape)
                        for c1,c2 in char_pairs:
                            label = label.replace(c1,c2)
                        dot_graph.add_node(dot_w_node)
                        dot_graph.add_edge(pydot.Edge(pname1, node_name, label=label, tooltip = tooltip))
                    else:
                        dot_graph.add_edge(pydot.Edge(pname1, node_name))

            if node.op == "call_module":
                leaf_module = self._get_leaf_node(graph_module, node)

                if not ignore_parameters_and_buffers and not isinstance(leaf_module, torch.fx.GraphModule):
                    get_module_params_or_buffers()

        for subgraph in buf_name_to_subgraph.values():
            subgraph.set('color', 'royalblue')
            subgraph.set('penwidth', '2')
            dot_graph.add_subgraph(subgraph)

        for node in graph_module.graph.nodes:
            if ignore_getattr and node.op == "get_attr":
                continue
            node_name = node.name.replace('.','_')
            for user in node.users:
                user_name = user.name.replace('.','_')
                if get_shape_inference: 
                    if 'example_value' in node.meta:
                        value = node.meta['example_value']
                    elif 'val' in node.meta:
                        value = node.meta['val']

                    if hasattr(value,'shape') and len(value.shape):
                        shape= list(value.shape) 
                    elif isinstance(value, (list, tuple)):
                        shape = []
                        for val in value:
                            if hasattr(val,'shape') and len(val.shape) != 0:
                                shape.append(list(val.shape))
                            else:
                                shape.append(val)
                    else:
                        shape = value
                    label = str(shape)
                    for c1,c2 in char_pairs:
                        label = label.replace(c1,c2)
                    dot_graph.add_edge(pydot.Edge(node_name, user_name, label=label,tooltip = str(value)))
                else:
                    dot_graph.add_edge(pydot.Edge(node_name, user_name))
                    
        return dot_graph


def clean(model:fx.GraphModule):
    with model.graph.inserting_after():
        hanging_nodes = [node for node in model.graph.nodes if len(node.users)  == 0 and node.op != 'output']
        while(len(hanging_nodes)):
            for node in hanging_nodes:
                model.graph.erase_node(node)    
            hanging_nodes = [node for node in model.graph.nodes if len(node.users)  == 0 and node.op != 'output']
    model.graph.lint()
    model.recompile()
    return model  


def save_svg_pt2e(model, model_name, path_to_export = '.', hanging_nodes=True):
    os.makedirs(path_to_export, exist_ok=True)
    if not hanging_nodes:
        model_name += '_nh'
        model = clean(model)
    g = CustomPT2EGraphDrawer(model,model_name, )
    with open(f'{path_to_export}/{model_name}_pt2e.svg', "wb") as f:
        try:
            f.write(g.get_dot_graph().create_svg())
        except Exception as e:
            g.get_dot_graph().write(f'{path_to_export}/temp.txt',None)
            raise e


def save_svg_fx(model, model_name, path_to_export = '.', hanging_nodes=True):
    os.makedirs(path_to_export, exist_ok=True)
    if not hanging_nodes:
        model_name += '_nh'
        model = clean(model)
    g = CustomFxGraphDrawer(model,model_name)
    with open(f'{path_to_export}/{model_name}_fx.svg', "wb") as f:
        try:
            f.write(g.get_dot_graph().create_svg())
        except Exception as e:
            g.get_dot_graph().write(f'{path_to_export}temp.txt',None)
            raise e