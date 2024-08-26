# Copyright (c) 2018-2021, Texas Instruments
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

# TI's adaptation of update_model_dims.
# Original code from onnx.tools.update_model_dims() has bug of iterating through all layers.
# But this version has less error checking code.

__all__ = ['onnx_update_model_dims', 'get_all_output_names']


import os
import logging
from typing import List, Any


def update_dim(tensor=None, new_dim_value=None, dim_idx=None):
    dim_proto = tensor.type.tensor_type.shape.dim[dim_idx]
    if isinstance(new_dim_value, str):
        dim_proto.dim_param = new_dim_value
    elif isinstance(new_dim_value, int):
        if new_dim_value >= 0:
            assert not (dim_proto.HasField('dim_value') and (dim_proto.dim_value != new_dim_value))
        else: #new_dim_value is negative. Not handled currently.
            assert False
    return


def onnx_update_model_dims(model, input_dims, output_dims):
    for input_name, input_dim_arr in input_dims.items():
        input_layer_tensor = [input_tensor for input_tensor in model.graph.input if input_tensor.name == input_name][0]
        for dim_idx, new_dim_value in enumerate(input_dim_arr):
            update_dim(tensor=input_layer_tensor, new_dim_value=new_dim_value, dim_idx=dim_idx)

    for output_name, output_dim_arr in output_dims.items():
        output_layer_tensor = [output_tensor for output_tensor in model.graph.output if output_tensor.name == output_name][0]
        for dim_idx, new_dim_value in enumerate(output_dim_arr):
            update_dim(tensor=output_layer_tensor, new_dim_value=new_dim_value, dim_idx=dim_idx)

    import onnx
    onnx.checker.check_model(model)
    return model


def apply_onnx_patch():
    # this patch is not needed if this onnx fork is used:
    # https://github.com/TexasInstruments/onnx/archive/tidl-j7.zip
    # # patch/hack starts here #######################################
    # # TODO: revisit this patch/hack
    # # onnx package supports Python 3.10 only from version 1.12.0
    # # there is an issuze in onnxopt.tidlOnnxModelOptimize() with onnx 1.9.0
    # # this is a hack for now
    # import platform
    # from packaging import version
    # import onnx
    # this_python_version = version.parse(platform.python_version())
    # py10_verion = version.parse("3.10.0")
    # onnx_version = version.parse(onnx.__version__)
    # py310_compatible_onnx_version = version.parse("1.12.0")
    # if this_python_version >= py10_verion and onnx_version < py310_compatible_onnx_version:
    #     import collections
    #     collections.Iterable = collections.abc.Iterable
    # # patch/hack ends here #######################################
    pass


def find_out_layers (curr_layer) -> List[Any]:
    """
    Return all input nodes to a given node
    """
    out_layers = list()
    for outp in curr_layer.outputs:
        out_layers.extend(outp.outputs)
    return out_layers


def is_end_node(node) -> bool:
    """
    Return True if a node is an output node of the model
    """
    if len(find_out_layers(node)) == 0:
        return True
    return False


def get_all_nodes(node, end_nodes, searched_nodes):
    # recursive function to find all the deny list nodes
    searched_node_names = [node.name for node in searched_nodes]
    if node.name in end_nodes:
        add_in_list = True
        if node.name not in searched_node_names:
            searched_nodes.append(node)
            logging.debug(f"Adding {node.name} to node list.")
        return add_in_list, searched_nodes

    elif is_end_node(node):
        add_in_list = False
        return add_in_list, searched_nodes

    node_outputs = find_out_layers(node)
    add_in_list = False
    for n_id in node_outputs:
        add_in_list_here, searched_nodes = get_all_nodes(n_id, end_nodes, searched_nodes)
        # to add the intermediate nodes if one node has a branch which need not be included in deny list
        add_in_list = add_in_list or add_in_list_here
        if add_in_list and add_in_list_here and (n_id.name not in searched_node_names):
            searched_nodes.append(n_id)
            logging.debug(f"Adding {n_id.name} to node list.")

    return add_in_list, searched_nodes


def get_all_node_names(model_path, start_end_layers={}, verbose=False, graph=None, **kwargs):
    """
    Main function
    ---------------------------------------------------------
    Inputs
    ---------------------------------------------------------
    model_path:             path to input ONNX model
    start_end_layers:       dictionary of the start and end layers, between which (including start
                            and end node) needs to be added to deny list
                            if "None" is passed in the end node (values of dict), then the model output nodes
                            are assumed as the end nodes
     ---------------------------------------------------------------
    Output
    ---------------------------------------------------------------
    nodes:                  comma separated string of all the nodes that need to be added in the deny list
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    if graph is None:
        if not os.path.isfile(model_path):
            # check for valid path
            logging.error(f"File {model_path} not found")
            sys.exit(-1)
        #
        model = onnx.load(model_path)
        graph = gs.import_onnx(model)

    model_outputs = [node.inputs[0].name for node in graph.outputs]

    searched_nodes = []
    for node in graph.nodes:
        if node.name in start_end_layers.keys():
            end_layers = start_end_layers[node.name]
            if end_layers is None:
                end_layers = model_outputs
            _, searched_nodes = get_all_nodes(node, end_layers, searched_nodes)
            searched_nodes.append(node)
            logging.debug(f"Adding {node.name} to node list.")

    return searched_nodes


def get_all_output_names(model_path, start_end_layers={}, verbose=False, **kwargs):
    """
    Main function
    ---------------------------------------------------------
    Inputs
    ---------------------------------------------------------
    model_path:             path to input ONNX model
    start_end_layers:       dictionary of the start and end layers, between which (including start
                            and end node) needs to be added to deny list
                            if "None" is passed in the end node (values of dict), then the model output nodes
                            are assumed as the end nodes
     ---------------------------------------------------------------
    Output
    ---------------------------------------------------------------
    nodes:                  comma separated string of all the nodes that need to be added in the deny list
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    import onnx
    import onnx_graphsurgeon as gs

    # check for valid path
    if not os.path.isfile(model_path):
        logging.error(f"File {model_path} not found")
        sys.exit(-1)

    model = onnx.load(model_path)
    graph = gs.import_onnx(model)

    start_end_node_names = {}
    for k, v in start_end_layers.items():
        start_node = end_node = None
        for node in graph.nodes:
            for out in node.outputs:
                if k == out.name:
                    start_node = node.name
                elif v == out.name:
                    end_node = node.name
        start_end_node_names.update({start_node: end_node})

    selected_nodes = get_all_node_names(model, start_end_node_names, verbose, graph=graph)

    output_names = [out.name for node in selected_nodes for out in node.outputs]
    comma_separated_output_names = ', '.join(output_names)
    logging.info(f"get_all_output_names with start:end={start_end_layers} returned {len(output_names)} nodes:")
    logging.info(f"{comma_separated_output_names} ")

    return comma_separated_output_names
