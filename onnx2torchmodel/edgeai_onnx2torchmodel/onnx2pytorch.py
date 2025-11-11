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
import onnx
import torch
import numpy as np

import logging
import tempfile
import types
import atexit
import os
import sys
import importlib

from . import onnx_ops


def simplify_graph(graph:gs.Graph):
    for idx, node in enumerate(graph.nodes):  
        if node.name.isnumeric() or not node.name:
            node.name = f'{node.op}_{idx}'
        else:
            node.name = node.name.replace('.','_').replace('/','_').replace(':','_')
    for name, tensor in graph.tensors().items():
        if name.isnumeric():
            tensor. name = f'tensor_{name}'
        else:
            tensor.name = tensor.name.replace('.','_').replace('/','_').replace(':','_')

    return graph


def add_slices_2_graph(graph, node, starts, ends, axes=None, steps=None, dict_mode=False):
    inp = node.inputs[0]
    out = node.outputs[0]
    node.inputs.clear()
    node.outputs.clear()
    axes = axes or list(range(len(starts)))
    steps = steps or [1]*len(axes)
    counter = 0
    for start, end, axis, step in zip(starts, ends, axes, steps):
        temp_out = gs.Variable(f'{out.name}_out{counter}', out.dtype) if counter<(len(starts)-1) else out
        if dict_mode:
            n_node = gs.Node('Slice',f'{out.name}_{counter}',dict(starts=[start], ends=[end], axes=[axis]),[inp], [temp_out], node.domain )
        else:
            start = gs.Constant(f'{out.name}_starts{counter}', values=np.array([start]).astype(np.int64))
            end = gs.Constant(f'{out.name}_ends{counter}', values=np.array([end]).astype(np.int64))
            axis = gs.Constant(f'{out.name}_axes{counter}', values=np.array([axis]).astype(np.int64))
            step = gs.Constant(f'{out.name}_steps{counter}', values=np.array([step]).astype(np.int64))
            n_node = gs.Node('Slice',f'{out.name}_{counter}',inputs=[inp, start, end, axis, step], outputs=[temp_out], domain=node.domain )
        inp = temp_out
        graph.nodes.append(n_node)
        counter += 1


def break_slices(graph:gs.Graph):
    slices = [node for node in graph.nodes if node.op == 'Slice']
    for node in slices:
        if 'starts' in  node.attrs:
            starts = node.attrs['starts']
            ends = node.attrs['ends']
            axes = node.attrs.get('axes', None)
            if len(starts) == 1:
                continue
            add_slices_2_graph(graph, node, starts, ends, axes, dict_mode=True)
            continue
        starts, ends = node.inputs[1:3]
        axes = node.inputs[3] if len(node.inputs)> 3 else None
        steps = node.inputs[4] if len(node.inputs)> 4 else None
        if not all(isinstance(i, gs.Constant) for i in (starts, ends)) or not all(isinstance(i, gs.Constant) or i is None for i in (axes, steps)):
            logging.warning(f'Can Not Break The Slice Node {node.name}, it may fail in pt2e export')
            continue
        starts = starts.values.tolist()
        ends = ends.values.tolist()
        axes = axes.values.tolist() if axes else None
        steps = steps.values.tolist() if steps else None
        if len(starts) == 1:
            continue
        add_slices_2_graph(graph, node, starts, ends, axes, steps)
    graph.toposort().cleanup()


def remove_identity_and_constant(graph:gs.Graph):
    nodes = [node for node in graph.nodes]# if node.op == 'Identity']
    for i, node in enumerate(nodes):
        if node.op == 'Identity':
            outputs = list(node.outputs)
            for out in outputs:
                for o in out.outputs:
                    if isinstance( node.inputs[0], gs.Constant ):
                        o.inputs[o.inputs.index(out)] = gs.Constant(out.name, node.inputs[0].values)
                    elif isinstance(node.inputs[0], gs.Variable):
                        o.inputs[o.inputs.index(out)] = gs.Variable(out.name, node.inputs[0].dtype, node.inputs[0].shape)
            graph.nodes.remove(node)
            if out in graph.outputs:
                graph.outputs.remove(out)
        if node.op == 'Constant':
            val = None
            for attr in ('sparse_value', 'value', 'value_float', 'value_int', 'value_floats', 'value_ints', 'value_string', 'value_strings'):
                val = node.attrs.get(attr, None)
                if val is None:
                    continue
                if isinstance(val, gs.Constant):
                    break
                val = np.array(val)
                if 'int' in attr:
                    val = val.astype(np.int64)
                if ('float' in attr ):
                    val = val.astype(np.float32)
                val = gs.Constant(node.name, val)
            if val is None:
                logging.warning(f'No Constant Found in Node {node.name}')
                continue
            out = node.outputs[0]
            node.outputs.clear()
            users = list(out.outputs)
            for o_node in users:
                o_node.inputs[o_node.inputs.index(out)] = val
    graph.toposort().cleanup()

def convert(model_path, for_training=False, module_based=True):
    onnx_model = onnx.load(model_path)
    graph = gs.import_onnx(onnx_model)
    simplify_graph(graph)
    remove_identity_and_constant(graph)
    break_slices(graph)

    torch_model = onnx_ops.get_torch_graph_module(graph, for_training=for_training, module_based=module_based)

    return torch_model
