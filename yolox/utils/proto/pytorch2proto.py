# Copyright (c) 2018-2020, Texas Instruments
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


import os
import numpy as np

import onnx
import torch
#from xmmdet.utils import save_model_proto


__all__ = ['pytorch2proto']


def prepare_model_for_layer_outputs(model_name, model_name_new, export_layer_types=None):
    onnx_model = onnx.load(model_name)
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = ''
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].output)):
            if export_layer_types is None or onnx_model.graph.node[i].op_type in export_layer_types:
                if(i+1<len(onnx_model.graph.node)) and onnx_model.graph.node[i+1].op_type == "Concat":
                    intermediate_layer_value_info.name = onnx_model.graph.node[i+1].output[0]
                    onnx_model.graph.output.append(intermediate_layer_value_info)
            #
        #
    #
    layer_output_names = [out.name for out in onnx_model.graph.output]
    onnx.save(onnx_model, model_name_new)
    return layer_output_names


def similar_tensor(t1, t2, rtol=1.e-3, atol=1.e-3):
    if len(t1.shape) != len(t2.shape):
        return False
    if not np.allclose(t1.shape, t2.shape):
        return False
    if np.isnan(t1).all() or np.isnan(t2).all():
        return False

    max_t1 = abs(np.nanmax(t1))
    atol = max_t1*atol

    is_close = np.allclose(t1, t2, rtol=rtol, atol=atol, equal_nan=True)
    if not is_close:
        eps = max_t1 / rtol
        diff = np.nanmax(np.abs((t1-t2)))
        ratio = np.nanmax(np.abs((t1-t2)/(t1+eps)))
        is_close = diff < atol and ratio < rtol
        print(f'{t1.shape} - max difference: {diff} vs {atol}, max ratio: {ratio} vs {rtol}')
    #
    return is_close


def retrieve_onnx_names(input_data, partial_model, full_model_path):
    import onnxruntime
    full_model_path_tmp = f'{full_model_path}.tmp'
    full_output_names = prepare_model_for_layer_outputs(full_model_path, full_model_path_tmp, export_layer_types='Conv')
    full_infer = onnxruntime.InferenceSession(full_model_path_tmp)
    full_input_name = full_infer.get_inputs()[0].name


    # partial_infer = onnxruntime.InferenceSession(partial_model_path)
    # partial_input_name = partial_infer.get_inputs()[0].name
    # partial_output_names = [o.name for o in partial_infer.get_outputs()]

    input_numpy = input_data.detach().numpy() if isinstance(input_data, torch.Tensor) else input_data
    full_outputs = full_infer.run(full_output_names, {full_input_name:input_numpy})
    partial_model.head.export_proto = True
    with torch.no_grad():
        partial_model.eval()
        partial_outputs = partial_model(input_data)

    matched_names = []
    num_classes = partial_model.head.num_classes
    for po in partial_outputs:
        matched_name = None
        bs,  no, ny, nx = po.shape
        for fname, fo in zip(full_output_names, full_outputs):
            fo[:,4:5+num_classes, ...] = 1.0/(1.0 + np.exp(-fo[:, 4:5+num_classes, ...]))
            if similar_tensor(po, fo):
                matched_name = fname
                break
            #
        #
        if matched_name is None:
            return None
        #
        matched_names.append(matched_name)
    #
    os.remove(full_model_path_tmp)
    return matched_names


# def pytorch2proto(cfg, model, input_data, output_onnx_file, out_proto_file, output_names, opset_version):
#     input_data = input_data[0] if isinstance(input_data, (list,tuple)) and \
#                                   isinstance(input_data[0], (torch.Tensor, np.ndarray)) else input
#     save_model_proto(cfg, model, input_data, output_filename=out_proto_file, output_names=output_names, opset_version=opset_version)
#     matched_names = None
#     if output_onnx_file is not None:
#         matched_names = retrieve_onnx_names(input_data, out_proto_file, output_onnx_file)
#         if matched_names is not None:
#             proto_names = list(matched_names.values())
#             save_model_proto(cfg, model, input_data, output_filename=output_onnx_file,
#                              opset_version=opset_version, proto_names=proto_names, output_names=output_names,
#                              save_onnx=False)
#         #
#     #
#     if not matched_names:
#         print('Tensor names could not be located; prototxt file corresponding to full model '
#               '(model.onnx) wont be written')
