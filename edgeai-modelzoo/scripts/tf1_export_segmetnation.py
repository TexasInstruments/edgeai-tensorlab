# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved
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

import tensorflow as tf
import os

model_names_=['deeplabv3_mnv2_cityscapes_train', 'edgetpu-deeplab', 'edgetpu-deeplab-slim']
input_arrays_=[['ImageTensor'], ['ImageTensor'], ['ImageTensor']]
output_arrays_ = [['SemanticPredictions'], ['SemanticPredictions'], ['SemanticPredictions']]

for model_name, input_arrays, output_arrays in zip(model_names_, input_arrays_, output_arrays_):
    local_dir = f"./downloads/tf1/seg/{model_name}"
    graph_def_file = f'{local_dir}/frozen_inference_graph.pb'
    output_file = f'{local_dir}/tflite/model.tflite'
    input_shapes = None #{input_arrays[0]: [1, 512, 1024, 3]}
    output_dir = os.path.split(output_file)[0]
    os.makedirs(output_dir, exist_ok=True)
    #Converting a GraphDef from file.
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays, input_shapes=input_shapes)
    tflite_model = converter.convert()
    open(output_file, "wb").write(tflite_model)
