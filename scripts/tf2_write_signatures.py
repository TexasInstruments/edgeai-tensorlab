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

def get_graph_def_from_saved_model(saved_model_dir):
    with tf.Session() as session:
        meta_graph_def = tf.saved_model.loader.load(
            session,
            tags=['serve'],
            export_dir=saved_model_dir
        )
    return meta_graph_def.graph_def

graph_def = get_graph_def_from_saved_model('/user/a0393608/files/work/ti/bitbucket/jacinto-ai/jacinto-ai-modelzoo/downloads/tf2/classification/imagenet_resnet_v2_50_classification_4/saved_model')

input_nodes = ['resnet_v2_50/block1/unit_1/bottleneck_v2/preact/moving_variance']
output_nodes = ["resnet_v2_50/postnorm/moving_variance"]

with tf.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    inputs = {input_node: session.graph.get_tensor_by_name(f'{input_node}:0') for input_node in input_nodes}
    outputs = {output_node: session.graph.get_tensor_by_name(f'{output_node}:0') for output_node in output_nodes}
    tf.saved_model.simple_save(
        session,
        '/user/a0393608/files/work/ti/bitbucket/jacinto-ai/jacinto-ai-modelzoo/downloads/tf2/classification/imagenet_resnet_v2_50_classification_4/saved_model_with_sig',
        inputs=inputs,
        outputs=outputs
    )