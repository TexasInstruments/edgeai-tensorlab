#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################


import onnx
from onnx import helper
from onnx import TensorProto,shape_inference 


def tidlIsNodeOutputNameUsedInGraph(originalGraph,name):
    for node in originalGraph.node:
        if(node.output[0] == name):
            return True
    return False


# This function updates intermediate input/output tensor names to be integers starting from 1
# The graph's final output names are not updated by default to ensure there are no issues in case they are being used for interfacing in any way
# In case changing graph's output names is desired, set updateGraphOutputNames = True in function call
def prune_layer_names(in_model_path, out_model_path, opset_version=11, updateGraphOutputNames=False):
    #Read ONNX Model
    model = onnx.load_model(in_model_path)
    op = onnx.OperatorSetIdProto()
    #Track orginal opset:
    op.version = model.opset_import[0].version
    #Get Graph:
    originalGraph = model.graph
    
    nodeIdx = 0
    for node1 in range(len(originalGraph.node)):
        #if new name is already used in graph for some output, increment nodeIdx to get different name
        while(tidlIsNodeOutputNameUsedInGraph(originalGraph, nodeIdx)):
            nodeIdx += 1
        newName = str(nodeIdx).encode('utf-8')
        originalGraph.node[node1].name = newName

        for node2 in range(len(originalGraph.node)):
            for inputIdx in range(len(originalGraph.node[node2].input)):
                if(originalGraph.node[node2].input[inputIdx] == originalGraph.node[node1].output[0]):
                    # Update node input name for corresponding input node's updated output name
                    originalGraph.node[node2].input[inputIdx] = newName
        isOutputNode = False
        for graphOutIdx in range(len(originalGraph.output)):
            if(originalGraph.output[graphOutIdx].name == originalGraph.node[node1].output[0]):
                isOutputNode = True
                if(updateGraphOutputNames):
                    originalGraph.output[graphOutIdx].name = newName
        if(isOutputNode):
            if(updateGraphOutputNames):
                originalGraph.node[node1].output[0] = newName
        else:
            originalGraph.node[node1].output[0] = newName
        nodeIdx += 1
    
    #Construct Model:
    op.version = opset_version
    model_def_noShape = helper.make_model(originalGraph, producer_name='onnx-TIDL', opset_imports=[op])
    model_def = shape_inference.infer_shapes(model_def_noShape)

    try:
        onnx.checker.check_model(model_def)
    except onnx.checker.ValidationError as e:
        print('Converted model is invalid: %s' % e)
    else:
        print('Converted model is valid!')
        onnx.save_model(model_def, out_model_path)
    
    return

