# dependencies
# Anaconda Python 3.7 for Linux - download and install from: https://www.anaconda.com/distribution/
# pytorch, torchvision - install using: conda install pytorch torchvision -c pytorch

import numpy as np
import argparse
import copy
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import tensor_util


parser = argparse.ArgumentParser()
parser.add_argument("--pb_path", type=str, default="/user/a0132471/Files/bit-bucket/pytorch/jacinto-ai-pytest/data/results/networks/tidl-q8/imagenet1k/jacinto_ai_qat_MobileNetV2_train100/mobilenet_v2_qat_2020-12-13_16-53-07.onnx")
parser.add_argument("--input_nodes", default=("sub_7"), nargs='*')
parser.add_argument("--output_nodes", default=("ArgMax"), nargs='*')
parser.add_argument("--input_shape", type=int,  default=[512,512], nargs='*')
args = parser.parse_args()


def optimize_graph(graph_model_def, frozen_graph_filename):
    transforms = [
    'remove_nodes(op=Identity)',
    'strip_unused_nodes',
    #'fold_constants',
    #'flatten_atrous_conv',
    'fold_constants(clear_output_shapes=true)',
    'fold_batch_norms',
    'fold_old_batch_norms',
    ]
    optimized_graph_def = TransformGraph(graph_model_def, ["sub_7"], ["ArgMax"], transforms )

    optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
        optimized_graph_def, args.input_nodes, args.output_nodes,  # an array of output nodes
        tf.float32.as_datatype_enum)

    for node in optimized_graph_def.node:
        # Modify resize layer
        if (('ResizeBilinear' in node.name) and ('/size' in node.name)) or ('strided_slice_6/_0__cf__0' in node.name) or ('strided_slice_7' in node.name) :
            print(node.name)
            print("old_tensor")
            print(node.attr['value'].tensor.tensor_content)
            print(tensor_util.MakeNdarray(node.attr['value'].tensor))
            new_bytes = (np.frombuffer(node.attr['value'].tensor.tensor_content, dtype=np.int32) - 1).tobytes()
            #new_bytes = ((np.frombuffer(node.attr['value'].tensor.tensor_content, dtype=np.int32) - 1) // np.array((2, 4), dtype=np.int32)).tobytes()
            print("new_tensor")
            print(new_tensor)
            print(np.frombuffer(new_bytes, dtype=np.int32))
            node.attr['value'].tensor.tensor_content=new_bytes
        #Modify AvgPool layer
        if node.name == "AvgPool2D/AvgPool":
            print(node.name)
            node.attr['ksize'].list.i.remove(1)
            node.attr['ksize'].list.i.remove(65) #65 129 33
            node.attr['ksize'].list.i.remove(65) #65 257 33
            node.attr['ksize'].list.i.append(64) #64 128 32
            node.attr['ksize'].list.i.append(64) #64 256 32
            node.attr['ksize'].list.i.append(1)
            edit_stride=True
            if edit_stride:
                node.attr['strides'].list.i.remove(1)
                node.attr['strides'].list.i.remove(65) #65 129
                node.attr['strides'].list.i.remove(65) #65 257
                node.attr['strides'].list.i.append(1) #64
                node.attr['strides'].list.i.append(1) #64
                node.attr['strides'].list.i.append(1)
        #Modify d/w conv layer
        if 'SpaceToBatchND/paddings' in node.name or 'BatchToSpaceND/crops' in node.name:
            print(node.name)
            #print(tensor_util.MakeNdarray(node.attr['value'].tensor))
            old_tensor = np.frombuffer(node.attr['value'].tensor.tensor_content, dtype=np.int32)
            new_tensor = copy.deepcopy(old_tensor)
            if 'aspp' in node.name:
                new_tensor[1], new_tensor[3] = new_tensor[1]+1, new_tensor[3]+1
            else:
                new_tensor[1], new_tensor[3] = new_tensor[0], new_tensor[2]
            print("old_tensor")
            print(old_tensor)
            print("new_tensor")
            print(new_tensor)
            new_bytes = new_tensor.tobytes()
            node.attr['value'].tensor.tensor_content = new_bytes
    output_file = frozen_graph_filename.replace(".pb", "_opt_512x512.pb")
    f = tf.gfile.FastGFile(output_file, "w")
    f.write(optimized_graph_def.SerializeToString())
    return output_file


def export_tflite(pb_file):
    output_file = pb_file.replace('pb', 'tflite')
    input_arrays = args.input_nodes
    output_arrays = args.output_nodes
    input_shapes = {input_arrays[0]: [1, args.input_shape[0], args.input_shape[1], 3]}

    # Converting a GraphDef from file.
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_file, input_arrays, output_arrays,
                                                                    input_shapes=input_shapes)
    tflite_model = converter.convert()
    open(output_file, "wb").write(tflite_model)


def main():
    frozen_graph_filename = "/user/a0132471/Files/utils/cloud_models/segmentation/deeplabv3_xception_pascal_train_aug/frozen_inference_graph.pb"
                            #"/user/a0132471/Files/utils/cloud_models/segmentation/deeplabv3_mnv2_pascal_trainaug/frozen_inference_graph.pb"
                            #"/user/a0132471/Files/utils/cloud_models/segmentation/deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb"
    graph_model = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.Open(frozen_graph_filename, "rb") as f:
         data2read = f.read()
         graph_model.ParseFromString(data2read)

    output_file = optimize_graph(graph_model, frozen_graph_filename)
    export_tflite(output_file)

if __name__ == "__main__":
    main()