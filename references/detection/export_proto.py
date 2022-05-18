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
import copy
import os
import numpy as np
import onnx
import torch
from google.protobuf import text_format
from torch.onnx import symbolic_helper as torch_symbolic_helper
from torch.onnx import symbolic_registry as torch_symbolic_registry

from torchvision import models

basename = os.path.splitext(os.path.basename(__file__))[0]
if __name__.startswith(basename):
    import detection_proto.detection_meta_arch_pb2 as detection_meta_arch_pb2
    import utils
else:
    from .detection_proto import detection_meta_arch_pb2 as detection_meta_arch_pb2
    from . import utils
#

__all__ = ['export_model_proto']


def export_model_proto(cfg, model, example_input, output_onnx_file, input_names, output_names, output_names_proto, opset_version, add_postproc_op=True):
    # configure_forward is used to export a model without preprocess
    model.configure_forward(with_preprocess=False, with_postprocess=True, with_intermediate_outputs=True)
    torch.onnx.export(
        model,
        example_input,
        output_onnx_file,
        input_names=input_names,
        output_names=None,
        # export_params=True,
        # keep_initializers_as_inputs=True,
        # do_constant_folding=False,
        # verbose=False,
        # training=torch.onnx.TrainingMode.PRESERVE,
        opset_version=opset_version)
    onnx_model = onnx.load(output_onnx_file)
    # remove the intermediate feature names that were only required to get the names to generate the prototxt
    feature_names = [node.name for node in onnx_model.graph.output[3:]]
    for opt in onnx_model.graph.output[3:]:
        onnx_model.graph.output.remove(opt)
    #
    # change the model output to those given in output_names
    for opt_name, opt in zip(output_names, onnx_model.graph.output[:3]):
        name_changed = False
        for node in onnx_model.graph.node:
            for node_o_idx in range(len(node.output)):
                if node.output[node_o_idx] == opt.name:
                    node.output[node_o_idx] = opt_name
                    opt.name = opt_name
                    name_changed = True
                    break
                #
            #
            if name_changed:
                break
            #
        #
    #
    # make model
    opset = onnx.OperatorSetIdProto()
    opset.version = opset_version
    onnx_model = onnx.helper.make_model(onnx_model.graph, opset_imports=[opset])
    # check model and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_onnx_file)
    # shape inference to make it easy for inference
    onnx.shape_inference.infer_shapes_path(output_onnx_file, output_onnx_file)
    # export proto
    _save_model_proto(cfg, model, example_input, output_filename=output_onnx_file,
                      feature_names=feature_names, output_names=output_names_proto)

    # # export prototxt file for detection model_copy without preprocess or postprocess
    # output_onnx_file_ext = os.path.splitext(output_onnx_file)
    # output_onnxproto_file = output_onnx_file_ext[0] + '-proto' + output_onnx_file_ext[1]
    # model.configure_forward(with_preprocess=False, with_postprocess=False, with_intermediate_outputs=True)
    # torch.onnx.export(
    #     model,
    #     example_input,
    #     output_onnxproto_file,
    #     input_names=input_names,
    #     output_names=None,
    #     # export_params=True,
    #     # keep_initializers_as_inputs=True,
    #     # do_constant_folding=False,
    #     # verbose=False,
    #     # training=torch.onnx.TrainingMode.PRESERVE,
    #     opset_version=opset_version)
    # onnx_model = onnx.load(output_onnxproto_file)
    # feature_names = [node.name for node in onnx_model.graph.output]
    # _save_model_proto(cfg, model, example_input, output_filename=output_onnxproto_file,
    #                   feature_names=feature_names, output_names=output_names_proto)
    # # add the postprocessing operator
    # if add_postproc_op:
    #     _add_dummydetection_operator(cfg, model, output_onnxproto_file, feature_names, opset_version=opset_version)
    #     # _add_detection_operator(cfg, model, output_onnxproto_file, feature_names, opset_version=opset_version)
    # #
    # # shape inference to make it easy for inference
    # onnx.shape_inference.infer_shapes_path(output_onnxproto_file, output_onnxproto_file)

    # restore the model
    model.configure_forward()


def _save_model_proto(cfg, model, input_tensor, output_filename, input_names = ('images',), feature_names=None, output_names=None):
    model = model.module if utils.is_quant_module(model) else model
    model = model.module if utils.is_parallel_module(model) else model
    is_ssd = isinstance(model, models.detection.SSD)
    is_retinanet = isinstance(model, models.detection.RetinaNet)
    if is_ssd:
        _save_proto_ssd(cfg, model, input_tensor, output_filename, input_names, feature_names, output_names)
    elif is_retinanet:
        _save_proto_retinanet(cfg, model, input_tensor, output_filename, input_names, feature_names, output_names)
    #
    return feature_names


###########################################################
def _add_detection_operator(cfg, model, onnx_filename, feature_names, output_names=('boxes', 'scores', 'labels'), opset_version=11):
    # get parameters for post processing
    num_classes = model.head.num_classes
    share_location = True
    background_label_id = 0
    confidence_threshold = model.score_thresh
    top_k = model.topk_candidates
    nms_threshold = model.nms_thresh
    keep_top_k = model.detections_per_img
    # load the onnx model
    onnx_model = onnx.load(onnx_filename)

    num_feature_names = len(feature_names)//2
    cls_feature_names = feature_names[:num_feature_names]
    reg_feature_names = feature_names[num_feature_names:]

    reshape_cls_nodes = []
    reshape_reg_nodes = []
    reshape_cls = []
    reshape_reg = []
    for cls_idx, (cls_feature_name, reg_feature_name) in enumerate(zip(cls_feature_names, reg_feature_names)):
        onnx_model.graph.initializer.extend([
            onnx.helper.make_tensor(f'reshape_{cls_feature_name}_shape', onnx.helper.TensorProto.INT64, dims=[3], vals=[1,-1,num_classes]),
            onnx.helper.make_tensor(f'reshape_{reg_feature_name}_shape', onnx.helper.TensorProto.INT64, dims=[3], vals=[1,-1,4])
        ])
        reshape_cls_node = onnx.helper.make_node(op_type='Reshape', name=f'reshape_{cls_feature_name}_node',
             inputs=[cls_feature_name, f'reshape_{cls_feature_name}_shape'], outputs=[f'reshape_{cls_feature_name}'])
        reshape_reg_node = onnx.helper.make_node(op_type='Reshape', name=f'reshape_{reg_feature_name}_node',
             inputs=[reg_feature_name, f'reshape_{reg_feature_name}_shape'], outputs=[f'reshape_{reg_feature_name}'])
        reshape_cls_nodes.append(reshape_cls_node)
        reshape_reg_nodes.append(reshape_reg_node)
        reshape_cls.append(f'reshape_{cls_feature_name}')
        reshape_reg.append(f'reshape_{reg_feature_name}')
    #
    concat_reshape_cls_node = onnx.helper.make_node(op_type='Concat', name=f'concat_reshape_cls_node', inputs=reshape_cls, outputs=['concat_reshape_cls'], axis=1)
    concat_reshape_reg_node = onnx.helper.make_node(op_type='Concat', name=f'concat_reshape_reg_node', inputs=reshape_reg, outputs=['concat_reshape_reg'], axis=1)

    # detection post process
    num_classes_tensor = onnx.helper.make_tensor('num_classes', onnx.helper.TensorProto.INT64, dims=[1], vals=[num_classes])
    share_location_tensor = onnx.helper.make_tensor('share_location', onnx.helper.TensorProto.BOOL, dims=[1], vals=[share_location])
    background_label_id_tensor = onnx.helper.make_tensor('background_label_id', onnx.helper.TensorProto.INT64, dims=[1], vals=[background_label_id])
    confidence_threshold_tensor = onnx.helper.make_tensor('confidence_threshold', onnx.helper.TensorProto.FLOAT, dims=[1], vals=[confidence_threshold])
    top_k_tensor = onnx.helper.make_tensor('top_k', onnx.helper.TensorProto.INT64, dims=[1], vals=[top_k])
    nms_threshold_tensor = onnx.helper.make_tensor('nms_threshold', onnx.helper.TensorProto.FLOAT, dims=[1], vals=[nms_threshold])
    keep_top_k_tensor = onnx.helper.make_tensor('keep_top_k', onnx.helper.TensorProto.INT64, dims=[1], vals=[keep_top_k])
    onnx_model.graph.initializer.extend([num_classes_tensor, share_location_tensor,
                                         background_label_id_tensor, confidence_threshold_tensor,
                                         top_k_tensor, nms_threshold_tensor, keep_top_k_tensor])
    # create output info
    assert 'boxes' in output_names, 'boxes must be in output_names'
    assert 'scores' in output_names, 'scores must be in output_names'
    assert 'labels' in output_names, 'labels must be in output_names'
    boxes = onnx.helper.make_tensor_value_info('boxes', onnx.helper.TensorProto.FLOAT, shape=[1,keep_top_k,4])
    scores = onnx.helper.make_tensor_value_info('scores', onnx.helper.TensorProto.FLOAT, shape=[1,keep_top_k,1])
    labels = onnx.helper.make_tensor_value_info('labels', onnx.helper.TensorProto.FLOAT, shape=[1,keep_top_k,1])
    # make node
    detection_node = onnx.helper.make_node(
        op_type='TIDL_Detection_PostProcess', # domain='torchvision',
        inputs=['concat_reshape_cls', 'concat_reshape_reg'],
        outputs=['boxes', 'scores', 'labels']
    )
    onnx_model.graph.node.extend([*reshape_cls_nodes, *reshape_reg_nodes,
                                  concat_reshape_cls_node, concat_reshape_reg_node,
                                  detection_node])
    # set output
    graph_outputs = [value for value in onnx_model.graph.output]
    for g_idx, g_tensor in enumerate(graph_outputs):
        onnx_model.graph.output.remove(g_tensor)
    #
    onnx_model.graph.output.extend([boxes, scores, labels])
    # make model
    opset = onnx.OperatorSetIdProto()
    opset.version = opset_version
    onnx_model = onnx.helper.make_model(onnx_model.graph, opset_imports=[opset])
    # onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_filename)


def _add_dummydetection_operator(cfg, model, onnx_filename, feature_names, output_names=('boxes', 'scores', 'labels'), opset_version=11):
    # get parameters for post processing
    num_classes = model.head.num_classes
    share_location = True
    background_label_id = 0
    confidence_threshold = model.score_thresh
    top_k = model.topk_candidates
    nms_threshold = model.nms_thresh
    keep_top_k = model.detections_per_img
    # load the onnx model
    onnx_model = onnx.load(onnx_filename)

    # create output info
    assert 'boxes' in output_names, 'boxes must be in output_names'
    assert 'scores' in output_names, 'scores must be in output_names'
    assert 'labels' in output_names, 'labels must be in output_names'

    # make node
    dummy_flatten_feature_names = []
    for feature_idx, feature_name in enumerate(feature_names):
        dummy_flatten_feature_name = f'dummy_flatten_{feature_name}'
        dummy_flatten_node = onnx.helper.make_node(op_type='Flatten', name=f'dummy_{dummy_flatten_feature_name}_node',
            inputs=[feature_name], outputs=[dummy_flatten_feature_name], axis=1)
        onnx_model.graph.node.append(dummy_flatten_node)
        dummy_flatten_feature_names.append(dummy_flatten_feature_name)
    #
    dummy_concat_node = onnx.helper.make_node(op_type='Concat', name=f'dummy_concat_flatten_features_node',
            inputs=dummy_flatten_feature_names, outputs=['dummy_concat_featrues'], axis=-1)

    onnx_model.graph.initializer.extend([
        onnx.helper.make_tensor('dummy_boxes_slice_starts', onnx.helper.TensorProto.INT64, dims=[1], vals=[0]),
        onnx.helper.make_tensor('dummy_boxes_slice_ends', onnx.helper.TensorProto.INT64, dims=[1], vals=[keep_top_k*4]),
        onnx.helper.make_tensor('dummy_boxes_slice_axes', onnx.helper.TensorProto.INT64, dims=[1], vals=[1]),
        onnx.helper.make_tensor('dummy_boxes_shape', onnx.helper.TensorProto.INT64, dims=[2], vals=[keep_top_k,4])
    ])
    dummy_boxes_slice_node = onnx.helper.make_node(op_type='Slice', name='dummy_boxes_slice_node',
            inputs=['dummy_concat_featrues', 'dummy_boxes_slice_starts', 'dummy_boxes_slice_ends', 'dummy_boxes_slice_axes'],
            outputs=['dummy_boxes_slice'])
    dummy_boxes_node = onnx.helper.make_node(op_type='Reshape', name='dummy_boxes_reshape', inputs=['dummy_boxes_slice', 'dummy_boxes_shape'],
            outputs=['boxes'])
    boxes = onnx.helper.make_tensor_value_info('boxes', onnx.helper.TensorProto.FLOAT, shape=[keep_top_k,4])

    onnx_model.graph.initializer.extend([
        onnx.helper.make_tensor('dummy_scores_slice_starts', onnx.helper.TensorProto.INT64, dims=[1], vals=[0]),
        onnx.helper.make_tensor('dummy_scores_slice_ends', onnx.helper.TensorProto.INT64, dims=[1], vals=[keep_top_k]),
        onnx.helper.make_tensor('dummy_scores_slice_axes', onnx.helper.TensorProto.INT64, dims=[1], vals=[1]),
        onnx.helper.make_tensor('dummy_scores_shape', onnx.helper.TensorProto.INT64, dims=[2], vals=[keep_top_k,1])
    ])
    dummy_scores_slice_node = onnx.helper.make_node(op_type='Slice', name='dummy_scores_slice_node',
            inputs=['dummy_concat_featrues', 'dummy_scores_slice_starts', 'dummy_scores_slice_ends', 'dummy_scores_slice_axes'],
            outputs=['dummy_scores_slice'])
    dummy_scores_node = onnx.helper.make_node(op_type='Reshape', name='dummy_scores_reshape_node',
            inputs=['dummy_scores_slice', 'dummy_scores_shape'], outputs=['scores'])
    scores = onnx.helper.make_tensor_value_info('scores', onnx.helper.TensorProto.FLOAT, shape=[keep_top_k,1])

    onnx_model.graph.initializer.extend([
        onnx.helper.make_tensor('dummy_labels_slice_starts', onnx.helper.TensorProto.INT64, dims=[1], vals=[0]),
        onnx.helper.make_tensor('dummy_labels_slice_ends', onnx.helper.TensorProto.INT64, dims=[1], vals=[keep_top_k]),
        onnx.helper.make_tensor('dummy_labels_slice_axes', onnx.helper.TensorProto.INT64, dims=[1], vals=[1]),
        onnx.helper.make_tensor('dummy_labels_shape', onnx.helper.TensorProto.INT64, dims=[2], vals=[keep_top_k,1])
    ])
    dummy_labels_slice_node = onnx.helper.make_node(op_type='Slice', name='dummy_labels_slice_node',
            inputs=['dummy_concat_featrues', 'dummy_labels_slice_starts', 'dummy_labels_slice_ends', 'dummy_labels_slice_axes'],
            outputs=['dummy_labels_slice'])
    dummy_labels_node_float = onnx.helper.make_node(op_type='Reshape', name='dummy_labels_reshape_node',
            inputs=['dummy_labels_slice', 'dummy_labels_shape'], outputs=['dummy_labels_float'])
    dummy_labels_node = onnx.helper.make_node(op_type='Cast', name='dummy_labels_cast_node',
            inputs=['dummy_labels_float'], to=onnx.helper.TensorProto.INT64, outputs=['labels'])
    labels = onnx.helper.make_tensor_value_info('labels', onnx.helper.TensorProto.INT64, shape=[keep_top_k,1])

    onnx_model.graph.node.extend([dummy_concat_node,
                                  dummy_boxes_slice_node, dummy_boxes_node,
                                  dummy_scores_slice_node, dummy_scores_node,
                                  dummy_labels_slice_node, dummy_labels_node_float, dummy_labels_node])

    # set output
    graph_outputs = [value for value in onnx_model.graph.output]
    for g_idx, g_tensor in enumerate(graph_outputs):
        onnx_model.graph.output.remove(g_tensor)
    #
    onnx_model.graph.output.extend([boxes, scores, labels])
    # make model
    opset = onnx.OperatorSetIdProto()
    opset.version = opset_version
    onnx_model = onnx.helper.make_model(onnx_model.graph, opset_imports=[opset])
    # check model and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_filename)


###########################################################
def _save_proto_ssd(cfg, model, input_tensor, output_filename, input_names=None, feature_names=None, output_names=None,
                    use_sigmoid_cls=False):
    input_size = input_tensor.size()
    # parameters
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    num_feature_names = len(feature_names)//2
    cls_feature_names = feature_names[:num_feature_names]
    reg_feature_names = feature_names[num_feature_names:]
    head = model.head
    anchor_generator = model.anchor_generator
    input_size_ssd = max(input_size[-2:])
    num_classes = head.num_classes
    background_label_id = -1 if use_sigmoid_cls else 0 #head.num_classes-1
    confidence_threshold = 0.3 #model.score_thresh
    nms_threshold = model.nms_thresh
    top_k = 300 #model.topk_candidates
    keep_top_k = 300 #model.detections_per_img

    prior_box_param = []
    for h_idx in range(num_feature_names):
        min_size=[anchor_generator.scales[h_idx]*input_size_ssd]
        max_size=[anchor_generator.scales[h_idx+1]*input_size_ssd]
        aspect_ratio=anchor_generator.aspect_ratios[h_idx]
        variance = [1.0/v for v in model.box_coder.weights]
        priorbox_kwargs = dict(min_size=min_size, max_size=max_size, aspect_ratio=aspect_ratio,
                               variance=variance, clip=False, flip=True)
        step = anchor_generator.get_steps()[h_idx]
        if isinstance(step, (list,tuple)):
            priorbox_kwargs.update(dict(step_h=step[0], step_w=step[1]))
        else:
            priorbox_kwargs.update(dict(step=step))
        #
        prior_box_param.append(detection_meta_arch_pb2.PriorBoxParameter(**priorbox_kwargs))
    #
    nms_param = detection_meta_arch_pb2.TIDLNmsParam(nms_threshold=nms_threshold, top_k=top_k)
    detection_output_param = detection_meta_arch_pb2.TIDLOdPostProc(
        num_classes=num_classes, share_location=True, background_label_id=background_label_id, nms_param=nms_param,
        code_type=detection_meta_arch_pb2.CENTER_SIZE, keep_top_k=keep_top_k, confidence_threshold=confidence_threshold)

    ssd = detection_meta_arch_pb2.TidlMaCaffeSsd(
        box_input=reg_feature_names, class_input=cls_feature_names, output=output_names, prior_box_param=prior_box_param,
        in_width=input_size[3], in_height=input_size[2], detection_output_param=detection_output_param)

    arch = detection_meta_arch_pb2.TIDLMetaArch(name='ssd',  caffe_ssd=[ssd])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_proto_retinanet(cfg, model, input_tensor, output_filename, input_names=None, feature_names=None, output_names=None,
                          use_sigmoid_cls=True):
    input_size = input_tensor.size()
    # parameters
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    num_feature_names = len(feature_names)//2
    cls_feature_names = feature_names[:num_feature_names]
    reg_feature_names = feature_names[num_feature_names:]
    head = model.head
    anchor_generator = head.anchor_generator
    num_classes = head.num_classes
    background_label_id = -1 if use_sigmoid_cls else 0 #head.num_classes-1
    confidence_threshold = 0.3
    nms_threshold = 0.45
    top_k = 300
    keep_top_k = 300

    score_converter = detection_meta_arch_pb2.SIGMOID if use_sigmoid_cls else detection_meta_arch_pb2.SOFTMAX
    anchor_param = detection_meta_arch_pb2.RetinaNetAnchorParameter(aspect_ratio=anchor_generator.ratios,
                                                                octave_base_scale=anchor_generator.octave_base_scale,
                                                                scales_per_octave=anchor_generator.scales_per_octave)

    nms_param = detection_meta_arch_pb2.TIDLNmsParam(nms_threshold=nms_threshold, top_k=top_k)
    detection_output_param = detection_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=detection_meta_arch_pb2.CENTER_SIZE, keep_top_k=keep_top_k,
                                            confidence_threshold=confidence_threshold)

    retinanet = detection_meta_arch_pb2.TidlMaRetinaNet(box_input=reg_feature_names, class_input=cls_feature_names, output=output_names,
                                              x_scale=1.0, y_scale=1.0, width_scale=1.0, height_scale=1.0,
                                              in_width=input_size[3], in_height=input_size[2],
                                              score_converter=score_converter, anchor_param=anchor_param,
                                              detection_output_param=detection_output_param)

    arch = detection_meta_arch_pb2.TIDLMetaArch(name='retinanet',  tidl_retinanet=[retinanet])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


#####################################################################################
# export_model_proto

def prepare_model_for_layer_outputs(model_name, model_name_new, export_layer_types=None):
    onnx_model = onnx.load(model_name)
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = ''
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].output)):
            if export_layer_types is None or onnx_model.graph.node[i].op_type in export_layer_types:
                intermediate_layer_value_info.name = onnx_model.graph.node[i].output[0]
                onnx_model.graph.output.append(intermediate_layer_value_info)
            #
        #
    #
    layer_output_names = [out.name for out in onnx_model.graph.output]
    onnx.save(onnx_model, model_name_new)
    return layer_output_names


def similar_tensor(t1, t2, rtol=1.e-5, atol=1.e-5):
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


def retrieve_onnx_names(input_data, full_model_path, partial_model_path):
    import onnxruntime
    full_model_path_tmp = f'{full_model_path}.tmp'
    full_output_names = prepare_model_for_layer_outputs(full_model_path, full_model_path_tmp, export_layer_types=None)
    full_infer = onnxruntime.InferenceSession(full_model_path_tmp)
    full_input_name = full_infer.get_inputs()[0].name

    partial_infer = onnxruntime.InferenceSession(partial_model_path)
    partial_input_name = partial_infer.get_inputs()[0].name
    partial_output_names = [o.name for o in partial_infer.get_outputs()]

    input_numpy = input_data.detach().numpy() if isinstance(input_data, torch.Tensor) else input_data
    full_outputs = full_infer.run(full_output_names, {full_input_name:input_numpy})
    partial_outputs = partial_infer.run(partial_output_names, {partial_input_name:input_numpy})

    matched_names = {}
    for pname, po in zip(partial_output_names, partial_outputs):
        matched_name = None
        for fname, fo in zip(full_output_names, full_outputs):
            if similar_tensor(po, fo):
                matched_name = fname
            #
        #
        if matched_name is None:
            return None
        #
        matched_names[pname] = matched_name
    #
    os.remove(full_model_path_tmp)
    return matched_names
