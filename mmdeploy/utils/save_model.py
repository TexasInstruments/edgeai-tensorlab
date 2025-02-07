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

import os
import numpy as np
import torch
# from .quantize import is_mmdet_quant_module
from .proto import tidl_meta_arch_mmdeploy_pb2
from google.protobuf import text_format

__all__ = ['save_model_proto']


def save_model_proto(cfg, model, onnx_model, input, output_filename, input_names=None, feature_names=None, output_names=None):
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    input = input[0] if isinstance(input, (list,tuple)) and \
                                  isinstance(input[0], (torch.Tensor, np.ndarray)) else input
    input_size = input.size() if isinstance(input, torch.Tensor) else input
    # model = model.module if is_mmdet_quant_module(model) else model
    is_ssd = hasattr(cfg.model, 'bbox_head') and ('SSD' in cfg.model.bbox_head.type)
    is_fcos = hasattr(cfg.model, 'bbox_head') and ('FCOS' in cfg.model.bbox_head.type)
    is_retinanet = hasattr(cfg.model, 'bbox_head') and ('Retina' in cfg.model.bbox_head.type)
    is_yolov3 = hasattr(cfg.model, 'bbox_head') and ('YOLOV3' in cfg.model.bbox_head.type)
    is_yolox = hasattr(cfg.model, 'bbox_head') and ('YOLOXHead' in cfg.model.bbox_head.type or 'YOLOXPoseHead' in cfg.model.bbox_head.type)
    is_efficientdet = hasattr(cfg.model,'bbox_head') and ('EfficientDetSepBNHead' in cfg.model.bbox_head.type)
    is_yoloxpose = hasattr(cfg.model, 'head') and ('YOLOXPoseHead' in cfg.model.head.type)
    is_yolov7 = hasattr(cfg.model, 'bbox_head') and ('YOLOV7Head' in  cfg.model.bbox_head.type)
    input_names = input_names or ('input',)

    if is_ssd:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Transpose',
                                                        return_layer='Conv')
        _save_mmdet_proto_ssd(cfg, model, input_size, output_filename, input_names, feature_names, output_names)
    elif is_retinanet:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Transpose',
                                                        return_layer='Conv')
        _save_mmdet_proto_retinanet(cfg, model, input_size, output_filename, input_names, feature_names, output_names)
    elif is_yolov3:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Transpose',
                                                        return_layer='Conv')
        _save_mmdet_proto_yolov3(cfg, model, input_size, output_filename, input_names, feature_names, output_names)
    elif is_yolox:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Concat',
                                                        return_layer='Concat')
        _save_mmdet_proto_yolox(cfg, model, input_size, output_filename, input_names, feature_names, output_names)
    elif is_efficientdet:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Transpose',
                                                        return_layer='Conv')
        _save_mmdet_proto_efficientdet(cfg, model, input_size, output_filename, input_names, feature_names, output_names)
    elif is_yoloxpose:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Concat',
                                                        return_layer='Concat')
        _save_mmdet_proto_yoloxpose(cfg, model, input_size, output_filename, input_names, feature_names, output_names)
    elif is_yolov7:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Mul', match_layer = 'Reshape',
                                                        return_layer='Mul')
        output_names = ['detections']
        _save_mmyolo_proto_yolov7(cfg, model, input_size, output_filename, input_names, feature_names, output_names)
    #


###########################################################
def _create_rand_inputs(input_size, is_cuda=False):
    x = torch.rand(input_size)
    x = x.cuda() if is_cuda else x
    return x


def _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names=None, proto_names=None, output_names=None,
                     opset_version=11):
    #https://github.com/open-mmlab/mmdetection/pull/1082
    assert hasattr(model, 'forward_dummy'), 'wrting onnx is not supported by this model'
    model.eval()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    forward_backup = model.forward #backup forward
    model.forward = model.forward_dummy #set dummy forward
    torch.onnx.export(
        model,
        input_list,
        output_filename,
        input_names=input_names,
        output_names=proto_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=opset_version)
    model.forward = forward_backup #restore forward

###########################################################
def prepare_model_for_layer_outputs(onnx_model, export_layer_types=None, match_layer=None, return_layer=None):
    layer_output_names = []
    for i in range(len(onnx_model.graph.node)):
        output_0 = onnx_model.graph.node[i].output[0]
        if onnx_model.graph.node[i].op_type in export_layer_types:
            for j in range(len(onnx_model.graph.node)):
                if output_0 in onnx_model.graph.node[j].input:
                    if onnx_model.graph.node[j].op_type == 'QuantizeLinear':
                        output_0 = onnx_model.graph.node[j].output[0]
                        for k in range(len(onnx_model.graph.node)):
                             if output_0 in onnx_model.graph.node[k].input and onnx_model.graph.node[k].op_type == 'DequantizeLinear':
                                output_0 = onnx_model.graph.node[k].output[0]
                    elif onnx_model.graph.node[j].op_type == match_layer:
                        if return_layer not in export_layer_types:
                            if onnx_model.graph.node[j].output[0] not in layer_output_names:
                                layer_output_names.append(onnx_model.graph.node[j].output[0])
                        else:
                            if onnx_model.graph.node[i].output[0] not in layer_output_names:
                                layer_output_names.append(onnx_model.graph.node[i].output[0])
    return layer_output_names

###########################################################
def _save_mmdet_proto_ssd(cfg, model, input_size, output_filename, input_names=None, proto_names=None, output_names=None):
    num_proto_names = len(proto_names)//2
    cls_proto_names=[]
    reg_proto_names=[]
    for i in range(num_proto_names):
        cls_proto_names.append(proto_names[i*2])
        reg_proto_names.append(proto_names[i*2+1])
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.anchor_generator

    prior_box_param = []
    for h_idx in range(num_proto_names):
        min_size=[anchor_generator.min_sizes[h_idx]]
        max_size=[anchor_generator.max_sizes[h_idx]]
        aspect_ratio=anchor_generator.ratios[h_idx][2::2]
        step=anchor_generator.strides[h_idx]
        step=step[0] if isinstance(step,(tuple,list)) else step
        prior_box_param.append(tidl_meta_arch_mmdeploy_pb2.PriorBoxParameter(min_size=min_size, max_size=max_size,
                                                                     aspect_ratio=aspect_ratio, step=step,
                                                                     variance=bbox_head.bbox_coder.stds, clip=False, flip=True))
    #

    nms_param = tidl_meta_arch_mmdeploy_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = tidl_meta_arch_mmdeploy_pb2.TIDLOdPostProc(num_classes=bbox_head.num_classes+1, share_location=True,
                                            background_label_id=bbox_head.num_classes, nms_param=nms_param,
                                            code_type=tidl_meta_arch_mmdeploy_pb2.CENTER_SIZE, keep_top_k=200,
                                            confidence_threshold=0.3)

    ssd = tidl_meta_arch_mmdeploy_pb2.TidlMaCaffeSsd(box_input=reg_proto_names, class_input=cls_proto_names,
                                             output=output_names, prior_box_param=prior_box_param,
                                             in_width=input_size[3], in_height=input_size[2],
                                             detection_output_param=detection_output_param,
                                             framework='MMDetection')

    arch = tidl_meta_arch_mmdeploy_pb2.TIDLMetaArch(name='ssd',  caffe_ssd=[ssd])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmdet_proto_retinanet(cfg, model, input_size, output_filename, input_names=None, proto_names=None, output_names=None):
    num_proto_names = len(proto_names)//2
    cls_proto_names=[]
    reg_proto_names=[]
    for i in range(num_proto_names):
        cls_proto_names.append(proto_names[i*2])
        reg_proto_names.append(proto_names[i*2+1])
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.anchor_generator

    background_label_id = -1 if bbox_head.use_sigmoid_cls else bbox_head.num_classes
    num_classes = bbox_head.num_classes if bbox_head.use_sigmoid_cls else bbox_head.num_classes+1
    score_converter = tidl_meta_arch_mmdeploy_pb2.SIGMOID if bbox_head.use_sigmoid_cls else tidl_meta_arch_mmdeploy_pb2.SOFTMAX
    anchor_param = tidl_meta_arch_mmdeploy_pb2.RetinaNetAnchorParameter(aspect_ratio=anchor_generator.ratios,
                                                                octave_base_scale=anchor_generator.octave_base_scale,
                                                                scales_per_octave=anchor_generator.scales_per_octave)

    nms_param = tidl_meta_arch_mmdeploy_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = tidl_meta_arch_mmdeploy_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=tidl_meta_arch_mmdeploy_pb2.CENTER_SIZE, keep_top_k=200,
                                            confidence_threshold=0.3)

    retinanet = tidl_meta_arch_mmdeploy_pb2.TidlMaRetinaNet(box_input=reg_proto_names, class_input=cls_proto_names,
                                              output=output_names, x_scale=1.0, y_scale=1.0, width_scale=1.0, height_scale=1.0,
                                              in_width=input_size[3], in_height=input_size[2],
                                              score_converter=score_converter, anchor_param=anchor_param,
                                              detection_output_param=detection_output_param,
                                              framework='MMDetection')

    arch = tidl_meta_arch_mmdeploy_pb2.TIDLMetaArch(name='retinanet',  tidl_retinanet=[retinanet])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)

##########################################################
def _save_mmdet_proto_efficientdet(cfg, model, input_size, output_filename, input_names=None, proto_names=None, output_names=None):
    num_proto_names = len(proto_names)//2
    cls_proto_names = proto_names[num_proto_names:]
    reg_proto_names = proto_names[:num_proto_names]
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.anchor_generator

    background_label_id = -1 if bbox_head.use_sigmoid_cls else bbox_head.num_classes
    num_classes = bbox_head.num_classes if bbox_head.use_sigmoid_cls else bbox_head.num_classes+1
    score_converter = tidl_meta_arch_mmdeploy_pb2.SIGMOID if bbox_head.use_sigmoid_cls else tidl_meta_arch_mmdeploy_pb2.SOFTMAX
    anchor_param = tidl_meta_arch_mmdeploy_pb2.RetinaNetAnchorParameter(aspect_ratio=anchor_generator.ratios,
                                                                octave_base_scale=anchor_generator.octave_base_scale,
                                                                scales_per_octave=anchor_generator.scales_per_octave)

    nms_param = tidl_meta_arch_mmdeploy_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = tidl_meta_arch_mmdeploy_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=tidl_meta_arch_mmdeploy_pb2.CENTER_SIZE, keep_top_k=200,
                                            confidence_threshold=0.3)

    retinanet = tidl_meta_arch_mmdeploy_pb2.TidlMaRetinaNet(box_input=reg_proto_names, class_input=cls_proto_names,
                                              output=output_names, x_scale=1.0, y_scale=1.0, width_scale=1.0, height_scale=1.0,
                                              in_width=input_size[3], in_height=input_size[2],
                                              score_converter=score_converter, anchor_param=anchor_param,
                                              detection_output_param=detection_output_param,
                                              framework='MMDetection')

    arch = tidl_meta_arch_mmdeploy_pb2.TIDLMetaArch(name='retinanet',  tidl_retinanet=[retinanet])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmdet_proto_yolov3(cfg, model, input_size, output_filename, input_names=None, proto_names=None, output_names=None):
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.prior_generator
    base_sizes = anchor_generator.base_sizes

    background_label_id = -1
    num_classes = bbox_head.num_classes
    #score_converter = tidl_meta_arch_mmdeploy_pb2.SIGMOID

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = tidl_meta_arch_mmdeploy_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[b[0] for b in base_size],
                                                        anchor_height=[b[1] for b in base_size])
        yolo_params.append(yolo_param)

    nms_param = tidl_meta_arch_mmdeploy_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = tidl_meta_arch_mmdeploy_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=tidl_meta_arch_mmdeploy_pb2.CENTER_SIZE_EXP, keep_top_k=200,
                                            confidence_threshold=0.3)

    yolov3 = tidl_meta_arch_mmdeploy_pb2.TidlYoloOd(name='yolo_v3', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            framework='MMDetection')

    arch = tidl_meta_arch_mmdeploy_pb2.TIDLMetaArch(name='yolo_v3',  tidl_yolo=[yolov3])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmdet_proto_yolox(cfg, model, input_size, output_filename, input_names=None, proto_names=None, output_names=None):
    bbox_head = model.bbox_head
    # anchor_generator = bbox_head.anchor_generator
    base_sizes = model.bbox_head.strides

    background_label_id = -1
    num_classes = bbox_head.num_classes
    #score_converter = tidl_meta_arch_mmdeploy_pb2.SIGMOID

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = tidl_meta_arch_mmdeploy_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[base_size],
                                                        anchor_height=[base_size])
        yolo_params.append(yolo_param)

    nms_param = tidl_meta_arch_mmdeploy_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = tidl_meta_arch_mmdeploy_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=tidl_meta_arch_mmdeploy_pb2.CODE_TYPE_YOLO_X, keep_top_k=200,
                                            confidence_threshold=0.3)

    yolox = tidl_meta_arch_mmdeploy_pb2.TidlYoloOd(name='yolox', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            framework='MMDetection')

    arch = tidl_meta_arch_mmdeploy_pb2.TIDLMetaArch(name='yolox',  tidl_yolo=[yolox])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmdet_proto_yoloxpose(cfg, model, input_size, output_filename, input_names=None, proto_names=None, output_names=None):
    bbox_head = model.head
    # anchor_generator = bbox_head.anchor_generator
    base_sizes = model.head.featmap_strides

    background_label_id = -1
    num_classes = model.head.head_module.num_classes
    #score_converter = tidl_meta_arch_mmdeploy_pb2.SIGMOID
    num_keypoint = model.head.num_keypoints if hasattr(model.head, "num_keypoints") else None
    keypoint_confidence = True if (num_keypoint is not None and num_keypoint>0) else None

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = tidl_meta_arch_mmdeploy_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[base_size],
                                                        anchor_height=[base_size])
        yolo_params.append(yolo_param)

    nms_param = tidl_meta_arch_mmdeploy_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = tidl_meta_arch_mmdeploy_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=tidl_meta_arch_mmdeploy_pb2.CODE_TYPE_YOLO_X, keep_top_k=200,
                                            confidence_threshold=0.3,num_keypoint=num_keypoint,keypoint_confidence=keypoint_confidence)

    yolox = tidl_meta_arch_mmdeploy_pb2.TidlYoloOd(name='yolox', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,)

    arch = tidl_meta_arch_mmdeploy_pb2.TIDLMetaArch(name='yolox',  tidl_yolo=[yolox])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmyolo_proto_yolov7(cfg, model, input_size, output_filename, input_names=None, proto_names=None, output_names=None):
    bbox_head = model.bbox_head
    base_sizes = model.bbox_head.anchor_cfg.anchor

    background_label_id = -1
    num_classes = bbox_head.num_classes

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = tidl_meta_arch_mmdet_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[base_size[idx*2] for idx in range(len(base_size)//2)],
                                                        anchor_height=[base_size[idx*2+1] for idx in range(len(base_size)//2)],
                                                         )
        yolo_params.append(yolo_param)

    nms_param = tidl_meta_arch_mmdet_pb2.TIDLNmsParam(nms_threshold=0.65, top_k=200)
    detection_output_param = tidl_meta_arch_mmdet_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=tidl_meta_arch_mmdet_pb2.CODE_TYPE_YOLO_V5, keep_top_k=200,
                                            confidence_threshold=0.001)

    yolov7 = tidl_meta_arch_mmdet_pb2.TidlYoloOd(name='yolov7', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param)

    arch = tidl_meta_arch_mmdet_pb2.TIDLMetaArch(name='yolov7',  tidl_yolo=[yolov7])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)