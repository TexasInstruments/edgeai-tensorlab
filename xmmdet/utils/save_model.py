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
import torch
from .quantize import is_mmdet_quant_module
from .proto import mmdet_meta_arch_pb2
from google.protobuf import text_format

__all__ = ['save_model_proto']


def save_model_proto(cfg, model, input, output_filename, output_names=None, save_onnx=True, opset_version=11):
    is_cuda = next(model.parameters()).is_cuda
    input_list = input if isinstance(input, torch.Tensor) else _create_rand_inputs(input, is_cuda)
    input_size = input.size() if isinstance(input, torch.Tensor) else input
    model = model.module if is_mmdet_quant_module(model) else model
    is_ssd = hasattr(cfg.model, 'bbox_head') and ('SSD' in cfg.model.bbox_head.type)
    is_fcos = hasattr(cfg.model, 'bbox_head') and ('FCOS' in cfg.model.bbox_head.type)
    is_retinanet = hasattr(cfg.model, 'bbox_head') and ('Retina' in cfg.model.bbox_head.type)
    is_yolov3 = hasattr(cfg.model, 'bbox_head') and ('YOLOV3' in cfg.model.bbox_head.type)
    if is_ssd:
        input_names = ['input']
        if output_names is None:
            output_names = []
            for cls_idx, cls in enumerate(model.bbox_head.cls_convs):
                output_names.append(f'cls_convs_{cls_idx}')
            #
            for reg_idx, reg in enumerate(model.bbox_head.reg_convs):
                output_names.append(f'reg_convs_{reg_idx}')
            #
        #
        _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names, output_names,
                         opset_version=opset_version, save_onnx=save_onnx)
        _save_mmdet_proto_ssd(cfg, model, input_size, output_filename, input_names, output_names)
    elif is_retinanet:
        input_names = ['input']
        if output_names is None:
            output_names = []
            for i in range(model.neck.num_outs):
                output_names.append(f'retina_cls_{i}')
            #
            for i in range(model.neck.num_outs):
                output_names.append(f'retina_reg_{i}')
            #
        #
        _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names, output_names,
                         opset_version=opset_version, save_onnx=save_onnx)
        _save_mmdet_proto_retinanet(cfg, model, input_size, output_filename, input_names, output_names)
    elif is_yolov3:
        input_names = ['input']
        if output_names is None:
            output_names = []
            for i in range(model.neck.num_scales):
                output_names.append(f'convs_pred_{i}')
            #
        #
        _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names, output_names,
                         opset_version=opset_version, save_onnx=save_onnx)
        _save_mmdet_proto_yolov3(cfg, model, input_size, output_filename, input_names, output_names)
    elif is_fcos:
        input_names = ['input']
        if output_names is None:
            output_names = []
            for i in range(model.neck.num_outs):
                output_names.append(f'cls_convs_{i}')
            #
            for i in range(model.neck.num_outs):
                output_names.append(f'reg_convs_{i}')
            #
            for i in range(model.neck.num_outs):
                output_names.append(f'centerness_convs_{i}')
            #
        #
        _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names, output_names,
                         opset_version=opset_version, save_onnx=save_onnx)
    else:
        _save_mmdet_onnx(cfg, model, input_list, output_filename,
                         opset_version=opset_version, save_onnx=save_onnx)
    #


###########################################################
def _create_rand_inputs(input_size, is_cuda=False):
    x = torch.rand(input_size)
    x = x.cuda() if is_cuda else x
    return x


def _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names=None, output_names=None,
                     opset_version=11, save_onnx=True):
    if not save_onnx:
        return
    #
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
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=opset_version)
    model.forward = forward_backup #restore forward


###########################################################
def _save_mmdet_proto_ssd(cfg, model, input_size, output_filename, input_names=None, output_names=None):
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    num_output_names = len(output_names)//2
    cls_output_names = output_names[:num_output_names]
    reg_output_names = output_names[num_output_names:]
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.anchor_generator

    prior_box_param = []
    for h_idx in range(num_output_names):
        min_size=[anchor_generator.min_sizes[h_idx]]
        max_size=[anchor_generator.max_sizes[h_idx]]
        aspect_ratio=anchor_generator.ratios[h_idx][2::2]
        step=anchor_generator.strides[h_idx]
        step=step[0] if isinstance(step,(tuple,list)) else step
        prior_box_param.append(mmdet_meta_arch_pb2.PriorBoxParameter(min_size=min_size, max_size=max_size,
                                                                     aspect_ratio=aspect_ratio, step=step,
                                                                     variance=bbox_head.bbox_coder.stds, clip=False, flip=True))
    #

    nms_param = mmdet_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=100)
    detection_output_param = mmdet_meta_arch_pb2.TIDLOdPostProc(num_classes=bbox_head.num_classes+1, share_location=True,
                                            background_label_id=bbox_head.num_classes, nms_param=nms_param,
                                            code_type=mmdet_meta_arch_pb2.CENTER_SIZE, keep_top_k=100,
                                            confidence_threshold=0.5)

    ssd = mmdet_meta_arch_pb2.TidlMaCaffeSsd(box_input=reg_output_names, class_input=cls_output_names,
                                             output=['boxes', 'labels'], prior_box_param=prior_box_param,
                                             in_width=input_size[3], in_height=input_size[2],
                                             detection_output_param=detection_output_param,
                                             framework='MMDetection')

    arch = mmdet_meta_arch_pb2.TIDLMetaArch(name='ssd',  caffe_ssd=[ssd])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmdet_proto_retinanet(cfg, model, input_size, output_filename, input_names=None, output_names=None):
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    num_output_names = len(output_names)//2
    cls_output_names = output_names[:num_output_names]
    reg_output_names = output_names[num_output_names:]
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.anchor_generator

    background_label_id = -1 if bbox_head.use_sigmoid_cls else bbox_head.num_classes
    num_classes = bbox_head.num_classes if bbox_head.use_sigmoid_cls else bbox_head.num_classes+1
    score_converter = mmdet_meta_arch_pb2.SIGMOID if bbox_head.use_sigmoid_cls else mmdet_meta_arch_pb2.SOFTMAX
    anchor_param = mmdet_meta_arch_pb2.RetinaNetAnchorParameter(aspect_ratio=anchor_generator.ratios,
                                                                octave_base_scale=anchor_generator.octave_base_scale,
                                                                scales_per_octave=anchor_generator.scales_per_octave)

    nms_param = mmdet_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=100)
    detection_output_param = mmdet_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmdet_meta_arch_pb2.CENTER_SIZE, keep_top_k=100,
                                            confidence_threshold=0.5)

    retinanet = mmdet_meta_arch_pb2.TidlMaRetinaNet(box_input=reg_output_names, class_input=cls_output_names, output='output',
                                              x_scale=1.0, y_scale=1.0, width_scale=1.0, height_scale=1.0,
                                              in_width=input_size[3], in_height=input_size[2],
                                              score_converter=score_converter, anchor_param=anchor_param,
                                              detection_output_param=detection_output_param)

    arch = mmdet_meta_arch_pb2.TIDLMetaArch(name='retinanet',  tidl_retinanet=[retinanet])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmdet_proto_yolov3(cfg, model, input_size, output_filename, input_names=None, output_names=None):
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    #num_output_names = len(output_names)
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.anchor_generator
    base_sizes = anchor_generator.base_sizes

    background_label_id = -1
    num_classes = bbox_head.num_classes
    #score_converter = mmdet_meta_arch_pb2.SIGMOID

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = mmdet_meta_arch_pb2.TIDLYoloParams(input=output_names[base_size_id],
                                                        anchor_width=[b[0] for b in base_size],
                                                        anchor_height=[b[1] for b in base_size])
        yolo_params.append(yolo_param)

    nms_param = mmdet_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=100)
    detection_output_param = mmdet_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmdet_meta_arch_pb2.CENTER_SIZE_EXP, keep_top_k=100,
                                            confidence_threshold=0.5)

    yolov3 = mmdet_meta_arch_pb2.TidlYoloOd(name='yolo_v3',
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param)

    arch = mmdet_meta_arch_pb2.TIDLMetaArch(name='yolo_v3',  tidl_yolo=[yolov3])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)