import os
import torch
from .quantize import is_mmdet_quant_module

__all__ = ['save_model_proto']


def save_model_proto(cfg, model, input_size, output_dir):
    model = model.module if is_mmdet_quant_module(model) else model
    is_ssd = hasattr(cfg.model, 'bbox_head') and ('SSDHead' in cfg.model.bbox_head.type)
    is_fcos = hasattr(cfg.model, 'bbox_head') and ('FCOSHead' in cfg.model.bbox_head.type)
    is_retinanet = hasattr(cfg.model, 'bbox_head') and ('RetinaHead' in cfg.model.bbox_head.type)
    if is_ssd:
        input_names = ['input']
        output_names = []
        for cls_idx, cls in enumerate(model.bbox_head.cls_convs):
            output_names.append(f'cls_convs_{cls_idx}')
        #
        for reg_idx, reg in enumerate(model.bbox_head.reg_convs):
            output_names.append(f'reg_convs_{reg_idx}')
        #
        _save_mmdet_onnx(cfg, model, input_size, output_dir, input_names, output_names)
        _save_mmdet_proto(cfg, model, input_size, output_dir, input_names, output_names)
    elif is_fcos:
        input_names = ['input']
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
        _save_mmdet_onnx(cfg, model, input_size, output_dir, input_names, output_names)
    elif is_retinanet:
        input_names = ['input']
        output_names = []
        for i in range(model.neck.num_outs):
            output_names.append(f'retina_cls_{i}')
        #
        for i in range(model.neck.num_outs):
            output_names.append(f'retina_reg_{i}')
        #
        _save_mmdet_onnx(cfg, model, input_size, output_dir, input_names, output_names)
    else:
        _save_mmdet_onnx(cfg, model, input_size, output_dir)
    #


###########################################################
def _create_rand_inputs(input_size, is_cuda=False):
    x = torch.rand(input_size)
    x = x.cuda() if is_cuda else x
    return x


def _save_mmdet_onnx(cfg, model, input_size, output_dir, input_names=None, output_names=None, name='model.onnx'):
    #https://github.com/open-mmlab/mmdetection/pull/1082
    assert hasattr(model, 'forward_dummy'), 'wrting onnx is not supported by this model'
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    forward_backup = model.forward #backup forward
    model.forward = model.forward_dummy #set dummy forward
    is_cuda = next(model.parameters()).is_cuda
    input_list = _create_rand_inputs(input_size, is_cuda)
    opset_version = 9
    torch.onnx.export(model, input_list, os.path.join(output_dir, name), input_names=input_names,
                      output_names=output_names, export_params=True, verbose=False, opset_version=opset_version)
    model.forward = forward_backup #restore forward


from .proto import mmdet_meta_arch_pb2
from google.protobuf import text_format
def _save_mmdet_proto(cfg, model, input_size, output_dir, input_names=None, output_names=None, name='model.prototxt'):
    num_output_names = len(output_names)//2
    cls_output_names = output_names[:num_output_names]
    reg_output_names = output_names[num_output_names:]
    heads = model.bbox_head
    anchor_generator = model.bbox_head.anchor_generator
    prior_box_param = []
    for h_idx in range(num_output_names):
        min_size=[anchor_generator.min_sizes[h_idx]]
        max_size=[anchor_generator.max_sizes[h_idx]]
        aspect_ratio=anchor_generator.ratios[h_idx][2::2]
        step=anchor_generator.strides[h_idx]
        step=step[0] if isinstance(step,(tuple,list)) else step
        prior_box_param.append(mmdet_meta_arch_pb2.PriorBoxParameter(min_size=min_size, max_size=max_size,
                                                                     aspect_ratio=aspect_ratio, step=step,
                                                                     variance=heads.bbox_coder.stds, clip=False, flip=True))
    #
    nms_param = mmdet_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=100)
    detection_output_param = mmdet_meta_arch_pb2.TIDLOdPostProc(num_classes=heads.num_classes, share_location=True,
                                            background_label_id=0, nms_param=nms_param, code_type=mmdet_meta_arch_pb2.CENTER_SIZE, keep_top_k=100,
                                            confidence_threshold=0.5)
    ssd = mmdet_meta_arch_pb2.TidlMaCaffeSsd(box_input=reg_output_names, class_input=cls_output_names, output='output', prior_box_param=prior_box_param,
                                             in_width=input_size[3], in_height=input_size[2], detection_output_param=detection_output_param)
    arch = mmdet_meta_arch_pb2.TIDLMetaArch(name='ssd',  caffe_ssd=[ssd])
    with open(os.path.join(output_dir, name), 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)