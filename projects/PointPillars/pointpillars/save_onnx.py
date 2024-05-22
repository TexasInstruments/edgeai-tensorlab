from typing import Dict, List, Optional

import os
import torch
import torch.nn as nn
import numpy as np
import copy

from google.protobuf import text_format
from mmdet3d.utils.proto import tidl_meta_arch_pb2



class CombinedModel(torch.nn.Module):

    def __init__(self, pfn_layers, middle_encoder, backbone, neck, conv_cls, conv_dir_cls,conv_reg, num_anchors):
        super().__init__()
        self.pfn_layers     = pfn_layers
        self.middle_encoder = middle_encoder
        self.backbone       = backbone
        self.neck           = neck
        self.conv_cls       = conv_cls
        self.conv_dir_cls   = conv_dir_cls
        self.conv_reg       = conv_reg
        self.num_ancohors   = num_anchors  # for each classs two anchors at 90 degree of placement is used

    def forward(self, raw_voxel_feat, coors, data):

        x = self.pfn_layers(raw_voxel_feat,ip_tensor_dim_correct=True)
        x = self.middle_encoder(x,coors,1,data)
        x = self.backbone(x)
        x = self.neck(x)
        y0= self.conv_cls(x[0])
        y1= self.conv_dir_cls(x[0])
        y2= self.conv_reg(x[0])
        y = torch.cat((y0, y1, y2), 1)
        y = torch.reshape(y,(-1,(int)((y0.shape[-3]+y1.shape[-3]+y2.shape[-3])/self.num_ancohors)))
        return y


def save_onnx_model(cfg, model, path, quantized_model=True,tag=''):

        print('onnx export started, at path {}'.format(path))

        model_cpu = model.to('cpu')

        #device_ids=[torch.cuda.current_device()]
        
        #if len(device_ids) > 1:
        #    print('Multiple GPUs visible, Currently only one GPU can be used for onnx export')
            
        #from . import CombinedModel

        model_cpu.eval()

        bev_size_x = (int)((cfg.point_cloud_range[3] - cfg.point_cloud_range[0])/cfg.voxel_size[0])
        bev_size_y = (int)((cfg.point_cloud_range[4] - cfg.point_cloud_range[1])/cfg.voxel_size[1])

        # Accessing one layer to find wheather wegihts are on cpu or cuda
        #if quantized_model == True:
        #    device = model_cpu.module.voxel_encoder.pfn_layers._modules['0'].linear.weight.device
        #else:
        #    device = model_cpu.voxel_encoder.pfn_layers._modules['0'].linear.weight.device

        device ='cpu'

        if hasattr(cfg.model.voxel_encoder, 'point_color_dim'):
            point_color_dim = cfg.model.voxel_encoder.point_color_dim
        else:
            point_color_dim = 0
        raw_voxel_feat = torch.ones([1, cfg.model.voxel_encoder.in_channels+point_color_dim+6, 
                                        cfg.model.data_preprocessor.voxel_layer.max_num_points, 
                                        cfg.model.data_preprocessor.voxel_layer.max_voxels[2]],
                                        device=torch.device(device))

        data = torch.zeros([1, cfg.model.voxel_encoder.feat_channels[0], bev_size_x*bev_size_y],device=torch.device(device))
        coors = torch.ones([1, cfg.model.voxel_encoder.feat_channels[0], cfg.model.data_preprocessor.voxel_layer.max_voxels[2]],device=torch.device(device))
        coors = coors.long()

        if quantized_model == True:
            combined_model = CombinedModel(model_cpu.module.voxel_encoder.pfn_layers._modules['0'],
                                            model_cpu.module.middle_encoder,
                                            model_cpu.module.backbone,
                                            model_cpu.module.neck,
                                            model_cpu.module.bbox_head.conv_cls,
                                            model_cpu.module.bbox_head.conv_dir_cls,
                                            model_cpu.module.bbox_head.conv_reg,
                                            model_cpu.module.bbox_head.num_anchors
                                            )
        else:
            combined_model = CombinedModel(model_cpu.voxel_encoder.pfn_layers._modules['0'],
                                            model_cpu.middle_encoder,
                                            model_cpu.backbone,
                                            model_cpu.neck,
                                            model_cpu.bbox_head.conv_cls,
                                            model_cpu.bbox_head.conv_dir_cls,
                                            model_cpu.bbox_head.conv_reg,
                                            model_cpu.bbox_head.num_anchors
                                            )

        torch.onnx.export(combined_model,
                (raw_voxel_feat,coors,data),
                os.path.join(path,tag+"combined_model.onnx"),
                opset_version=11,
                verbose=False)
                
        # change the data type from int64 to int32 for co-ordinates in scatter layer, as TIDL doesnt supprt that
        # it can not be changes in pytorch operator as int32 is not supported there.
        import onnx
        onnx_model = onnx.load(os.path.join(path,tag+"combined_model.onnx"))
        graph = onnx_model.graph

        for inp in graph.input:
            if inp.name == 'coors' and inp.type.tensor_type.elem_type == onnx.TensorProto.INT64 :
                inp.type.tensor_type.elem_type = onnx.TensorProto.INT32

        onnx.save(onnx_model, os.path.join(path,tag+"combined_model_int32.onnx"))
        save_tidl_prototxt(cfg, path, tag)


def save_tidl_prototxt(cfg, path,tag=''):

    import onnx
    prototxt_file = os.path.join(path,tag+"pointPillars.prototxt")
    model = onnx.load(os.path.join(path,tag+"combined_model_int32.onnx"))
    graph = model.graph
    
    box_input = []
    dir_input = []
    cls_input = []

    for node in graph.node:
        for inp in node.input:
            if 'conv_cls' in inp and (len(cls_input) == 0):
                cls_input.append(node.output)
            if 'conv_dir' in inp and (len(dir_input) == 0):
                dir_input.append(node.output)
            if 'conv_reg' in inp and (len(box_input) == 0):
                box_input.append(node.output)

    if cfg.quantize == True:
        for node in graph.node:
            if node.op_type == 'Clip':
                for inp in node.input:
                    if cls_input[0][0] == inp and (len(cls_input) <= 1):
                        cls_input = []
                        cls_input.append(node.output)
                    if dir_input[0][0] == inp and (len(dir_input) <= 1):
                        dir_input = []
                        dir_input.append(node.output)
                    if box_input[0][0] == inp and (len(box_input) <= 1):
                        box_input = []
                        box_input.append(node.output)


    bev_size_x = (int)((cfg.point_cloud_range[3] - cfg.point_cloud_range[0])/cfg.voxel_size[0])
    bev_size_y = (int)((cfg.point_cloud_range[4] - cfg.point_cloud_range[1])/cfg.voxel_size[1])

    offset_dir= -cfg.model.bbox_head.anchor_generator.rotations[1]    
    prior_box_param = []
    
    for id in range(cfg.model.bbox_head.num_classes):
        step_x = (cfg.model.bbox_head.anchor_generator.ranges[id][3] - cfg.model.bbox_head.anchor_generator.ranges[id][0])/(bev_size_x/2) 
        step_y = (cfg.model.bbox_head.anchor_generator.ranges[id][4] - cfg.model.bbox_head.anchor_generator.ranges[id][1])/(bev_size_y/2)   
        step_z = 1

        offset_x = (cfg.model.bbox_head.anchor_generator.ranges[id][0])/step_x + 0.5 # 0.5 = step_x/2 * step_x
        offset_y = (cfg.model.bbox_head.anchor_generator.ranges[id][1])/step_y + 0.5 # 0.5 = step_y/2 * step_y
        offset_z = (cfg.model.bbox_head.anchor_generator.ranges[id][2])/step_z + 0.0 # 0.0 = step_z/2 * step_z

        prior_box_param.append(tidl_meta_arch_pb2.PriorBox3DODParameter(anchor_width=[cfg.model.bbox_head.anchor_generator.sizes[id][0]], 
                                                                        anchor_height=[cfg.model.bbox_head.anchor_generator.sizes[id][1]], 
                                                                        anchor_length=[cfg.model.bbox_head.anchor_generator.sizes[id][2]],
                                                                        step_x=step_x, step_y=step_y, step_z=step_z,
                                                                        offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
                                                                        offset_dir = offset_dir,
                                                                        rotation=cfg.model.bbox_head.anchor_generator.rotations))

    nms_param = tidl_meta_arch_pb2.TIDLNmsParam(nms_threshold=cfg.model.test_cfg.nms_thr,top_k=cfg.model.test_cfg.nms_pre)

    detection_param   = tidl_meta_arch_pb2.TIDLOdPostProc(num_classes=cfg.model.bbox_head.num_classes,
                                                          share_location=True,
                                                          background_label_id=-1,
                                                          code_type="CODE_TYPE_3DOD",
                                                          keep_top_k=cfg.model.test_cfg.max_num,
                                                          confidence_threshold=cfg.model.test_cfg.score_thr,
                                                          nms_param=nms_param
                                                        )                                            

    od3d = tidl_meta_arch_pb2.TidlMa3DOD(name="point_pillars", 
                                            min_x=cfg.point_cloud_range[0],
                                            max_x=cfg.point_cloud_range[3],
                                            min_y=cfg.point_cloud_range[1],
                                            max_y=cfg.point_cloud_range[4],
                                            min_z=cfg.point_cloud_range[2], 
                                            max_z=cfg.point_cloud_range[5],
                                            voxel_size_x=cfg.voxel_size[0],voxel_size_y=cfg.voxel_size[1],voxel_size_z=cfg.voxel_size[2],
                                            max_points_per_voxel=cfg.model.data_preprocessor.voxel_layer.max_num_points,
                                            box_input=box_input[0],class_input=cls_input[0],dir_input=dir_input[0],
                                            output=[node.name for node in model.graph.output],
                                            prior_box_3dod_param=prior_box_param,
                                            detection_output_param=detection_param)

    arch = tidl_meta_arch_pb2.TIDLMetaArch(name='3dod_ssd',  tidl_3dod=[od3d])

    with open(prototxt_file, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)\

