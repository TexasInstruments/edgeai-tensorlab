# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList

import os
import numpy as np

# Point Pillars VoxelNet
@MODELS.register_module()
class PPVoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.max_num_voxel = 0
        self.max_num_points_per_voxel = 0


    def extract_feat(self, batch_inputs_dict: dict, batch_input_metas: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']

        count_max_voxel_cnt = False
        if count_max_voxel_cnt is True:
            if(self.max_num_voxel < voxel_dict['voxels'].shape[0]):
                self.max_num_voxel = voxel_dict['voxels'].shape[0]
                print('maximum number of voxel is', self.max_num_voxel)

            if(self.max_num_points_per_voxel < voxel_dict['num_points'].max()):
                self.max_num_points_per_voxel = voxel_dict['num_points'].max()
                print('maximum number of point in voxel is', self.max_num_points_per_voxel)

        # What is x01918.bin?
        if (batch_input_metas is not None) and os.path.split(batch_input_metas[0]['lidar_path'])[1] == 'x01918.bin':
            dump_voxel          = True
            dump_voxel_feature  = True
            dump_middle_encoder = True
            dump_backbone       = True
            dump_neck           = True
            dump_raw_voxel_feat = False
        else:
            dump_voxel          = False
            dump_voxel_feature  = False
            dump_middle_encoder = False
            dump_backbone       = False
            dump_neck           = False
            dump_raw_voxel_feat = False

        if dump_voxel is True:
            dumpTensor('voxel_data.txt', voxel_dict['voxels'], 32.0)

        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'],
                                            dump_raw_voxel_feat)

        # avoid using scatter operator when multiple frame inputs are used i.e. batch_size >1
        batch_size = voxel_dict['coors'][-1, 0].item() + 1

        if dump_voxel_feature is True:
            dumpTensor('voxel_features.txt',voxel_features)

        scatter_op_flow = False

        if self.middle_encoder.use_scatter_op is True and batch_size == 1:
            # use the scatter operator only in this flow. This happens at inference time when batch size is 1
            # when batch size is not 1, then multiple frame data is combined in singel tensor hence difficult to use scatter operator
            voxel_dict['coors'] = voxel_dict['coors'][:, 2] * self.middle_encoder.nx + voxel_dict['coors'][:, 3]
            voxel_dict['coors'] = voxel_dict['coors'].long()
            voxel_dict['coors'] = voxel_dict['coors'].repeat(self.middle_encoder.in_channels,1)
            scatter_op_flow = True

        if ((scatter_op_flow is False) and (self.voxel_encoder.pfn_layers.replace_mat_mul is True)) or \
           ((scatter_op_flow is True) and (self.voxel_encoder.pfn_layers.replace_mat_mul is False)):
            # when replace_mat_mul is True then voxel_features is in form of 64(C)xP, this flow is introduced by TI
            # when replace_mat_mul is False then voxel_features is in form of PxC, this flow is original default flow
            # when use_scatter_op is enabled then voxel feature is needed in CxP form. This flow is introduced by TI
            # when use_scatter_op is disabled then voxel feature is needed in PxC form. This flow is original default flow
            # when complete flow is of TI or original mmdetd3d one then this transofrmation is not needed, otherwise it is needed to make data compatible
            voxel_features = voxel_features.t()

        x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)

        if dump_middle_encoder is True:
            dumpTensor('middle_encoder.txt',x[0])

        x = self.backbone(x)

        if dump_backbone is True:
            dumpTensor("backbone_0.txt",x[0][0])
            dumpTensor("backbone_1.txt",x[1][0])
            dumpTensor("backbone_2.txt",x[2][0])

        if self.with_neck:
            x = self.neck(x)

        if dump_neck is True:
            dumpTensor("neck.txt",x[0][0])

        return x

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        dump_txt_op = False
        read_txt_op = False

        batch_input_metas = [ds.metainfo for ds in batch_data_samples]

        save_raw_point_data = False

        if save_raw_point_data is True:
            np_arr = batch_inputs_dict['points'][0].cpu().detach().numpy()
            base_file_name = os.path.split(batch_input_metas[0]['lidar_path'])
            dst_dir = '/data/adas_vision_data1/datasets/other/vision/common/kitti/training/velodyne_reduced_custom_point_pillars'
            np_arr.tofile(os.path.join(dst_dir,base_file_name[1]))

        x = self.extract_feat(batch_inputs_dict, batch_input_metas)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)

        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)

        if dump_txt_op:
            file_name = os.path.split(batch_input_metas[0]['lidar_path'])[1]
            f = open(file_name+'.txt','w')
            for score, label, det_tensor in zip(results_list[0]['scores_3d'], results_list[0]['labels_3d'],results_list[0]['bboxes_3d'].tensor):
                f.write("{:3d} ".format(label))
                f.write("{:.4f} ".format(score))
                f.write("{:.4f} {:.4f} {:.4f} ".format(det_tensor[0],det_tensor[1],det_tensor[2]))
                f.write("{:.4f} {:.4f} {:.4f} ".format(det_tensor[3],det_tensor[4],det_tensor[5]))
                f.write("{:.4f}".format(det_tensor[6]))
                f.write("\n")
            f.close()

        if read_txt_op:
            file_name = os.path.split(batch_input_metas[0]['lidar_path'])[1]
            file_name = os.path.join('/user/a0393749/deepak_files/ti/c7x-mma-tidl-before/ti_dl/test/testvecs/output',file_name)
            f = open(file_name+'.txt','r')
            lines = f.readlines()
            det_tensor = torch.empty((len(lines),7), dtype=torch.float32, device = 'cpu')

            results_list[0]['scores_3d'] = torch.empty((len(lines)), dtype=torch.float32, device = 'cpu')
            results_list[0]['labels_3d'] = torch.empty((len(lines)), dtype=torch.float32, device = 'cpu')
            results_list[0]['bboxes_3d'] = batch_input_metas[0]['box_type_3d'](det_tensor, box_dim=7)

            for i, line in enumerate(lines):
                det = line.strip().split()
                results_list[0]['labels_3d'][i] = float(det[0])
                results_list[0]['scores_3d'][i] = float(det[1])
                for j in range(7):
                    det_tensor[i][j] = float(det[j+2])
            results_list[0]['bboxes_3d'] = batch_input_metas[0]['box_type_3d'](det_tensor, box_dim=7)

            f.close()

        return predictions

    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        if data_samples is not None:
            input_metas = [ds.metainfo for ds in data_samples]
        else:
            input_metas = None

        x = self.extract_feat(batch_inputs_dict, input_metas)
        results = self.bbox_head.forward(x)
        return results



    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_input_metas = [ds.metainfo for ds in batch_data_samples]

        x = self.extract_feat(batch_inputs_dict, batch_input_metas)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses


def dumpTensor(filleName, x, scale_fact=1.0):
    x_np = x.cpu().detach().numpy()

    if len(x_np.shape) == 2:
        x_np = np.expand_dims(x_np, axis=0)

    f = open(filleName,'w')
    for c, x_ch in  enumerate(x_np):
        for i, x_i in enumerate(x_ch):
            for j, pt in enumerate(x_i):
                f.write("{:.2f} ".format(pt*scale_fact))
            f.write("\n");
    f.close()
