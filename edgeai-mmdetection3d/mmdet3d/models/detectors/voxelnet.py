# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from .. import builder
from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector
import os
import numpy as np


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.max_num_voxel = 0
        self.max_num_points_per_voxel = 0

    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)

        count_max_voxel_cnt = False
        if count_max_voxel_cnt == True:
            if(self.max_num_voxel < voxels.shape[0]):
                self.max_num_voxel = voxels.shape[0]
                print('maximum number of voxel is', self.max_num_voxel)

            if(self.max_num_points_per_voxel < num_points.max()):
                self.max_num_points_per_voxel = num_points.max()
                print('maximum number of point in voxel is', self.max_num_points_per_voxel)

        if (img_metas is not None) and os.path.split(img_metas[0]['pts_filename'])[1] == 'x01918.bin':
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

        if dump_voxel == True:
            dumpTensor('voxel_data.txt',voxels,32.0)

        voxel_features = self.voxel_encoder(voxels, num_points, coors, dump_raw_voxel_feat)

        # avoid using scatter operator when multiple frame inputs are used i.e. batch_size >1
        batch_size = coors[-1, 0].item() + 1

        if dump_voxel_feature == True:
            dumpTensor('voxel_features.txt',voxel_features)

        scatter_op_flow = False

        if self.middle_encoder.use_scatter_op == True and batch_size == 1:
            # use the scatter operator only in this flow. This happens at inference time when batch size is 1
            # when batch size is not 1, then multiple frame data is combined in singel tensor hence difficult to use scatter operator
            coors = coors[:, 2] * self.middle_encoder.nx + coors[:, 3]
            coors = coors.long()
            coors = coors.repeat(self.middle_encoder.in_channels,1)
            scatter_op_flow = True

        if ((scatter_op_flow == False) and (self.voxel_encoder.pfn_layers.replace_mat_mul == True)) or ((scatter_op_flow == True) and (self.voxel_encoder.pfn_layers.replace_mat_mul == False)):
            # when replace_mat_mul is True then voxel_features is in form of 64(C)xP, this flow is introduced by TI
            # when replace_mat_mul is False then voxel_features is in form of PxC, this flow is original default flow
            # when use_scatter_op is enabled then voxel feature is needed in CxP form. This flow is introduced by TI
            # when use_scatter_op is disabled then voxel feature is needed in PxC form. This flow is original default flow
            # when complete flow is of TI or original mmdetd3d one then this transofrmation is not needed, otherwise it is needed to make data compatible
            voxel_features = voxel_features.t()

        x = self.middle_encoder(voxel_features, coors, batch_size)



        if dump_middle_encoder == True:
            dumpTensor('middle_encoder.txt',x[0])

        x = self.backbone(x)

        if dump_backbone == True:
            dumpTensor("backbone_0.txt",x[0][0])
            dumpTensor("backbone_1.txt",x[1][0])
            dumpTensor("backbone_2.txt",x[2][0])

        if self.with_neck:
            x = self.neck(x)

        if dump_neck == True:
            dumpTensor("neck.txt",x[0][0])

        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""

        save_raw_point_data = False

        import numpy as np
        import os

        if save_raw_point_data == True:
            np_arr = points[0].cpu().detach().numpy()
            base_file_name = os.path.split(img_metas[0]['pts_filename'])
            dst_dir = '/data/adas_vision_data1/datasets/other/vision/common/kitti/training/velodyne_reduced_custom_point_pillars'
            np_arr.tofile(os.path.join(dst_dir,base_file_name[1]))

        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)

        if os.path.split(img_metas[0]['pts_filename'])[1] == '00000x.bin':
            dump_bbox_head = True
        else:
            dump_bbox_head = False

        if dump_bbox_head == True:
            dumpTensor('conf.txt',outs[0][0][0])
            dumpTensor('loc.txt',outs[1][0][0])
            dumpTensor('dir.txt',outs[2][0][0])

        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

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
