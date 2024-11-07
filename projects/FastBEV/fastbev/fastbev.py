# -*- coding: utf-8 -*-
from typing import Dict, List, Union, Optional

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from mmdet.models import BaseDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.ops import bbox3d2result

from mmseg.models import build_head as build_seg_head
from mmseg.models.utils import resize
from mmengine.structures import InstanceData
from mmdet3d.structures.det3d_data_sample import SampleList, ForwardResults, OptSampleList
from mmdet3d.utils.typing_utils import OptInstanceList
#from mmcv.runner import get_dist_info, auto_fp16

import copy

from mmdet3d.models.detectors.onnx_export import export_FastBEV
from mmdet3d.models.detectors.onnx_network import backproject_tidl

@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points

def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    #          Feature filling: only valid features are filled,
    #          and repeated features are directly overwritten
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    #volume = volume.view(1, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume


@MODELS.register_module()
class FastBEV(BaseDetector):
    def __init__(
        self,
        data_preprocessor,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        bbox_head,
        seg_head,
        n_voxels,
        voxel_size,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        seq_detach=False,
        multi_scale_id=None,
        multi_scale_3d_scaler=None,
        with_cp=False,
        backproject='inplace',
        style='v4',
        save_onnx_model=False
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.neck_3d = MODELS.build(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        
        # style
        # v1: fastbev wo/ ms
        # v2: fastbev + img ms
        # v3: fastbev + bev ms
        # v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style
        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = MODELS.build(bbox_head)
            self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = MODELS.build(bbox_head_2d)
        else:
            self.bbox_head_2d = None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # detach adj feature
        self.seq_detach = seq_detach
        self.backproject = backproject
        # checkpoint
        self.with_cp = with_cp

        # for onnx model export
        self.save_onnx_model = save_onnx_model
        self.model_exported = False


    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            if isinstance(data_samples[0], list):
                # aug test
                assert len(data_samples[0]) == 1, 'Only support ' \
                                                  'batch_size 1 ' \
                                                  'in mmdet3d when ' \
                                                  'do the test' \
                                                  'time augmentation.'
                return self.aug_test(inputs, data_samples, **kwargs)
            else:
                if self.save_onnx_model is True and self.model_exported is False:
                    export_FastBEV(self, inputs, data_samples, **kwargs)
                    # Export onnx only once
                    self.model_exported = True

                return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')


    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def extract_feat(self, img, img_metas, mode, xy_coors=None):
        batch_size = img.shape[0]
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        x = self.backbone(
            img
        )  # [6, 256, 232, 400]; [6, 512, 116, 200]; [6, 1024, 58, 100]; [6, 2048, 29, 50]

        # use for vovnet
        if isinstance(x, dict):
            tmp = []
            for k in x.keys():
                tmp.append(x[k])
            x = tmp

        # fuse features
        def _inner_forward(x):
            out = self.neck(x)
            return out  # [6, 64, 232, 400]; [6, 64, 116, 200]; [6, 64, 58, 100]; [6, 64, 29, 50])

        if self.with_cp and x.requires_grad:
            mlvl_feats = cp.checkpoint(_inner_forward, x)
        else:
            mlvl_feats = _inner_forward(x)
        mlvl_feats = list(mlvl_feats)

        features_2d = None
        if self.bbox_head_2d:
            features_2d = mlvl_feats

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_
        # v3 bev ms
        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            if batch_size == 1:
                mlvl_feat_split = torch.split(mlvl_feat, 6, dim=0)
            else:
                # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
                mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
                # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
                mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id, mlvl_feat_i in enumerate(mlvl_feat_split):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    if batch_size == 1:
                        feat_i = mlvl_feat_i
                    else:
                        feat_i = mlvl_feat_i[batch_id]  # [nv, c, h, w]

                    #if vol_coors is None:
                    if xy_coors is None:
                        img_meta = copy.deepcopy(seq_img_meta)
                        img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*6:(seq_id+1)*6]
                        if isinstance(img_meta["img_shape"], list):
                            img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                            img_meta["img_shape"] = img_meta["img_shape"][0]
                        height = math.ceil(img_meta["img_shape"][0] / stride_i)
                        width = math.ceil(img_meta["img_shape"][1] / stride_i)

                        projection = self._compute_projection(
                            img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                        if self.style in ['v1', 'v2']:
                            # wo/ bev ms
                            n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                        else:
                            # v3/v4 bev ms
                            n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                        points = get_points(  # [3, vx, vy, vz]
                            n_voxels=torch.tensor(n_voxels),
                            voxel_size=torch.tensor(voxel_size),
                            origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                        ).to(feat_i.device)

                        if self.backproject == 'inplace':
                            volume = backproject_inplace(
                                feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
                    else:
                        if self.style in ['v1', 'v2']:
                            volume = backproject_tidl(feat_i, xy_coors[seq_id], self.n_voxels[0])
                        else:
                            volume = None
                            raise RuntimeError('TIDL implementation is NOT available ofr v3 and v4')

                    # to reduce redundant operator
                    if batch_size == 1:
                        volume = volume.view([1, feat_i.shape[1]] + self.n_voxels[0])
                    else:
                        volume = volume.view([feat_i.shape[1]] + self.n_voxels[0])
                        volumes.append(volume)

                if batch_size ==1:
                    volume_list.append(volume)
                else:
                    volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])
        
        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
        else:
            # bev ms: multi-scale bev map (different x/y/z)
            for i, mlvl_volume in enumerate(mlvl_volumes):
                bs, c, x, y, z = mlvl_volume.shape
                # collapse h, [bs, seq*c, vx, vy, vz] -> [bs, seq*c*vz, vx, vy]
                mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z*c).permute(0, 3, 1, 2)
                
                # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
                if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                    # pooling to bottom level
                    mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
                elif self.multi_scale_3d_scaler == 'upsample' and i != 0:  
                    # upsampling to top level 
                    mlvl_volume = resize(
                        mlvl_volume,
                        mlvl_volumes[0].size()[2:4],
                        mode='bilinear',
                        align_corners=False)
                else:
                    # same x/y
                    pass

                # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
                mlvl_volume = mlvl_volume.unsqueeze(-1)
                mlvl_volumes[i] = mlvl_volume
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]

        x = mlvl_volumes
        def _inner_forward(x):
            # v1/v2: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy]
            # v3/v4: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
            out = self.neck_3d(x)
            return out
            
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x, None, features_2d

    #@auto_fp16(apply_to=('img', ))
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `imgs` keys.

                - imgs (torch.Tensor): Tensor of batched multi-view images.
                    It has shape (B, N, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 9).
        """
        img = batch_inputs_dict['imgs']
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        xy_coors=None
        if self.save_onnx_model is True:
            xy_coors = self.precompute_proj_info(img, batch_input_metas)
        bbox_pts = self.simple_test(img, batch_input_metas, xy_coors=xy_coors)

        ret_list = []
        for _, preds in enumerate(bbox_pts):
            results = InstanceData()
            #preds = bbox_pts[i]
            results.bboxes_3d = preds['bboxes_3d']
            results.scores_3d = preds['scores_3d']
            results.labels_3d = preds['labels_3d']
            # change box dim and yaw
            #    nus_box_dims = box_dims[:, [0, 1, 2]]
            #    box_yaw = -box_yaw - np.pi/2
            # It coulde be removed with a trained model using new picle file
            results.bboxes_3d.tensor = results.bboxes_3d.tensor[:, [0, 1, 2, 4, 3, 5, 6, 7, 8]]
            results.bboxes_3d.tensor[:, 6] = -results.bboxes_3d.tensor[:, 6] - np.pi/2
            ret_list.append(results)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 ret_list)
        return detsamples



    #def forward_train(
    #    self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    #):
    def loss(
        self, input=None, data_samples=None, gt_bboxes_3d=None, gt_labels_3d=None, gt_bev_seg=None, **kwargs
    ):
        feature_bev, valids, features_2d = self.extract_feat(img, img_metas, "train")
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)

        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            gt_bev = gt_bev_seg[0][None, ...].long()
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        if self.bbox_head_2d is not None:
            gt_bboxes = kwargs["gt_bboxes"][0]
            gt_labels = kwargs["gt_labels"][0]
            assert len(kwargs["gt_bboxes"]) == 1 and len(kwargs["gt_labels"]) == 1
            # hack a img_metas_2d
            img_metas_2d = []
            img_info = img_metas[0]["img_info"]
            for idx, info in enumerate(img_info):
                tmp_dict = dict(
                    filename=info["filename"],
                    ori_filename=info["filename"].split("/")[-1],
                    ori_shape=img_metas[0]["ori_shape"],
                    img_shape=img_metas[0]["img_shape"],
                    pad_shape=img_metas[0]["pad_shape"],
                    scale_factor=img_metas[0]["scale_factor"],
                    flip=False,
                    flip_direction=None,
                )
                img_metas_2d.append(tmp_dict)

            #rank, world_size = get_dist_info()
            loss_2d = self.bbox_head_2d.forward_train(
                features_2d, img_metas_2d, gt_bboxes, gt_labels
            )
            losses.update(loss_2d)

        return losses


    def _forward(self, inputs=None, data_samples=None, **kwargs):
        raise NotImplementedError('tensor mode is yet to add')


    def precompute_volume_info(self, points, projection):
        """
        function: 2d feature + predefined point cloud -> 3d volume
        """
        feat_shape = [6, 64, 64, 176]
        n_images, n_channels, height, width = feat_shape
        n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
        # [3, 200, 200, 4] -> [1, 3, 160000] -> [6, 3, 160000]
        points = points.view(1, 3, -1).expand(n_images, 3, -1)
        # [6, 3, 160000] -> [6, 4, 160000]
        points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    
        # ego_to_cam
        points_2d_3 = torch.bmm(projection, points)  # lidar2img
        x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 160000]
        y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 160000]
        z = points_2d_3[:, 2]  # [6, 160000]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 160000]

        # xy coordinate
        xy_coor = y * width + x
        
        coor      = torch.full((1, xy_coor.shape[1]), width*height*n_images).to(points.device)
        cum_valid = torch.full((1, xy_coor.shape[1]), True).to(points.device)
        cum_valid = cum_valid[0]

        for i in reversed(range(n_images)):
            valid_idx = torch.mul(cum_valid, valid[i]).to(torch.bool)
            coor[0, valid_idx] = xy_coor[i, valid_idx] + i*width*height
            cum_valid[valid_idx] = False

        return coor[0]


    def precompute_proj_info(self, img, img_metas):
        xy_coor_list   = []

        n_times = self.neck_3d.fuse.in_channels // 256
        stride_i = 4
        for seq_id in range(n_times):
            for batch_id, seq_img_meta in enumerate(img_metas):
                img_meta = copy.deepcopy(seq_img_meta)
                img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*6:(seq_id+1)*6]
                if isinstance(img_meta["img_shape"], list):
                    img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                    img_meta["img_shape"] = img_meta["img_shape"][0]

                projection = self._compute_projection(
                    img_meta, stride_i, noise=self.extrinsic_noise).to(img.device)

                if self.style in ['v1', 'v2']:
                    # wo/ bev ms
                    n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                else:
                    # v3/v4 bev ms
                    n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]

                points = get_points(  # [3, vx, vy, vz]
                    n_voxels=torch.tensor(n_voxels),
                    voxel_size=torch.tensor(voxel_size),
                    origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                ).to(img.device)

                xy_coor = self.precompute_volume_info(points, projection)
                xy_coor_list.append(xy_coor)

        return torch.stack(xy_coor_list)

    """
    def forward_test(self, img, img_metas, **kwargs):
        if self.save_onnx_model:
            export_FastBEV(self, img, img_metas)
            self.save_onnx_model = False

        if not self.test_cfg.get('use_tta', False):
            xy_coors = self.precompute_proj_info(img, img_metas)
            return self.simple_test(img, img_metas, xy_coors)
        return self.aug_test(img, img_metas)
    """

    def simple_test(self, img, img_metas, xy_coors=None):
        bbox_results = []

        if xy_coors is None:
            feature_bev, _, features_2d = self.extract_feat(img, img_metas, "test")
        else:
            feature_bev, _, features_2d = self.extract_feat(img, img_metas, "test",
                                                            xy_coors)

        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]
        else:
            bbox_results = [dict()]

        # BEV semantic seg
        #if self.seg_head is not None:
        #    x_bev = self.seg_head(feature_bev)
        #    bbox_results[0]['bev_seg'] = x_bev

        return bbox_results

    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

    # aug_test() is not tested/verified
    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):

            img_metas[0]['img_shape'] = img_shape_copy[24*tta_id:24*(tta_id+1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24*tta_id:24*(tta_id+1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24*tta_id:24*(tta_id+1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results


    def show_results(self, *args, **kwargs):
        pass


