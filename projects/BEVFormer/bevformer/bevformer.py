# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

#from projects.mmdet3d_plugin.models.utils.bricks import run_time
from typing import Dict, List, Optional

import torch
from torch import Tensor
import time
import copy
import numpy as np

from mmdet3d.structures.ops import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData
from .grid_mask import GridMask


@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 data_preprocessor=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(img_backbone=img_backbone,
                             img_neck=img_neck,
                             pts_bbox_head=pts_bbox_head,  
                             train_cfg=train_cfg,
                             test_cfg=test_cfg,
                             data_preprocessor=data_preprocessor)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, img, batch_input_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0) 
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in batch_input_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped


    # This funtion should be renamed to extract_feat() once all codes are merged
    def extract_feat(self, batch_inputs_dict, batch_input_metas=None, len_queue=None):
        """Extract features from images and points."""
        imgs = batch_inputs_dict.get('imgs', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas, len_queue=len_queue)
        
        return img_feats


    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            #img_feats_list = self.extract_feat(batch_inputs_dict=imgs_queue, len_queue=len_queue)

            img_feats_list = self.extract_img_feat(imgs_queue, None, len_queue=len_queue)

            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev


    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        imgs = batch_inputs_dict.get('imgs', None)
        len_queue = imgs.size(1)
        prev_imgs = imgs[:, :-1, ...] # [B, Q-1, N, C, H, W]
        imgs = imgs[:, -1, ...]       # [B, N, C, H, W]

        batch_input_metas = [item.metainfo_queue for item in batch_data_samples]
        prev_input_metas = copy.deepcopy(batch_input_metas)
        prev_bev = self.obtain_history_bev(prev_imgs, prev_input_metas)

        batch_input_metas = [each[len_queue-1] for each in batch_input_metas]

        if not batch_input_metas[0]['prev_bev_exists']:
            prev_bev = None

        img_feats = self.extract_img_feat(imgs, batch_input_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, batch_input_metas, prev_bev=prev_bev)

        losses.update(losses_pts)
        return losses


    def forward_pts_train(self, x, batch_input_metas, prev_bev=None, rescale=False):
        outs = self.pts_bbox_head(x, batch_input_metas, prev_bev=prev_bev)

        batch_gt_bboxes_3d = []
        batch_gt_labels_3d = []
        for input_metas in batch_input_metas:
            batch_gt_bboxes_3d.append(input_metas['gt_bboxes_3d'])
            batch_gt_labels_3d.append(input_metas['gt_labels_3d'])

        loss_inputs = [batch_gt_bboxes_3d, batch_gt_labels_3d, outs]
        losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=batch_input_metas)

        return losses_pts


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
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        # 'lidar2img' already in metainfo. Do don't need to call
        #batch_input_metas = self.add_lidar2img(batch_input_metas)

        if batch_input_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = batch_input_metas[0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(batch_input_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(batch_input_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            batch_input_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            batch_input_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            batch_input_metas[0]['can_bus'][-1] = 0
            batch_input_metas[0]['can_bus'][:3] = 0


        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        #bbox_list = [dict() for i in range(len(batch_input_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, batch_input_metas, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)

        #for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #    result_dict['pts_bbox'] = pts_bbox

        ret_list = []
        for i in range(len(bbox_pts)):
            results = InstanceData()
            preds = bbox_pts[i]
            results.bboxes_3d = preds['bboxes_3d']
            results.scores_3d = preds['scores_3d']
            results.labels_3d = preds['labels_3d']
            ret_list.append(results)

        #print("==========================")
        #print(bbox_pts[0]['scores_3d'][0:10])
        #print(bbox_pts[0]['bboxes_3d'][0])
        #print(bbox_pts[0]['labels_3d'][0:10])
        #print("==========================")


        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 ret_list)

        return detsamples


    def simple_test_pts(self, x, batch_input_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, batch_input_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, batch_input_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

