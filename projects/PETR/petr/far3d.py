# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------

from typing import List, Union
import torch
from mmengine.structures import InstanceData
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures.det3d_data_sample import ForwardResults, OptSampleList
from .grid_mask import GridMask
from .utils import locations

import numpy as np

from mmdet3d.models.detectors.onnx_export import export_Far3D, export_Far3D_combined


@MODELS.register_module()
class Far3D(MVXTwoStageDetector):
    """Far3D."""

    def __init__(self,
                 use_grid_mask=False,
                 save_onnx_model=False,
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
                 init_cfg=None,
                 depth_branch=None,
                 stride=[16],
                 position_level=[0],
                 aux_2d_only=True,
                 single_test=False,
                 data_preprocessor=None):
        super(Far3D, self).__init__(pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, init_cfg, data_preprocessor)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only

        if depth_branch is not None:
            self.depth_branch =  MODELS.build(depth_branch)
        else:
            self.depth_branch = None

        # for onnx model export
        self.save_onnx_model = save_onnx_model

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
            # convert list of TxNxCxHxW to BatchxTxNxCxHxW
            # T is the number of temporal instances
            inputs['imgs'] = torch.stack(inputs['imgs'])
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
                if self.save_onnx_model is True:
                    export_Far3D_combined(self, inputs, data_samples, opset_version=18, **kwargs)
                    # Export onnx only once
                    self.save_onnx_model = False

                return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def predict(self, inputs=None, data_samples=None, mode=None, **kwargs):
        img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]
        for var, name in [(batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        results_list_3d = self.simple_test(batch_img_metas, img, **kwargs)

        for i, data_sample in enumerate(data_samples):
            results_list_3d_i = InstanceData(
                metainfo=results_list_3d[i]['pts_bbox'])
            data_sample.pred_instances_3d = results_list_3d_i
            data_sample.pred_instances = InstanceData()

        return data_samples


    def simple_test(self, batch_img_metas, img=None):
        """Test function without augmentaiton."""
        img_feats = self.extract_img_feat(img)

        bbox_list = [dict() for i in range(len(batch_img_metas))]
        bbox_pts, bbox_roi = self.simple_test_pts(img_feats, batch_img_metas)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def extract_img_feat(self, img, return_depth=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
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
        for i in self.position_level:
            BN, C, H, W = img_feats[i].size()
            img_feat_reshaped = img_feats[i].view(B, int(BN/B), C, H, W)
            img_feats_reshaped.append(img_feat_reshaped)
        
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(img_feats_reshaped)
        else:
            depths = None

        if return_depth:
            return img_feats_reshaped, depths
        return img_feats_reshaped


    def simple_test_pts(self, img_feats, img_metas):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_feats, img_metas)
        outs_roi = self.forward_roi_head(location, img_feats, img_metas)

        bbox_dict = self.img_roi_head.predict_by_feat(outs_roi)  # {'bbox_list': BN x (Mi, 4), 'bbox_score_list': BN x (Mi, 1)}
        bbox_roi = bbox_dict['bbox_list']
        outs_roi.update(bbox_dict)

        # Multiple batch?
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            img_metas[0]['prev_exists'] = img_feats[0].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            img_metas[0]['prev_exists'] = img_feats[0].new_ones(1)

        outs = self.pts_bbox_head(img_feats, img_metas, outs_roi)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results, bbox_roi


    def loss(self,
             inputs=None,
             data_samples=None,
             mode=None):
        """Forward training function.

        Args:
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        #for key in data:
        #    if key != 'gt_depth':
        #        data[key] = data[key][:, 0]

        rec_img = inputs['imgs']

        rec_img_feats, depths = self.extract_feat(rec_img, True)
        input['img_feats'] = rec_img_feats

        # TBA
        losses = [] #self.forward_pts_train(inputs, batch_img_metas)

        return losses


    def extract_feat(self, img, return_detph=False):
        """Extract features from images and points."""
        img_feats, depths = self.extract_img_feat(img, return_depth)
        if return_detph:
            return img_feats, depths
        return img_feats

    """
    def obtain_history_memory(self,
                              inputs,
                              batch_img_metas):
        losses = dict()
        T = inputs['imgs'].size(1)
        device = inputs['imgs'].device
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses

        # collect info for forward_pts_train()
        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_bboxes = []
        gt_labels = []
        centers_2d = []
        depths = []

        for _, img_metas in enumerate(batch_img_metas):
            # On same device. Why are they not on the same device as input img?
            for i in range(len(img_metas['gt_bboxes'])):
                img_metas['gt_bboxes'][i]        = img_metas['gt_bboxes'][i].to(device)
                img_metas['gt_bboxes_labels'][i] = img_metas['gt_bboxes_labels'][i].to(device)
                img_metas['centers_2d'][i]       = img_metas['centers_2d'][i].to(device)
                img_metas['depths'][i]           = img_metas['depths'][i].to(device)

        for i in range(T):
            gt_bboxes_3d_t = []
            gt_labels_3d_t = []
            gt_bboxes_t = []
            gt_labels_t = []
            centers_2d_t = []
            depths_t = []
            for _, img_metas in enumerate(batch_img_metas):
                gt_bboxes_3d_t.append(img_metas['gt_bboxes_3d'].to(device))
                gt_labels_3d_t.append(img_metas['gt_labels_3d'].to(device))
                gt_bboxes_t.append(img_metas['gt_bboxes'])
                gt_labels_t.append(img_metas['gt_bboxes_labels'])
                centers_2d_t.append(img_metas['centers_2d'])
                depths_t.append(img_metas['depths'])
        
        gt_bboxes_3d.append(gt_bboxes_3d_t)
        gt_labels_3d.append(gt_labels_3d_t)
        gt_bboxes.append(gt_bboxes_t)
        gt_labels.append(gt_labels_t)
        centers_2d.append(centers_2d_t)
        depths.append(depths_t)

        for i in range(T):
            requires_grad = False
            return_losses = False

            #data_t = dict()
            #for key in data:
            #    data_t[key] = data[key][:, i]
            #data_t['img_feats'] = data_t['img_feats']

            # Squeeze img_feats
            img_feats = inputs['img_feats'][:, i]

            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True

            loss = self.forward_pts_train(img_feats, batch_img_metas,
                                        gt_bboxes_3d[i], gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], centers_2d[i], depths[i],
                                        requires_grad=requires_grad, return_losses=return_losses)

            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value

        return losses
    """

    def prepare_location(self, img_feats, img_metas):
        # Multiple batch?
        pad_h, pad_w = img_metas[0]['pad_shape']
        assert len(self.stride) == len(img_feats)
        location_r = []
        for i in range(len(img_feats)):
            bs, n = img_feats[i].shape[:2]
            x = img_feats[i].flatten(0, 1)
            location = locations(x, self.stride[i], pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
            location_r.append(location)
        return location_r

    def forward_roi_head(self, location, img_feats, img_metas):

        intrinsics = []
        extrinsics = []
        for img_meta in img_metas:
            intrinsics.append(img_meta['intrinsics'])
            extrinsics.append(img_meta['extrinsics'])
        intrinsics = np.asarray(intrinsics)
        extrinsics = np.asarray(extrinsics)
        intrinsics = torch.from_numpy(intrinsics).to(img_feats[0].device)
        extrinsics = torch.from_numpy(extrinsics).to(img_feats[0].device)

        outs_roi = self.img_roi_head(img_feats, location, intrinsics, extrinsics)
        return outs_roi

    def forward_pts_train(self,
                          img_feats,
                          img_metas,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_feats, img_metas)

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_feats, img_metas, None)
            self.train()
        else:
            outs_roi = self.forward_roi_head(location, img_feats)
            topk_indexes = outs_roi['topk_indexes']
            outs = self.pts_bbox_head(location, img_feats, img_metas, topk_indexes)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss_by_feat(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss_by_feat(*loss2d_inputs)
                losses.update(losses2d)

            return losses
        else:
            return None


    def _forward(self, inputs=None, data_samples=None, mode=None, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        raise NotImplementedError('tensor mode is yet to add')