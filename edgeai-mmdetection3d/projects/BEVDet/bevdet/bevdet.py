# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
#from mmcv.runner import force_fp32


from .ops.bev_pool_v2.bev_pool import TRTBEVPoolv2

from mmengine.structures import InstanceData
from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet

from mmdet3d.structures.ops import bbox3d2result
from mmdet3d.registry import MODELS
import numpy as np

@MODELS.register_module()
class BEVDet(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self,
                 img_view_transformer,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 use_grid_mask=False,
                 **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.grid_mask = None if not use_grid_mask else \
            GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
                     prob=0.7)
        self.img_view_transformer = MODELS.build(img_view_transformer)
        if img_bev_encoder_neck and img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = \
                MODELS.build(img_bev_encoder_backbone)
            self.img_bev_encoder_neck = MODELS.build(img_bev_encoder_neck)

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    #@force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x


    def extract_img_feat(self, img, sensor2ego, cam2img, lidar2cam, ego2global, post_rts, bda):
        """Extract features of images."""
        #img = self.prepare_inputs(img, img_metas)
        x, _ = self.image_encoder(img)
        x, depth = self.img_view_transformer(x, sensor2ego, cam2img, lidar2cam, ego2global, post_rts, bda)
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, imgs, img_metas, **kwargs):
        """Extract features from images and points."""
        if imgs is not None:
            imgs = imgs.contiguous()
            sensor2ego, cam2img, lidar2cam, ego2global = [], [], [], []
            post_rts, bda = [], []

            # post_rts accounts for scaling and translation.
            # Therefore we send the original camera intrinsic for view transform
            for i, meta in enumerate(img_metas):
                sensor2ego.append(meta['sensor2ego'])
                cam2img.append(meta['ori_cam2img'])
                lidar2cam.append(meta['lidar2cam'])
                ego2global.append(meta['ego2global'])
                post_rts.append(meta['post_rts'])
                bda.append(meta['bda_mat'])

            sensor2ego = imgs.new_tensor(np.asarray(sensor2ego))
            cam2img = imgs.new_tensor(np.asarray(cam2img))
            lidar2cam = imgs.new_tensor(np.array(lidar2cam))
            ego2global = imgs.new_tensor(np.asarray(ego2global))
            post_rts = imgs.new_tensor(np.asarray(post_rts))
            bda = imgs.new_tensor(np.asarray(bda))

        img_feats, depth = self.extract_img_feat(imgs, sensor2ego, cam2img, lidar2cam,
                                                 ego2global, post_rts, bda)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    '''
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
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
        img_feats, pts_feats, _ = self.extract_feat(
            imgs=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
    '''

    def forward_pts_train(self,
                          pts_feats,
                          batch_data_samples):

        outs = self.pts_bbox_head(pts_feats)
        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)
        losses = self.pts_bbox_head.loss_by_feat(outs, batch_gt_instance_3d)

        return losses


    def loss(self, inputs=None, data_samples=None, **kwargs):

        img = inputs['imgs']
        img = [img] if img is None else img

        batch_img_metas = [ds.metainfo for ds in data_samples]
        for var, name in [(batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        img_feats, _, _ = self.extract_feat(imgs=img, img_metas=batch_img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, data_samples)

        losses.update(losses_pts)
        return losses

    '''
    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False
    '''

    def predict(self, inputs=None, data_samples=None, **kwargs):
        img = inputs['imgs']
        #points = inputs['points']
        img = [img] if img is None else img

        batch_img_metas = [ds.metainfo for ds in data_samples]
        for var, name in [(batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        #batch_img_metas = self.add_lidar2img(img, batch_img_metas)
        results_list_3d = self.simple_test(batch_img_metas, img, **kwargs)
        #return results_list_3d

        for i, data_sample in enumerate(data_samples):
            results_list_3d_i = InstanceData(
                metainfo=results_list_3d[i]['pts_bbox'])
            data_sample.pred_instances_3d = results_list_3d_i
            data_sample.pred_instances = InstanceData()


        #print("==========================")
        #print("sample_idx:{}".format(batch_img_metas[0]['sample_idx']))
        #print(results_list_3d[0]['pts_bbox']['scores_3d'][0:311])
        #print(results_list_3d[0]['pts_bbox']['bboxes_3d'][0:311])
        #print(results_list_3d[0]['pts_bbox']['labels_3d'][0:311])
        #print("==========================")

        return data_samples


    def simple_test(self,
                    img_metas,
                    imgs=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(imgs=imgs, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results


    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(imgs=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs

