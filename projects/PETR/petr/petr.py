# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from typing import Dict, List, Union, Optional
import queue
import torch
import copy
import numpy as np
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.det3d_data_sample import ForwardResults, OptSampleList
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from .grid_mask import GridMask

from .onnx_export import export_PETR

@MODELS.register_module()
class PETR(MVXTwoStageDetector):
    """PETR."""

    def __init__(self,
                 version='v1',
                 img_feat_size=None,
                 use_grid_mask=False,
                 save_onnx_model=False,
                 optimized_inference=False,
                 quantized_model=False,
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
                 data_preprocessor=None,
                 **kwargs):
        super(PETR,
              self).__init__(pts_voxel_encoder, pts_middle_encoder,
                             pts_fusion_layer, img_backbone, pts_backbone,
                             img_neck, pts_neck, pts_bbox_head, img_roi_head,
                             img_rpn_head, train_cfg, test_cfg, init_cfg,
                             data_preprocessor)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        # for onnx model export
        self.save_onnx_model     = save_onnx_model
        self.optimized_inference = optimized_inference
        self.quantized_model     = quantized_model
        self.img_feat_size       = img_feat_size
        self.version             = version

        if self.version == 'v2':
            self.memory = dict()
            self.queue = queue.Queue(maxsize=1)
            self.img_metas_save = None


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
                if self.save_onnx_model is True:
                    export_PETR(self, inputs, data_samples,
                        quantized_model=self.quantized_model, opset_version=18, **kwargs)
                    # Export onnx only once
                    self.save_onnx_model = False

                return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')


    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
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
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
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
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss_by_feat(*loss_inputs)

        return losses

    '''
    def _forward(self, mode='loss', **kwargs):
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
    '''

    def _forward(self, inputs: Dict[str, Optional[Tensor]],
                data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]

        batch_img_metas = self.add_lidar2img(img, batch_img_metas)
        img_feats = self.extract_feat(img=img, img_metas=batch_img_metas)
        results = self.pts_bbox_head(img_feats, batch_img_metas)

        return results


    def loss(self,
             inputs=None,
             data_samples=None,
             mode=None,
             points=None,
             img_metas=None,
             gt_bboxes_3d=None,
             gt_labels_3d=None,
             gt_labels=None,
             gt_bboxes=None,
             img=None,
             proposals=None,
             gt_bboxes_ignore=None,
             img_depth=None,
             img_mask=None):
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
        img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]
        batch_gt_instances_3d = [ds.gt_instances_3d for ds in data_samples]
        gt_bboxes_3d = [gt.bboxes_3d for gt in batch_gt_instances_3d]
        gt_labels_3d = [gt.labels_3d for gt in batch_gt_instances_3d]
        gt_bboxes_ignore = None

        batch_img_metas = self.add_lidar2img(img, batch_img_metas)
        img_feats = self.extract_feat(img=img, img_metas=batch_img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, batch_img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def add_prev_img_metas(self, prev_img_meta, img_meta):
        e2g = np.array(img_meta['ego2global'])
        l2e = np.array(img_meta['lidar2ego'])

        e2g_p = np.array(prev_img_meta['ego2global'])
        l2e_p = np.array(prev_img_meta['lidar2ego'])

        for i in range(len(img_meta['lidar2cam'])):
            l2c_p = np.array(prev_img_meta['lidar2cam'][i])
            # Transform [R|t] from the (temporal) previous camera  to the current lidar
            cam2lidar_p_c = np.linalg.inv(l2e) @ np.linalg.inv(e2g) @ e2g_p @ l2e_p @ np.linalg.inv(l2c_p)
            # Transform [R|t] from the current lidar to the (temporal) previous camera
            lidar2cam_p_c = np.linalg.inv(cam2lidar_p_c)
            # Transform [R|t] from the current lidar to the (temporal) previous image
            lidar2img_p_c = np.array(prev_img_meta['cam2img'][i]) @ lidar2cam_p_c

            img_meta['cam2img'].append(prev_img_meta['cam2img'][i])
            img_meta['lidar2cam'] = np.concatenate((img_meta['lidar2cam'], np.expand_dims(lidar2cam_p_c, axis=0)), axis=0)
            #img_meta['lidar2img'].append(lidar2img_p_c)
            img_meta['delta_timestamp'].append(img_meta['timestamp'] - prev_img_meta['img_timestamp'][i])

        return img_meta


    def get_temporal_feats(self, queue, memory, img, img_metas, img_feat_size):
        # Support only batch_size = 1
        for batch_id, img_meta in enumerate(img_metas):
            cur_sample_idx = img_meta['sample_idx']
            if queue.qsize() == 0 or \
                img_meta['scene_token'] != memory[cur_sample_idx-1]['img_meta']['scene_token']:

                #prev_feat = []
                #prev_feat.append(torch.zeros([1]+img_feat_size[0], dtype=img.dtype, device=img.device))
                #prev_feat.append(torch.zeros([1]+img_feat_size[1], dtype=img.dtype, device=img.device))
                prev_feat = 0
                prev_img_meta = copy.deepcopy(img_meta)
            else:
                prev_feat = memory[cur_sample_idx-1]['feature_map']
                prev_img_meta = memory[cur_sample_idx-1]['img_meta']

            # Update img_metas with concatenated fields
            img_meta = self.add_prev_img_metas(prev_img_meta, img_meta)

        return prev_feat, img_metas


    def predict(self, inputs=None, data_samples=None, mode=None, **kwargs):
        img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]
        for var, name in [(batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        prev_feats_map = None
        if self.optimized_inference is False:
            batch_img_metas = self.add_lidar2img(img, batch_img_metas)
            feature_map, results_list_3d = self.simple_test(batch_img_metas, img, prev_feats_map, **kwargs)
        else:
            if self.version == 'v2':
                # Save current batch_img_metas, which
                # will be stored in the queue
                self.img_metas_save = copy.deepcopy(batch_img_metas)
                prev_feats_map, batch_img_metas = self.get_temporal_feats(
                    self.queue, self.memory, img, batch_img_metas, self.img_feat_size)

            batch_img_metas = self.add_lidar2img(img, batch_img_metas)
            feature_map, results_list_3d = self.simple_test(batch_img_metas, img, prev_feats_map, **kwargs)

            if self.version == 'v2':
                if self.queue.full():
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

                # add the current feature map
                # For inference, it should be batch_size = 1
                for batch_id, img_meta in enumerate(self.img_metas_save):
                    # PETRv2 only use the first layer of img_feats after neck
                    # So save only the first layer of img feature_map
                    self.memory[img_meta['sample_idx']] = \
                        dict(feature_map=feature_map[0], img_meta=img_meta)
                    self.queue.put(img_meta['sample_idx'])

        for i, data_sample in enumerate(data_samples):
            results_list_3d_i = InstanceData(
                metainfo=results_list_3d[i]['pts_bbox'])
            data_sample.pred_instances_3d = results_list_3d_i
            data_sample.pred_instances = InstanceData()

        return data_samples

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, img=None, prev_feats_map=0, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        if self.optimized_inference is False:
            bbox_list = [dict() for i in range(len(img_metas))]
            bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        else:
            if torch.is_tensor(prev_feats_map) is False and self.version == 'v1':
                img_feats_all = img_feats
            else:
                if torch.is_tensor(prev_feats_map) is False:
                    is_prev_feat = 0
                else:
                    is_prev_feat = 1

                # PETRv2 only use the first layer of img_feats after neck
                # So concatenatee only the first layer
                img_feats_all = []
                img_feats_all.append(
                     torch.cat((img_feats[0], is_prev_feat*prev_feats_map + (1-is_prev_feat)*img_feats[0]), dim=1))

            bbox_list = [dict() for i in range(len(img_metas))]
            bbox_pts = self.simple_test_pts(img_feats_all, img_metas, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return img_feats, bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    # may need speed-up
    def add_lidar2img(self, img, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            lidar2img_rts = []
            # obtain lidar to image transformation matrix
            for i in range(len(meta['cam2img'])):
                lidar2cam_rt = torch.tensor(meta['lidar2cam'][i]).double()
                intrinsic = torch.tensor(meta['cam2img'][i]).double()
                viewpad = torch.eye(4).double()
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                # The extrinsics mean the transformation from lidar to camera.
                # If anyone want to use the extrinsics as sensor to lidar,
                # please use np.linalg.inv(lidar2cam_rt.T)
                # and modify the ResizeCropFlipImage
                # and LoadMultiViewImageFromMultiSweepsFiles.
                lidar2img_rts.append(lidar2img_rt)
            meta['lidar2img'] = lidar2img_rts
            img_shape = meta['img_shape'][:3]
            meta['img_shape'] = [img_shape] * len(meta['cam2img'])

        return batch_input_metas
