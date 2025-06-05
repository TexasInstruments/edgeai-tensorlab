# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import numpy as np

from torch import Tensor

from mmdet3d.structures import limit_period, xywhr2xyxyr
from mmdet3d.registry import MODELS
from mmdet3d.models.dense_heads import FreeAnchor3DHead

from .box3d_nms import box3d_multiclass_scale_nms, box3d_multiclass_scale_nms_python, \
                       box3d_multiclass_nms, box3d_multiclass_nms_python, box3d_multiclass_nms_python_simple

@MODELS.register_module()
class CustomFreeAnchor3DHead(FreeAnchor3DHead):
    r"""`FreeAnchor <https://arxiv.org/abs/1909.02466>`_ head for 3D detection.

    Note:
        This implementation is directly modified from the `mmdet implementation
        <https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/free_anchor_retina_head.py>`_.
        We find it also works on 3D detection with minor modification, i.e.,
        different hyper-parameters and a additional direction classifier.

    Args:
        pre_anchor_topk (int): Number of boxes that be token in each bag.
        bbox_thr (float): The threshold of the saturated linear function. It is
            usually the same with the IoU threshold used in NMS.
        gamma (float): Gamma parameter in focal loss.
        alpha (float): Alpha parameter in focal loss.
        kwargs (dict): Other arguments are the same as those in :class:`Anchor3DHead`.
    """  
    def __init__(self,
                 pre_anchor_topk: int = 50,
                 bbox_thr: float = 0.6,
                 gamma: float = 2.0,
                 alpha: float = 0.5,
                 init_cfg: dict = None,
                 is_transpose: bool = False,
                 **kwargs) -> None:
        super().__init__(pre_anchor_topk, bbox_thr, gamma, alpha, 
                         init_cfg=init_cfg, **kwargs)
        self.is_transpose = is_transpose


    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward function on a single-scale feature map.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_base_priors * C.
                dir_cls_pred (Tensor | None): Direction classification
                    prediction for a single scale level, the channels
                    number is num_base_priors * 2.
        """
        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            x = x.transpose(-1, -2)

        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_pred

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   input_metas,
                   valid=None,
                   cfg=None,
                   rescale=False,
                   mlvl_anchors=None):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        if mlvl_anchors is None:
            mlvl_anchors = self.prior_generator.grid_anchors(
                featmap_sizes, device=device)

            for i, mlvl_anchor in enumerate(mlvl_anchors):
                mlvl_anchor = mlvl_anchor.to(torch.float32)

        result_list = []
        
        # For onnx export to make the exported model simpler
        if torch.onnx.is_in_onnx_export() and num_levels == 1 and len(input_metas) == 1:
            mlvl_anchors = mlvl_anchors[0].reshape(-1, self.box_code_size)

            cls_score = cls_scores[0].detach() # 0 = level id
            bbox_pred = bbox_preds[0].detach()
            dir_cls_pred = dir_cls_preds[0].detach()

            proposals = self.get_bboxes_single_onnx(cls_score, bbox_pred,
                                                    dir_cls_pred, mlvl_anchors,
                                                    input_metas[0], cfg, rescale)
            result_list.append(proposals)
        else:
            mlvl_anchors = [
                anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
            ]

            for img_id, input_meta in enumerate(input_metas):
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach() for i in range(num_levels)
                ]

                proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                   dir_cls_pred_list, mlvl_anchors,
                                                   input_meta, cfg, rescale)
                result_list.append(proposals)

        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          cfg=None,
                          rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        #if self.use_sigmoid_cls:
        #    # Add a dummy background class to the front when using sigmoid
        #    padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        #    mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        if cfg.get('use_scale_nms', False):
            mlvl_bboxes_for_nms = input_meta['box_type_3d'](mlvl_bboxes, box_dim=self.box_code_size).bev
            results = box3d_multiclass_scale_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                                 mlvl_scores, score_thr, cfg.max_num,
                                                 cfg, mlvl_dir_scores)
        else:
            mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
                mlvl_bboxes, box_dim=self.box_code_size).bev)
            results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                           mlvl_scores, score_thr, cfg.max_num,
                                           cfg, mlvl_dir_scores)

        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels


    def get_bboxes_single_onnx(self,
                               cls_score,
                               bbox_pred,
                               dir_cls_pred,
                               anchors,
                               input_meta,
                               cfg=None,
                               rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        #assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []

        #for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
        #        cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]

        #dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

        #cls_score = cls_score.permute(1, 2,
        #                              0).reshape(-1, self.num_classes)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.num_classes)
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(-1)
        #bbox_pred = bbox_pred.permute(1, 2,
        #                              0).reshape(-1, self.box_code_size)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)

        # scores.shape[0] is constant (e.g 80000) > nms_pre
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            if self.use_sigmoid_cls:
                max_scores, _ = scores.max(dim=1)
            else:
                max_scores, _ = scores[:, :-1].max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            # convert to int32
            topk_inds = topk_inds.to(torch.int32)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_cls_score = dir_cls_score[topk_inds]

        bboxes = self.bbox_coder.decode(anchors, bbox_pred)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        # zero padding (for background class) is not used actually in
        # multi-class NMS. So it can be removed
        #if self.use_sigmoid_cls:
        #    # Add a dummy background class to the front when using sigmoid
        #    padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        #    mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        # For onnx export, we use always normal multi-class NMS
        # regardless of use_scale_nms
        if 0: #cfg.get('use_scale_nms', False):
            mlvl_bboxes_for_nms = input_meta['box_type_3d'](mlvl_bboxes, box_dim=self.box_code_size).bev
        else:
            mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
                mlvl_bboxes, box_dim=self.box_code_size).bev)

        #return mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, mlvl_dir_scores

        score_thr = cfg.get('score_thr', 0)
        if 0: # cfg.get('use_scale_nms', False):
            results = box3d_multiclass_scale_nms_python(mlvl_bboxes, mlvl_bboxes_for_nms,
                                                 mlvl_scores, score_thr, cfg.max_num,
                                                 cfg, mlvl_dir_scores)
        else:
            # box3d_multiclass_nms_python_simple is actually single-class NMS
            # But, with TIDL offload, we can enable multi-class NMS with proper metaarch configs.
            results = box3d_multiclass_nms_python_simple(mlvl_bboxes, mlvl_bboxes_for_nms,
                                           mlvl_scores, score_thr, cfg.max_num,
                                           cfg, mlvl_dir_scores)

        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        return bboxes, scores, labels


    def get_tta_bboxes(self,
                       x_list,
                       input_metas_list,
                       valid=None,
                       cfg=None,
                       rescale=False):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        mlvl_bboxes_list = []
        mlvl_scores_list = []
        mlvl_dir_scores_list = []
        for x, input_metas in zip(x_list, input_metas_list):
            cls_scores, bbox_preds, dir_cls_preds = x

            assert len(cls_scores) == len(bbox_preds)
            assert len(cls_scores) == len(dir_cls_preds)
            num_levels = len(cls_scores)
            featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
            device = cls_scores[0].device
            mlvl_anchors = self.prior_generator.grid_anchors(
                featmap_sizes, device=device)
            mlvl_anchors = [
                anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
            ]

            for img_id in range(len(input_metas)):
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach() for i in range(num_levels)
                ]

                input_meta = input_metas[img_id]
                mlvl_bboxes, mlvl_scores, mlvl_dir_scores = self.get_tta_mlvl_output_single(
                    cls_score_list, bbox_pred_list,
                    dir_cls_pred_list, mlvl_anchors,
                    input_meta, cfg, rescale
                )
                mlvl_bboxes_list.append(mlvl_bboxes)
                mlvl_scores_list.append(mlvl_scores)
                mlvl_dir_scores_list.append(mlvl_dir_scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes_list, dim=0)
        mlvl_scores = torch.cat(mlvl_scores_list, dim=0)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores_list, dim=0)

        result_list = self.get_tta_mlvl_bboxes(
            mlvl_bboxes, mlvl_scores, mlvl_dir_scores, input_meta=input_metas_list[0][0])
        return result_list

    def get_tta_mlvl_output_single(self,
                                   cls_scores,
                                   bbox_preds,
                                   dir_cls_preds,
                                   mlvl_anchors,
                                   input_meta,
                                   cfg=None,
                                   rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        return mlvl_bboxes, mlvl_scores, mlvl_dir_scores

    def get_tta_mlvl_bboxes(self,
                            mlvl_bboxes,
                            mlvl_scores,
                            mlvl_dir_scores,
                            input_meta=None,
                            cfg=None):

        cfg = self.test_cfg if cfg is None else cfg
        score_thr = cfg.get('score_thr', 0)
        if cfg.get('use_scale_nms', False):
            mlvl_bboxes_for_nms = input_meta['box_type_3d'](mlvl_bboxes, box_dim=self.box_code_size).bev
            results = box3d_multiclass_scale_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                                 mlvl_scores, score_thr, cfg.max_num,
                                                 cfg, mlvl_dir_scores)
        else:
            mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
                mlvl_bboxes, box_dim=self.box_code_size).bev)
            results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                           mlvl_scores, score_thr, cfg.max_num,
                                           cfg, mlvl_dir_scores)

        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels
