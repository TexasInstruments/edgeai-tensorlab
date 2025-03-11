# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops import multiclass_nms
from mmdeploy.utils import Backend

from mmdet.structures import OptSampleList, SampleList
from mmdet.models.utils.yolo_model_utils import LossConfig, MatcherConfig, Vec2Box, NMSConfig, PostProccess


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.yolov7_head.'
    'YOLOV7Head.predict_by_feat')
def yolov7_head__predict_by_feat(self,
                cls_scores: Tensor,
                preds: Tensor,
                batch_data_samples: SampleList,
                cfg: Optional[ConfigDict] = None,
                rescale: bool = False,
                with_nms: bool = True):
    """Rewrite `predict_by_feat` of `YOLOXHead` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        ctx: Context that contains original meta information.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        objectnesses (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, 1, H, W).
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """
    ctx = FUNCTION_REWRITER.get_context()

    if not with_nms:
        return preds, cls_scores
    
    assert len(cls_scores) == len(preds)
    cfg = self.test_cfg if cfg is None else cfg

    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    # max_output_boxes_per_class = post_params.max_output_boxes_per_class
    max_output_boxes_per_class = 1000
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    nms_type = cfg.nms.get('type')
    dets =  multiclass_nms(
        preds,
        cls_scores,
        max_output_boxes_per_class,
        nms_type=nms_type,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)


    out = torch.cat([dets[0], dets[1].unsqueeze(-1)], dim=-1)
    return out
