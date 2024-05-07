# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet3d.models.utils import (clip_sigmoid, draw_heatmap_gaussian,
                                  gaussian_radius)
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import Det3DDataSample, xywhr2xyxyr
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead

from mmcv.ops import nms_rotated

# This function duplicates functionality of mmcv.ops.iou_3d.nms_bev
# from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms_rotated.
# Nms api will be unified in mmdetection3d one day.
def nms_bev(boxes, scores, thresh, pre_max_size=None, post_max_size=None,
            xyxyr2xywhr=True):
    """NMS function GPU implementation (for BEV boxes). The overlap of two
    boxes for IoU calculation is defined as the exact overlapping area of the
    two boxes. In this function, one can also set ``pre_max_size`` and
    ``post_max_size``.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Default: None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Default: None.

    Returns:
        torch.Tensor: Indexes after NMS.
    """
    assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
    order = scores.sort(0, descending=True)[1]
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()
    scores = scores[order]

    # xyxyr -> back to xywhr
    # note: better skip this step before nms_bev call in the future
    if xyxyr2xywhr:
        boxes = torch.stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
             boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
            dim=-1)

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]

    return keep

@MODELS.register_module()
class BEVDetHead(CenterHead):
    """CenterHead for CenterPoint.

    Args:
        in_channels (list[int] | int, optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        bbox_coder (dict, optional): Bbox coder configs. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        loss_cls (dict, optional): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict, optional): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
        norm_bbox (bool): Whether normalize the bbox predictions.
            Defaults to True.
        train_cfg (dict, optional): Train-time configs. Default: None.
        test_cfg (dict, optional): Test-time configs. Default: None.
        init_cfg (dict, optional): Config for initialization.
    """

    def __init__(self,
                 in_channels: Union[List[int], int] = [128],
                 tasks: Optional[List[dict]] = None,
                 bbox_coder: Optional[dict] = None,
                 common_heads: dict = dict(),
                 loss_cls: dict = dict(
                     type='mmdet.GaussianFocalLoss', reduction='mean'),
                 loss_bbox: dict = dict(
                     type='mmdet.L1Loss', reduction='none', loss_weight=0.25),
                 separate_head: dict = dict(
                     type='mmdet.SeparateHead',
                     init_bias=-2.19,
                     final_kernel=3),
                 share_conv_channel: int = 64,
                 num_heatmap_convs: int = 2,
                 conv_cfg: dict = dict(type='Conv2d'),
                 norm_cfg: dict = dict(type='BN2d'),
                 bias: str = 'auto',
                 norm_bbox: bool = True,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 task_specific=True,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(BEVDetHead, self).__init__(in_channels, tasks, bbox_coder, common_heads,
                                         loss_cls, loss_bbox, separate_head, share_conv_channel,
                                         num_heatmap_convs, conv_cfg, norm_cfg, bias,
                                         norm_bbox, train_cfg, test_cfg, init_cfg, **kwargs)

        self.with_velocity = 'vel' in common_heads.keys()
        self.task_specific = task_specific

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            nms_type = self.test_cfg.get('nms_type')
            if isinstance(nms_type, list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas,
                                             task_id))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])

        return ret_list

    def get_task_detections(self, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas,
                            task_id):
        """Rotate nms for each task.

        Args:
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):
            default_val = [1.0 for _ in range(len(self.task_heads))]
            factor = self.test_cfg.get('nms_rescale_factor',
                                       default_val)[task_id]
            if isinstance(factor, list):
                for cid in range(len(factor)):
                    box_preds[cls_labels == cid, 3:6] = \
                        box_preds[cls_labels == cid, 3:6] * factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] * factor

            # Apply NMS in birdeye view
            top_labels = cls_labels.long()
            top_scores = cls_preds.squeeze(-1) if cls_preds.shape[0]>1 \
                else cls_preds

            if top_scores.shape[0] != 0:
                boxes_for_nms = img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev
                # the nms in 3d detection just remove overlap boxes.
                if isinstance(self.test_cfg['nms_thr'], list):
                    nms_thresh = self.test_cfg['nms_thr'][task_id]
                else:
                    nms_thresh = self.test_cfg['nms_thr']
                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=nms_thresh,
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'],
                    xyxyr2xywhr=False)
            else:
                selected = []

            if isinstance(factor, list):
                for cid in range(len(factor)):
                    box_preds[top_labels == cid, 3:6] = \
                        box_preds[top_labels == cid, 3:6] / factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] / factor

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                predictions_dict = dict(
                    bboxes=selected_boxes,
                    scores=selected_scores,
                    labels=selected_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
