# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numba
import numpy as np
import torch

from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.models.layers import nms_bev, circle_nms, circle_nms_python
from torchvision.ops import nms
from torch import Tensor


def box3d_multiclass_scale_nms(
        mlvl_bboxes,
        mlvl_bboxes_for_nms,
        mlvl_scores,
        score_thr,
        max_num,
        cfg,
        mlvl_dir_scores=None,
        mlvl_attr_scores=None,
        mlvl_bboxes2d=None):
    """Multi-class nms for 3D boxes.

    Args:
        mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
        mlvl_scores (torch.Tensor): Multi-level boxes with shape
            (N, C + 1). N is the number of boxes. C is the number of classes.
        score_thr (float): Score thredhold to filter boxes with low
            confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
            of direction classifier. Defaults to None.
        mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
            of attribute classifier. Defaults to None.
        mlvl_bboxes2d (torch.Tensor, optional): Multi-level 2D bounding
            boxes. Defaults to None.

    Returns:
        tuple[torch.Tensor]: Return results after nms, including 3D \
            bounding boxes, scores, labels, direction scores, attribute \
            scores (optional) and 2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []

    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]

        nms_func = {'rotate': nms_bev, 'circle': circle_nms}[cfg.nms_type_list[i]]

        nms_thre = cfg.nms_thr_list[i]
        nms_radius_thre = cfg.nms_radius_thr_list[i]
        nms_target_thre = {'rotate': nms_thre, 'circle': nms_radius_thre}[cfg.nms_type_list[i]]

        nms_rescale = cfg.nms_rescale_factor[i]
        _bboxes_for_nms[:, 2:4] *= nms_rescale

        if cfg.nms_type_list[i] == 'rotate':
            _bboxes_for_nms = xywhr2xyxyr(_bboxes_for_nms)
            selected = nms_func(_bboxes_for_nms, _scores, nms_target_thre)
        else:
            _centers = _bboxes_for_nms[:, [0, 1]]
            _bboxes_for_nms = torch.cat([_centers, _scores.view(-1, 1)], dim=1)
            selected = nms_func(_bboxes_for_nms.detach().cpu().numpy(), nms_target_thre)
            selected = torch.tensor(selected, dtype=torch.long, device=_bboxes_for_nms.device)

        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ), i, dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results

# complex version
def box3d_multiclass_nms_python_complex(
        mlvl_bboxes: Tensor,
        mlvl_bboxes_for_nms: Tensor,
        mlvl_scores: Tensor,
        score_thr: float,
        max_num: int,
        cfg: dict,
        mlvl_dir_scores: Optional[Tensor] = None,
        mlvl_attr_scores: Optional[Tensor] = None,
        mlvl_bboxes2d: Optional[Tensor] = None) -> Tuple[Tensor]:
    """Multi-class NMS for 3D boxes. The IoU used for NMS is defined as the 2D
    IoU between BEV boxes.

    Args:
        mlvl_bboxes (Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (Tensor): Multi-level boxes with shape (N, 5)
            ([x1, y1, x2, y2, ry]). N is the number of boxes.
            The coordinate system of the BEV boxes is counterclockwise.
        mlvl_scores (Tensor): Multi-level boxes with shape (N, C + 1).
            N is the number of boxes. C is the number of classes.
        score_thr (float): Score threshold to filter boxes with low confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (Tensor, optional): Multi-level scores of direction
            classifier. Defaults to None.
        mlvl_attr_scores (Tensor, optional): Multi-level scores of attribute
            classifier. Defaults to None.
        mlvl_bboxes2d (Tensor, optional): Multi-level 2D bounding boxes.
            Defaults to None.

    Returns:
        Tuple[Tensor]: Return results after nms, including 3D bounding boxes,
        scores, labels, direction scores, attribute scores (optional) and
        2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []

    _scores, _labels = torch.max(mlvl_scores, dim=1)

    # add dummy lablels
    _scores = torch.cat([_scores, _scores.new_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])
    _labels = torch.cat([_labels, _labels.new_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])])
    mlvl_bboxes = torch.cat([mlvl_bboxes,
                             torch.zeros([num_classes, 9], dtype=mlvl_bboxes.dtype, device=mlvl_bboxes.device)])
    mlvl_bboxes_for_nms = torch.cat([mlvl_bboxes_for_nms,
                                     torch.zeros([num_classes, 5], dtype=mlvl_bboxes_for_nms.dtype, device=mlvl_bboxes_for_nms.device)])
    mlvl_dir_scores = torch.cat([mlvl_dir_scores, mlvl_dir_scores.new_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

    for i in range(0, num_classes):
        cls_inds = (_labels == i)

        _class_scores = _scores[cls_inds]
        _class_labels = _labels[cls_inds]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]

        selected = nms(_bboxes_for_nms[:, :-1], _class_scores, cfg.nms_thr).to(torch.int32)
        bboxes.append(_mlvl_bboxes[selected][:-1])
        scores.append(_class_scores[selected][:-1])
        labels.append(_class_labels[selected][:-1])

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected][:-1])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)

        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))

    results = (bboxes, scores, labels)
    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )

    return results


# simpler version, which is used for ONNX export
def box3d_multiclass_nms_python(
        mlvl_bboxes: Tensor,
        mlvl_bboxes_for_nms: Tensor,
        mlvl_scores: Tensor,
        score_thr: float,
        max_num: int,
        cfg: dict,
        mlvl_dir_scores: Optional[Tensor] = None,
        mlvl_attr_scores: Optional[Tensor] = None,
        mlvl_bboxes2d: Optional[Tensor] = None) -> Tuple[Tensor]:
    """Multi-class NMS for 3D boxes. The IoU used for NMS is defined as the 2D
    IoU between BEV boxes.

    Args:
        mlvl_bboxes (Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (Tensor): Multi-level boxes with shape (N, 5)
            ([x1, y1, x2, y2, ry]). N is the number of boxes.
            The coordinate system of the BEV boxes is counterclockwise.
        mlvl_scores (Tensor): Multi-level boxes with shape (N, C + 1).
            N is the number of boxes. C is the number of classes.
        score_thr (float): Score threshold to filter boxes with low confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (Tensor, optional): Multi-level scores of direction
            classifier. Defaults to None.
        mlvl_attr_scores (Tensor, optional): Multi-level scores of attribute
            classifier. Defaults to None.
        mlvl_bboxes2d (Tensor, optional): Multi-level 2D bounding boxes.
            Defaults to None.

    Returns:
        Tuple[Tensor]: Return results after nms, including 3D bounding boxes,
        scores, labels, direction scores, attribute scores (optional) and
        2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    _scores, _labels = torch.max(mlvl_scores, dim=1)

    #for i in range(0, num_classes):
    selected = nms(mlvl_bboxes_for_nms[:, :-1], _scores, cfg.nms_thr).to(torch.int32)
    bboxes = mlvl_bboxes[selected]
    scores = _scores[selected]
    labels = _labels[selected]

    if mlvl_dir_scores is not None:
        dir_scores = mlvl_dir_scores[selected]
    if mlvl_attr_scores is not None:
        attr_scores = mlvl_attr_scores[selected]
    if mlvl_bboxes2d is not None:
        bboxes2d = mlvl_bboxes2d[selected]

    """
    if bboxes.shape[0] > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]
        if mlvl_dir_scores is not None:
            dir_scores = dir_scores[inds]
        if mlvl_attr_scores is not None:
            attr_scores = attr_scores[inds]
        if mlvl_bboxes2d is not None:
            bboxes2d = bboxes2d[inds]
    elif bboxes.shape[0] == 0:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))
    
    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )
    """

    results = (bboxes, scores, labels, dir_scores)

    return results


def box3d_multiclass_scale_nms_python(
        mlvl_bboxes,
        mlvl_bboxes_for_nms,
        mlvl_scores,
        score_thr,
        max_num,
        cfg,
        mlvl_dir_scores=None,
        mlvl_attr_scores=None,
        mlvl_bboxes2d=None):
    """Multi-class nms for 3D boxes.

    Args:
        mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
        mlvl_scores (torch.Tensor): Multi-level boxes with shape
            (N, C + 1). N is the number of boxes. C is the number of classes.
        score_thr (float): Score thredhold to filter boxes with low
            confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
            of direction classifier. Defaults to None.
        mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
            of attribute classifier. Defaults to None.
        mlvl_bboxes2d (torch.Tensor, optional): Multi-level 2D bounding
            boxes. Defaults to None.

    Returns:
        tuple[torch.Tensor]: Return results after nms, including 3D \
            bounding boxes, scores, labels, direction scores, attribute \
            scores (optional) and 2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []

    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]

        nms_radius_thre = cfg.nms_radius_thr_list[i]
        nms_target_thre = nms_radius_thre
        
        nms_rescale = cfg.nms_rescale_factor[i]
        _bboxes_for_nms[:, 2:4] *= nms_rescale

        # circle_nms
        _centers = _bboxes_for_nms[:, [0, 1]]
        _bboxes_for_nms = torch.cat([_centers, _scores.view(-1, 1)], dim=1)
        np_array =_bboxes_for_nms.detach().cpu().numpy()
        selected = circle_nms_python(np_array, nms_target_thre)

        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ), i, dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)

        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results
