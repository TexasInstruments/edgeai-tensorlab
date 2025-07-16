# Copyright (c) 2018-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

##############################################################################

# Also includes parts from: https://github.com/open-mmlab/mmpose
# License: https://github.com/open-mmlab/mmpose/blob/master/LICENSE
#
# Copyright 2018-2020 Open-MMLab. All rights reserved.
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright 2018-2020 Open-MMLab.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

##############################################################################

import os
import sys
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt 

from PIL import ImageDraw
from munkres import Munkres
from numpy.lib.stride_tricks import as_strided
import math

from typing import List, Sequence, Tuple, Dict


def circle_nms(dets, thresh, post_max_size=83):
    """Circular NMS.

    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.

    Args:
        dets (torch.Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int, optional): Max number of prediction to be kept.
            Defaults to 83.

    Returns:
        torch.Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1

    if post_max_size < len(keep):
        return keep[:post_max_size]

    return keep

def _gather_feat(feats, inds, feat_masks=None):
    """Given feats and indexes, returns the gathered feats.

    Args:
        feats (torch.Tensor): Features to be transposed and gathered
            with the shape of [B, 2, W, H].
        inds (torch.Tensor): Indexes with the shape of [B, N].
        feat_masks (torch.Tensor, optional): Mask of the feats.
            Default: None.

    Returns:
        torch.Tensor: Gathered feats.
    """
    dim = feats.shape[2]
    inds = np.broadcast_to(np.expand_dims(inds, 2), (inds.shape[0], inds.shape[1], dim))

    # To resolve shape inference error while simplifying the model
    if dim==1:
        inds=inds.reshape(1, 500, 1)
    elif dim==2:
        inds=inds.reshape(1, 500, 2)
    else:
        inds=inds.reshape(1, 500, 3)

    feats = np.take_along_axis(feats, inds, axis=1)
    if feat_masks is not None:
        feat_masks = np.broadcast_to(np.expand_dims(feat_masks, 2), feats.shape)
        feats = feats[feat_masks]
        feats = feats.reshape(-1, dim)

    return feats


def _transpose_and_gather_feat(feat, ind):
    """Given feats and indexes, returns the transposed and gathered feats.

    Args:
        feat (torch.Tensor): Features to be transposed and gathered
            with the shape of [B, 2, W, H].
        ind (torch.Tensor): Indexes with the shape of [B, N].

    Returns:
        torch.Tensor: Transposed and gathered feats.
    """
    feat = np.ascontiguousarray(np.transpose(feat, (0, 2, 3, 1)))
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat

'''
def nms_rotated(dets,
                scores,
                iou_threshold,
                labels = None,
                clockwise = True):
    """Performs non-maximum suppression (NMS) on the rotated boxes according to
    their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Args:
        dets (torch.Tensor):  Rotated boxes in shape (N, 5).
            They are expected to be in
            (x_ctr, y_ctr, width, height, angle_radian) format.
        scores (torch.Tensor): scores in shape (N, ).
        iou_threshold (float): IoU thresh for NMS.
        labels (torch.Tensor, optional): boxes' label in shape (N,).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.
            `New in version 1.4.3.`

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the
        same data type as the input.
    """
    if dets.shape[0] == 0:
        return dets, None
    if not clockwise:
        #flip_mat = dets.new_ones(dets.shape[-1])
        flip_mat = np.ones(dets.shape[-1], dtype=dets.dtype)
        flip_mat[-1] = -1
        #dets_cw = dets * flip_mat
        dets_cw = np.matmul(dets * flip_mat)
    else:
        dets_cw = dets

    multi_label = labels is not None
    if multi_label:
        dets_wl = np.concatenate((dets_cw, np.expand_dims(labels, 1)), 1)  # type: ignore
    else:
        dets_wl = dets_cw

    # Followings are not needed for CPU implementation
    """
    # sorting in descending order
    scores *= -1
    order = np.argsort(scores, axis=0)
    scores *= -1

    #dets_sorted = dets_wl.index_select(0, order)
    order = np.broadcast_to(order, (dets_wl.shape[1], dets_wl[2]))
    order = np.transpose(order)
    dets_sorted = np.take_along_axis(dets_wl, order, axis=0)
    """

    # TO REVISIT
    keep_inds = ext_module.nms_rotated(dets_wl, scores, order, dets_sorted,
                                       iou_threshold, multi_label)
    dets = np.concatenate((dets[keep_inds], scores[keep_inds].reshape(-1, 1)),
                     dim=1)
    return dets, keep_inds


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

    # for sorting in descending order
    scores *= -1
    order = np.argsort(scores, axis=0)
    scores *= -1
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = np.ascontiguousarray(boxes[order])
    scores = scores[order]

    # xyxyr -> back to xywhr
    # note: better skip this step before nms_bev call in the future
    if xyxyr2xywhr:
        boxes = np.stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
             boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
            axis=-1)

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep
'''


def topk_by_sort(scores, k, axis=-1, descending=True):
    if descending:
        scores *= -1
    ind = np.argsort(scores, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)
    if descending:
        scores *= -1
    val = np.take_along_axis(scores, ind, axis=axis)
    return val, ind

# From https://github.com/HuangJunJie2017/BEVDet
def _topk(scores, K=80):
    """Get indexes based on scores.

    Args:
        scores (torch.Tensor): scores with the shape of [B, N, W, H].
        K (int, optional): Number to be kept. Defaults to 80.

    Returns:
        tuple[torch.Tensor]
            torch.Tensor: Selected scores with the shape of [B, K].
            torch.Tensor: Selected indexes with the shape of [B, K].
            torch.Tensor: Selected classes with the shape of [B, K].
            torch.Tensor: Selected y coord with the shape of [B, K].
            torch.Tensor: Selected x coord with the shape of [B, K].
    """
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = topk_by_sort(scores.reshape(batch, cat, -1), K, axis=2)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds.astype(np.float32) /
               np.array(width, dtype=np.float32)).astype(int).astype(np.float32)
    topk_xs = (topk_inds % width).astype(int).astype(np.float32)

    topk_score, topk_ind = topk_by_sort(topk_scores.reshape(batch, -1), K, axis=1)
    topk_clses = (topk_ind / np.array(K, dtype=np.float32)).astype(int)
    topk_inds = _gather_feat(topk_inds.reshape(batch, -1, 1),
                                  topk_ind).reshape(batch, K)
    topk_ys = _gather_feat(topk_ys.reshape(batch, -1, 1),
                                topk_ind).reshape(batch, K)
    topk_xs = _gather_feat(topk_xs.reshape(batch, -1, 1),
                                topk_ind).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

# Identical to torch.sigmoid()
def sigmoid_np(z):
    return 1.0/(1.0 + np.exp(-1*z))


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list


def limit_period(val,
                 offset: float = 0.5,
                 period: float = np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def nms_normal(boxes, scores ,threshold):

    eps   = 0.0
    x1    = boxes[:,0]
    y1    = boxes[:,1]
    x2    = boxes[:,2]
    y2    = boxes[:,3]

    # Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = scores.argsort()[::-1]

    # Calculating areas of all bboxes
    bbox_areas = (x2 - x1 + eps) *(y2 - y1 + eps)

    # List to keep filtered bboxes.
    filtered = []
    while len(sorted_idx) > 0:
        # Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        # Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)

        # Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x1[rbbox_i], x1[sorted_idx[1:]])
        overlap_ymins = np.maximum(y1[rbbox_i], y1[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x2[rbbox_i], x2[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y2[rbbox_i], y2[sorted_idx[1:]])
        
        # Calculating overlap bbox widths,heights and there by areas.
        overlap_widths  = np.maximum(0,(overlap_xmaxs-overlap_xmins+eps))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+eps))
        overlap_areas   = overlap_widths*overlap_heights
        
        # Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas/(bbox_areas[rbbox_i] + bbox_areas[sorted_idx[1:]] - overlap_areas)
        
        # Select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0] + 1
        delete_idx = np.concatenate(([0], delete_idx))

        # delete the above indices
        sorted_idx = np.delete(sorted_idx,delete_idx)

    return filtered

def nms_normal_bev(boxes,
                   scores,
                   iou_threshold: float):
    """Normal NMS function (for BEV boxes). The overlap of
    two boxes for IoU calculation is defined as the exact overlapping area of
    the two boxes WITH their yaw angle set to 0.

    Args:
        boxes: Input boxes with shape (N, 5).
        scores: Scores of predicted boxes with shape (N).
        thresh (float): Overlap threshold of NMS.

    Returns:
        Tensor: Remaining indices with scores in descending order.
    """
    assert boxes.shape[1] == 5, 'Input boxes shape should be [N, 5]'

    inds = nms_normal(boxes[:, :-1], scores, iou_threshold)
    dets = np.concatenate([boxes[inds], scores[inds].reshape(-1, 1)], axis=1)

    return dets, inds


def box3d_multiclass_nms(
        mlvl_bboxes,
        mlvl_bboxes_for_nms,
        mlvl_scores,
        score_thr: float,
        max_num: int,
        cfg: dict,
        mlvl_dir_scores = None,
        mlvl_attr_scores = None,
        mlvl_bboxes2d = None):
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
    attr_scores = []
    bboxes2d = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

        assert cfg['use_rotate_nms'] is False, 'NMS is NOT supported'
        nms_func = nms_normal_bev

        _, selected = nms_func(_bboxes_for_nms, _scores, cfg['nms_thr'])
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = np.full((len(selected), ),
                            i,
                            dtype=np.int64)
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
        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)
        if mlvl_dir_scores is not None:
            dir_scores = np.concatenate(dir_scores, axis=0)
        if mlvl_attr_scores is not None:
            attr_scores = np.concatenate(attr_scores, axis=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = np.concatenate(bboxes2d, axis=0)
        if bboxes.shape[0] > max_num:
            inds = scores.argsort()[::-1]
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
        bboxes = np.zeros((0, mlvl_bboxes.shape[-1]), dtype=mlvl_bboxes.dtype)
        scores = np.zeros((0, ))
        labels = np.zeros((0, ), dtype=np.int64)
        if mlvl_dir_scores is not None:
            dir_scores = np.zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = np.zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = np.zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results


def box3d_multiclass_scale_nms(
        mlvl_bboxes,
        mlvl_bboxes_for_nms,
        mlvl_scores,
        score_thr: float,
        max_num: int,
        cfg: dict,
        mlvl_dir_scores = None,
        mlvl_attr_scores = None,
        mlvl_bboxes2d = None):
    """Multi-class NMS for 3D boxes. 

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
    # the fg class id range: [0, num_classes]
    num_classes = mlvl_scores.shape[1]
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

        # support only circle_nms
        nms_func = circle_nms
        nms_target_thre = cfg['nms_radius_thr_list'][i]

        nms_rescale = cfg['nms_rescale_factor'][i]
        _bboxes_for_nms[:, 2:4] *= nms_rescale

        _centers = _bboxes_for_nms[:, [0, 1]]
        _bboxes_for_nms = np.concatenate([_centers, _scores.reshape(-1, 1)], axis=1)
        selected = nms_func(_bboxes_for_nms, nms_target_thre)
        selected = np.array(selected, dtype=np.int64)

        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = np.full((len(selected), ), i, dtype=np.int64)
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
        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)
        if mlvl_dir_scores is not None:
            dir_scores = np.concatenate(dir_scores, axis=0)
        if mlvl_attr_scores is not None:
            attr_scores = np.concatenate(attr_scores, axis=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = np.concatenate(bboxes2d, axis=0)
        if bboxes.shape[0] > max_num:
            inds = scores.argsort()[::-1]
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
        bboxes = np.zeros((0, mlvl_bboxes.shape[-1]), dtype=mlvl_bboxes.dtype)
        scores = np.zeros((0, ))
        labels = np.zeros((0, ), dtype=np.int64)
        if mlvl_dir_scores is not None:
            dir_scores = np.zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = np.zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = np.zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results


def _is_polygon_valid(position: np.ndarray, img_size: np.ndarray) -> bool:
    """Judge whether the position is in image.

    Args:
        position (np.ndarray): The position to judge which last dim must
            be two and the format is [x, y].

    Returns:
        bool: Whether the position is in image.
    """
    flag = (position[..., 0] < img_size[1]).all() and \
           (position[..., 0] >= 0).all() and \
           (position[..., 1] < img_size[0]).all() and \
           (position[..., 1] >= 0).all()
    return flag

def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (Tensor or np.ndarray): 2.5D points in 2D images with shape
            [N, 3], 3 corresponds with x, y in the image and depth.
        cam2img (Tensor or np.ndarray): Camera intrinsic matrix. The shape can
            be [3, 3], [3, 4] or [4, 4].

    Returns:
        Tensor or np.ndarray: Points in 3D space with shape [N, 3], 3
        corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = np.concatenate([xys * depths, depths], axis=1)

    pad_cam2img = np.eye(4, dtype=xys.dtype)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = np.linalg.inv(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = np.concatenate((unnormed_xys, np.ones((num_points, 1), dtype=xys.dtype)), axis=1)

    points3D = np.matmul(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D

def points_cam2img(points_3d: np.ndarray,
                   proj_mat: np.ndarray,
                   with_depth: bool = False) -> np.ndarray:
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (Tensor or np.ndarray): Points in shape (N, 3).
        proj_mat (Tensor or np.ndarray): Transformation matrix between
            coordinates.
        with_depth (bool): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        Tensor or np.ndarray: Points in image coordinates with shape [N, 2] if
        ``with_depth=False``, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, \
        'The dimension of the projection matrix should be 2 ' \
        f'instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or \
        (d1 == 4 and d2 == 4), 'The shape of the projection matrix ' \
        f'({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = np.eye(4,  dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = np.concatenate([point_2d_res, point_2d[..., 2:3]], axis=-1)

    return point_2d_res


def rotation_3d_in_axis(points, angles, axis=1):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray or Tensor): Points with shape (N, M, 3).
        angles (np.ndarray or Tensor or float): Vector of angles with shape
            (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it
            will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[Tensor, Tensor] or np.ndarray or
        Tensor: Rotated points with shape (N, M, 3) and rotation matrix with
        shape (N, 3, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = np.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and \
        points.shape[0] == angles.shape[0], 'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = np.stack([
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = np.stack([
                np.stack([rot_cos, rot_sin, zeros]),
                np.stack([-rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = np.stack([
                np.stack([ones, zeros, zeros]),
                np.stack([zeros, rot_cos, rot_sin]),
                np.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(
                f'axis should in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = np.stack([
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos])
        ])

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = np.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    return points_new

def get_camera_box_corners_3d(bboxes_3d):
    """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
    x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

    .. code-block:: none

                     front z
                          /
                         /
           (x0, y0, z1) + -----------  + (x1, y0, z1)
                       /|            / |
                      / |           /  |
        (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                     |  /      .   |  /
                     | / origin    | /
        (x0, y1, z0) + ----------- + -------> right x
                     |             (x1, y1, z0)
                     |
                     v
                down y

    Returns:
        Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
    """
    dims = copy.deepcopy(bboxes_3d[:,3:6])
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1).astype(dims.dtype)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin (0.5, 1, 0.5)
    corners_norm = corners_norm - np.array([0.5, 1, 0.5])
    corners = dims.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    corners = rotation_3d_in_axis(
        corners, bboxes_3d[:, 6], axis=1)
    corners += bboxes_3d[:, :3].reshape(-1, 1, 3)
    return corners


def get_lidar_box_corners_3d(bboxes_3d):
    """ Convert boxes to corners in clockwise order, in the form of (x0y0z0,
    x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y <------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Returns:
        Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
    """
    dims = copy.deepcopy(bboxes_3d[:,3:6])
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1).astype(dims.dtype)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin (0.5, 0.5, 0)
    corners_norm = corners_norm - np.array([0.5, 0.5, 0])
    corners = dims.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    corners = rotation_3d_in_axis(
        corners, bboxes_3d[:, 6], axis=2)
    corners += bboxes_3d[:, :3].reshape(-1, 1, 3)
    return corners


def proj_lidar_bbox3d_to_img(corners_3d, single_lidar2img):
    """Project the 3D bbox on 2D plane.

    Args:
        bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bbox in lidar coordinate
            system to visualize.
        input_meta (dict): Meta information.
    """
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
        np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img = copy.deepcopy(single_lidar2img).reshape(4, 4)
    pts_2d = pts_4d @ lidar2img.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return imgfov_pts_2d


# BBoxes regresssion (post-processing) class for BEVDet
class GetBEVDetBBoxes(object):
    def __init__(self):
        self.max_num = 500

        # should be from config
        self.score_threshold = 0.1
        self.out_size_factor = 8
        self.post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.voxel_size = [0.1, 0.1]
        self.pc_range = [-51.2, -51.2]
        self.pre_max_size = 1000
        self.nms_thr=[0.2]
        self.nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55, \
                                  1.1, 1.0, 1.0, 1.5, 3.5]]
        self.bbox_code_size = 9

    '''
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
            #default_val = [1.0 for _ in range(len(self.task_heads))]
            factor = self.nms_rescale_factor[task_id]
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
                    box_preds[:, :], self.bbox_code_size).bev
                # the nms in 3d detection just remove overlap boxes.
                if isinstance(self.nms_thr, list):
                    nms_thresh = self.nms_thr[task_id]
                else:
                    nms_thresh = self.nms_thr
                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=nms_thresh,
                    pre_max_size=self.pre_max_size,
                    post_max_size=self.post_max_size,
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
                #device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=np.zeros([0, self.bbox_code_size],
                                       dtype=dtype),
                    scores=np.zeros([0], dtype=dtype),
                    labels=np.zeros([0],
                                    dtype=top_labels.dtype))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
    '''

    def bbox_coder_decode(self,
                          heat,
                          rot_sine,
                          rot_cosine,
                          hei,
                          dim,
                          vel,
                          reg=None,
                          task_id=-1):
        """Decode bboxes.
        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.shape

        scores, inds, clses, ys, xs = _topk(heat, K=self.max_num)

        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.reshape(batch, self.max_num, 2)
            xs = xs.reshape(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.reshape(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.reshape(batch, self.max_num, 1) + 0.5
            ys = ys.reshape(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = _transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.reshape(batch, self.max_num, 1)

        rot_cosine = _transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.reshape(batch, self.max_num, 1)
        rot = np.arctan2(rot_sine, rot_cosine)

        # height in the bev
        hei = _transpose_and_gather_feat(hei, inds)
        hei = hei.reshape(batch, self.max_num, 1)

        # dim of the box
        dim = _transpose_and_gather_feat(dim, inds)
        dim = dim.reshape(batch, self.max_num, 3)

        # class label
        clses = clses.reshape(batch, self.max_num).astype(np.float32)
        scores = scores.reshape(batch, self.max_num)

        # Is reshape necesary?
        xs = xs.reshape(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys.reshape(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = np.concatenate([xs, ys, hei, dim, rot], axis=2)
        else:  # exist velocity, nuscene format
            vel = _transpose_and_gather_feat(vel, inds)
            vel = vel.reshape(batch, self.max_num, 2) # is necessary?
            final_box_preds = np.concatenate([xs, ys, hei, dim, rot, vel], axis=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = np.array(self.post_center_range)

            mask = ((final_box_preds[..., :3] >= self.post_center_range[:3]) & 
                    (final_box_preds[..., :3] <= self.post_center_range[3:])).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask = cmask & thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts


    # Based on https://github.com/HuangJunJie2017/BEVDet
    def __call__(self, preds_dicts, info_dict):

        # preds_dicts[0]: reg
        # preds_dicts[1]: height
        # preds_dicts[2]: dim
        # preds_dicts[3]: rot
        # preds_dicts[4]: vel
        # preds_dicts[5]: heatmap
        batch_size = preds_dicts[5].shape[0]
        batch_heatmap = sigmoid_np(preds_dicts[5])

        batch_reg = preds_dicts[0]
        batch_hei = preds_dicts[1]

        batch_dim = np.exp(preds_dicts[2])

        batch_rots = np.expand_dims(preds_dicts[3][:, 0], 1)
        batch_rotc = np.expand_dims(preds_dicts[3][:, 1], 1)

        batch_vel = preds_dicts[4]

        ret = self.bbox_coder_decode(
            batch_heatmap,
            batch_rots,
            batch_rotc,
            batch_hei,
            batch_dim,
            batch_vel,
            reg=batch_reg,
            task_id=0)

        # ret_list[0]: bboxes_3d (LIDARInstance3DBoxes),
        # ret_list[1]: scores_3d,
        # ret_list[2]: labels_3d
        ret_list = []
        ret_list.append(ret[0]['bboxes'])
        ret_list.append(ret[0]['scores'])
        ret_list.append(ret[0]['labels'])

        return ret_list, info_dict

    '''
    def __call__(self, preds_dicts, info_dict):
        rets = []

        # preds_dicts[0]: reg
        # preds_dicts[1]: height
        # preds_dicts[2]: dim
        # preds_dicts[3]: rot
        # preds_dicts[4]: vel
        # preds_dicts[5]: heatmap
        batch_size = preds_dicts[5].shape[0]
        batch_heatmap = sigmoid_np(preds_dicts[5])

        batch_reg = preds_dicts[0]
        batch_hei = preds_dicts[1]

        batch_dim = np.exp(preds_dicts[2])

        batch_rots = np.expand_dims(preds_dicts[3][:, 0], 1)
        batch_rotc = np.expand_dims(preds_dicts[3][:, 1], 1)

        batch_vel = preds_dicts[4]

        temp = self.bbox_coder_decode(
            batch_heatmap,
            batch_rots,
            batch_rotc,
            batch_hei,
            batch_dim,
            batch_vel,
            reg=batch_reg,
            task_id=0)

        batch_reg_preds = [box['bboxes'] for box in temp]
        batch_cls_preds = [box['scores'] for box in temp]
        batch_cls_labels = [box['labels'] for box in temp]

        nms_type = self.nms_type
        if isinstance(nms_type, list):
            nms_type = nms_type[0] # task_id = 0

        if nms_type == 'circle':
            # circular-nms
            ret_task = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                centers = boxes3d[:, [0, 1]]
                boxes = np.concatenate([centers, scores.reshape(-1, 1)], 1)
                keep = np.array(
                    circle_nms(
                        boxes,
                        self.min_radius[0], # task_id = 0
                        post_max_size=self.post_max_size),
                    dtype=np.long)
                boxes3d = boxes3d[keep]
                scores = scores[keep]
                labels = labels[keep]
                ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_task.append(ret)
            rets.append(ret_task)
        else:
            # rotated nms is too complex. So not supported
            """
            rets.append(
                self.get_task_detections(batch_cls_preds, batch_reg_preds,
                                         batch_cls_labels, img_metas,
                                         task_id))
            """
            raise NotImplementedError('Not supported NMS type')

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = np.concatenate([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    #bboxes = img_metas[i]['box_type_3d'](
                    #    bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = np.concatenate([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = np.concatenate([ret[i][k].astype(int) for ret in rets])
            ret_list.append([bboxes, scores, labels])

        # ref_list[0][0]: bboxes_3d (LIDARInstance3DBoxes),
        # ref_list[0][1]: scores_3d,
        # ref_list[0][1]: labels_3d
        return ret_list[0], info_dict
        '''

# Non-maximum suppression (circular NMS) for BEVDet
class BEVDetNMS(object):
    def __init__(self):
        # should be from config
        self.nms_type = ['circle']
        self.num_classes = [10]
        self.min_radius = [4, 12, 10, 1, 0.85, 0.175]

        self.post_max_size = 500

    def __call__(self, preds_dicts, info_dict):
        rets = []

        # preds_dicts[0]: BBoxes
        # preds_dicts[1]: Scores
        # preds_dicts[2]: Lables
        nms_type = self.nms_type
        if isinstance(nms_type, list):
            nms_type = nms_type[0] # task_id = 0

        if nms_type == 'circle':
            # circular-nms
            boxes3d = preds_dicts[0]
            scores = preds_dicts[1]
            labels = preds_dicts[2]
            centers = boxes3d[:, [0, 1]]
            boxes = np.concatenate([centers, scores.reshape(-1, 1)], 1)
            keep = np.array(
                circle_nms(
                    boxes,
                    self.min_radius[0], # task_id = 0
                    post_max_size=self.post_max_size),
                dtype=np.long)
            boxes3d = boxes3d[keep]
            scores = scores[keep]
            labels = labels[keep]
            ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
            rets.append(ret)
        else:
            raise NotImplementedError('Not supported NMS type')

        # Merge branches results
        num_samples = len(rets)

        ret_list = []
        for i in range(num_samples):
            """
            for k in rets[i].keys():
                if k == 'bboxes':
                    bboxes = np.concatenate([ret[k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    #bboxes = img_metas[i]['box_type_3d'](
                    #    bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = np.concatenate([ret[k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[i][k] += flag
                        flag += num_class
                    labels = np.concatenate([ret[k].astype(int) for ret in rets])
            """
            bboxes = rets[i]['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            scores = rets[i]['scores']
            labels = rets[i]['labels'].astype(np.int32)
            ret_list.append([bboxes, scores, labels])

        # ref_list[0][0]: bboxes_3d (LIDARInstance3DBoxes),
        # ref_list[0][1]: scores_3d,
        # ref_list[0][1]: labels_3d
        return ret_list[0], info_dict


# (Multi-layer) Multi-class NMS for FCOS3D
class MultiClassNMS(object):
    def __init__(self):
        # should be from config
        self.nms_type    = ['circle']
        self.num_classes = 10
        self.num_levels  = 5
        self.use_direction_classifier=True
        self.pred_attrs  = True
        self.num_attrs   = 9
        self.attr_background_label = self.num_attrs
        self.strides     = (4, 8, 16, 32, 64)
        # is it from info_dict
        self.rescale     = True
        self.test_cfg    = {'use_rotate_nms': False,
                            'num_across_levels': False,
                            'nms_pre': 1000,
                            'nms_thr': 0.8,
                            'score_thr': 0.05,
                            'min_bbox_size': 0,
                            'max_per_img': 200}
        self.cls_out_channels = self.num_classes
        self.group_reg_dims = (2, 1, 3, 1, 2)
        self.bbox_code_size = 9
        self.dir_offset     = 0.7854

        # to align origin for CameraInstance3DBoxes
        self.dst_origin     = np.array([0.5, 1.0, 0.5])
        self.src_origin     = np.array([0.5, 0.5, 0.5])

    def __call__(self, predicts, info_dict):
        mlvl_bboxes         = predicts[0]
        mlvl_bboxes_for_nms = predicts[1]
        mlvl_nms_scores     = predicts[2]
        mlvl_dir_scores     = predicts[3]
        mlvl_attr_scores    = predicts[4]

        ret_list = []

        cfg = self.test_cfg

        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg['score_thr'],
                                       cfg['max_per_img'], cfg, mlvl_dir_scores,
                                       mlvl_attr_scores)
        bboxes, scores, labels, dir_scores, attrs = results
        attrs = attrs.astype(labels.dtype)  # change data type to int

        # align origin of CameraInstance3DBoxes
        bboxes[:, :3] += bboxes[:, 3:6]*(self.dst_origin - self.src_origin)

        ret_list.append(bboxes)
        ret_list.append(scores)
        ret_list.append(labels)

        if self.pred_attrs and attrs is not None:
            ret_list.append(attrs)

        return ret_list, info_dict


    '''
    def __call__(self, preds_dicts, info_dict):
        cls_scores      = preds_dicts[0:self.num_levels]
        bbox_preds      = preds_dicts[self.num_levels:self.num_levels*2]
        dir_cls_preds   = preds_dicts[self.num_levels*2:self.num_levels*3]
        attr_preds      = preds_dicts[self.num_levels*3:self.num_levels*4]
        centernesses    = preds_dicts[self.num_levels*4:self.num_levels*5]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype)

        result_list = []
        # To modify
        #for img_id in range(len(batch_img_metas)):
        for img_id in range(1):
            img_meta = info_dict
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)

            if self.use_direction_classifier:
                dir_cls_pred_list = select_single_mlvl(dir_cls_preds, img_id)
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(self.num_levels)
                ]

            if self.pred_attrs:
                attr_pred_list = select_single_mlvl(attr_preds, img_id)
            else:
                attr_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(self.num_levels)
                ]

            centerness_pred_list = select_single_mlvl(centernesses, img_id)
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dir_cls_pred_list=dir_cls_pred_list,
                attr_pred_list=attr_pred_list,
                centerness_pred_list=centerness_pred_list,
                mlvl_points=mlvl_points,
                #img_meta=img_meta,
                #cfg=cfg,
                info_dict=info_dict,
                rescale=self.rescale)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list,
                                bbox_pred_list,
                                dir_cls_pred_list,
                                attr_pred_list,
                                centerness_pred_list,
                                mlvl_points,
                                info_dict: dict,
                                rescale: bool = False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            info_dict (dict): Metadata of input image.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            :obj:`InstanceData`: 3D Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                  (num_instances, C), where C >= 7.
        """
        view = np.array(info_dict['cam2img'])
        scale_factor = info_dict['scale_factor']
        cfg = self.test_cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_points)
        mlvl_centers_2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points in zip(cls_score_list, bbox_pred_list,
                              dir_cls_pred_list, attr_pred_list,
                              centerness_pred_list, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = sigmoid_np(np.transpose(cls_score, (1, 2, 0)).reshape(
                -1, self.cls_out_channels))
            dir_cls_pred  = np.transpose(dir_cls_pred, (1, 2, 0)).reshape(-1, 2)
            dir_cls_score = np.argmax(dir_cls_pred, dim=-1)
            attr_pred = np.transpose(attr_pred, (1, 2, 0)).reshape(-1, self.num_attrs)
            attr_score = np.argmax(attr_pred, dim=-1)
            centerness = sigmoid_np(np.transpose(centerness, (1, 2, 0)).reshape(-1))

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, sum(self.group_reg_dims))
            bbox_pred = bbox_pred[:, :self.bbox_code_size]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                attr_score = attr_score[topk_inds]
            # change the offset to actual center predictions
            bbox_pred[:, :2] = points - bbox_pred[:, :2]
            if rescale:
                bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)
            pred_center2d = bbox_pred[:, :3].clone()
            bbox_pred[:, :3] = points_img2cam(bbox_pred[:, :3], view)
            mlvl_centers_2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)

        mlvl_centers_2d = np.concatenate(mlvl_centers_2d)
        mlvl_bboxes = np.concatenate(mlvl_bboxes)
        mlvl_dir_scores = np.concatenate(mlvl_dir_scores)

        # change local yaw to global yaw for 3D nms
        cam2img = np.zeros((4, 4), dtype=mlvl_centers_2d.dtype)
        # double check outputs
        cam2img[:view.shape[0], :view.shape[1]] = np.array(view)
        mlvl_bboxes = self.decode_yaw(mlvl_bboxes, mlvl_centers_2d,
                                      mlvl_dir_scores,
                                      self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(info_dict['box_type_3d'](
            mlvl_bboxes, box_dim=self.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = np.concatenate(mlvl_scores)
        padding = np.zeros((mlvl_scores.shape[0], 1), dtype=mlvl_scores.dtype)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = np.concatenate((mlvl_scores, padding), axis=1)
        mlvl_attr_scores = np.concatenate(mlvl_attr_scores)
        mlvl_centerness = np.concatenate(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg['score_thr'],
                                       cfg['max_per_img'], cfg, mlvl_dir_scores,
                                       mlvl_attr_scores)
        bboxes, scores, labels, dir_scores, attrs = results
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = info_dict['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers_2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        if self.pred_attrs and attrs is not None:
            results.attr_labels = attrs

        return results


    def _get_points_single(self,
                           featmap_size: Tuple[int],
                           stride: int,
                           dtype,
                           flatten: bool = False):
        """Get points of a single scale level.

        Args:
            featmap_size (tuple[int]): Single scale level feature map size.
            stride (int): Downsample factor of the feature map.
            dtype (torch.dtype): Type of points.
            flatten (bool): Whether to flatten the tensor.
                Defaults to False.

        Returns:
            Tensor: points of each image.
        """
        h, w = featmap_size
        x_range = np.arange(w, dtype=dtype)
        y_range = np.arange(h, dtype=dtype)
        y, x = np.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()

        points = np.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    @staticmethod
    def decode_yaw(bbox, centers2d, dir_cls,
                   dir_offset: float, cam2img):
        """Decode yaw angle and change it from local to global.i.

        Args:
            bbox (torch.Tensor): Bounding box predictions in shape
                [N, C] with yaws to be decoded.
            centers2d (torch.Tensor): Projected 3D-center on the image planes
                corresponding to the box predictions.
            dir_cls (torch.Tensor): Predicted direction classes.
            dir_offset (float): Direction offset before dividing all the
                directions into several classes.
            cam2img (torch.Tensor): Camera intrinsic matrix in shape [4, 4].

        Returns:
            torch.Tensor: Bounding boxes with decoded yaws.
        """
        if bbox.shape[0] > 0:
            dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi)
            bbox[..., 6] = \
                dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype)

        bbox[:, 6] = np.arctan2(centers2d[:, 0] - cam2img[0, 2],
                                cam2img[0, 0]) + bbox[:, 6]

        return bbox

    def get_points(self,
                   featmap_sizes,
                   dtype,
                   flatten: bool = False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
            flatten (bool): Whether to flatten the tensor.
                Defaults to False.

        Returns:
            list[tuple]: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, flatten))
        return mlvl_points
    '''


# Multi-class Scale NMS for FastBEV
class MultiClassScaleNMS(object):
    def __init__(self):
        # should be from config
        #self.nms_type    = ['circle']
        #self.num_classes = 10
        #self.num_levels  = 5
        #self.use_direction_classifier=True
        #self.pred_attrs  = True
        #self.num_attrs   = 9
        #self.attr_background_label = self.num_attrs
        #self.strides     = (4, 8, 16, 32, 64)
        # is it from info_dict
        #sself.rescale     = True
        self.test_cfg    = {'use_rotate_nms': False,
                            'num_across_levels': False,
                            'nms_pre': 1000,
                            'max_num': 500,
                            'nms_thr': 0.2,
                            'score_thr': 0.05,
                            'min_bbox_size': 0,
                            'nms_radius_thr_list': [4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1],
                            'nms_rescale_factor': [1.0, 0.7, 0.55, 0.4, 0.7, 1.0, 1.0, 4.5, 9.0, 1.0]
                            }

        #self.bbox_code_size = 9
        self.dir_offset       = 0.7854
        self.dir_limit_offset = 0

        # to align origin for CameraInstance3DBoxes
        #self.dst_origin     = np.array([0.5, 1.0, 0.5])
        #self.src_origin     = np.array([0.5, 0.5, 0.5])

    def __call__(self, predicts, info_dict):
        mlvl_bboxes         = predicts[0]
        mlvl_bboxes_for_nms = predicts[1]
        mlvl_nms_scores     = predicts[2]
        mlvl_dir_scores     = predicts[3]

        # save output
        #print("============= Save output ===============")
        #mlvl_bboxes.tofile("out_mlvl_bboxes.dat")
        #mlvl_nms_scores.tofile("out_mlvl_nms_scores.dat")
        #mlvl_dir_scores.tofile("out_mlvl_dir_scores.dat")

        ret_list = []
        cfg = self.test_cfg

        results = box3d_multiclass_scale_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                             mlvl_nms_scores, cfg['score_thr'],
                                             cfg['max_num'], cfg, mlvl_dir_scores)
        bboxes, scores, labels, dir_scores = results

        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (dir_rot + self.dir_offset + np.pi * dir_scores.astype(bboxes.dtype))

        ret_list.append(bboxes)
        ret_list.append(scores)
        ret_list.append(labels)

        return ret_list, info_dict


# Pack the output into a list of dictionary
class Bbox3d2result(object):
    def __init__(self):
        pass

    def __call__(self, bbox_list, info_dict):

        result_dict = dict(
            bboxes_3d=bbox_list[0],
            scores_3d=bbox_list[1],
            labels_3d=bbox_list[2]
        )

        if len(bbox_list) == 4:
            result_dict['attr_labels'] = bbox_list[3]

        return result_dict, info_dict


class BEVImageSave():
    def __init__(self, num_output_frames=None, score_threshold=0.2, mode='frame'):
        self.num_output_frames = num_output_frames
        self.output_frame_idx = 0
        self.score_threshold = score_threshold
        self.max_label= 10
        self.bbox_color = [(255, 187, 120), (255, 152, 150), (140, 86, 75),
                           (188, 189, 34),  (44, 160, 44),   (247, 182, 210),
                           (196, 156, 148), (23, 190, 207),  (148, 103, 189),
                           (227, 119, 194)]
        self.thickness = 2
        self.mode = mode

        # for LiDAR visualization
        self.sf = 8
        self.lidar_image_size = (1000, 1000, 3)
        self.lidar_image_center = self.lidar_image_size[0] // 2
        self.xy_bound = [-55, 55, -55, 55]
        self.use_dim = 5


    def visualize_LiDAR_detections(self, corners_3d, labels_3d, info_dict):
        # Read LiDAR file
        with open(info_dict['lidar_path'], 'rb') as f:
            if info_dict['lidar_path'].endswith('.pkl'):
                self.use_dim = 4
                import pickle as pkl
                lidar_points = pkl.load(f).values[:, :self.use_dim]
                lidar_points = lidar_points[:,:3].astype(np.float32)
                lidar2ego = info_dict['lidar2ego']
                ego2global = info_dict['ego2globals'][0][0]
                global2lidar = np.linalg.inv(ego2global @ lidar2ego)
                lidar_points = lidar_points @ global2lidar[:3, :3].T + global2lidar[:3, 3]
                pass
            else:
                raw_lidar = f.read()
                lidar_points = np.frombuffer(raw_lidar, dtype=np.float32)
                lidar_points = lidar_points.reshape(-1, self.use_dim)
                lidar_points = lidar_points[:, [0,1,2]]

        lidar_img = np.zeros(self.lidar_image_size, dtype=np.uint8)
        for point in lidar_points:
            x, y, z = point
            if self.xy_bound[0] <= x < self.xy_bound[1] and \
               self.xy_bound[2] <= y < self.xy_bound[3]:
                lidar_img[int(y*self.sf) + self.lidar_image_center, int(x*self.sf) + self.lidar_image_center] = (255, 255, 255)

        for idx, corners in enumerate(corners_3d):
            corners = (corners[:, [0, 1]] * self.sf + self.lidar_image_center).astype(np.int32)
            cv2.line(lidar_img, tuple(corners[0]), tuple(corners[3]), self.bbox_color[labels_3d[idx]], self.thickness)
            cv2.line(lidar_img, tuple(corners[0]), tuple(corners[4]), self.bbox_color[labels_3d[idx]], self.thickness)
            cv2.line(lidar_img, tuple(corners[4]), tuple(corners[7]), self.bbox_color[labels_3d[idx]], self.thickness)
            cv2.line(lidar_img, tuple(corners[3]), tuple(corners[7]), self.bbox_color[labels_3d[idx]], self.thickness)

        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'outputs/LiDAR')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'output_lidar-{:04d}.png'.format(self.output_frame_idx))
        cv2.imwrite(save_path, lidar_img)


    def __call__(self, detections, info_dict):
        if self.output_frame_idx >= self.num_output_frames:
            self.output_frame_idx += 1
            return detections, info_dict

        bboxes_3d = detections['bboxes_3d']
        scores_3d = detections['scores_3d']
        labels_3d = detections['labels_3d']

        bboxes_3d = bboxes_3d[scores_3d > self.score_threshold]
        labels_3d = labels_3d[scores_3d > self.score_threshold].astype(np.int32)
        scores_3d = scores_3d[scores_3d > self.score_threshold]

        if self.mode == 'mv_image':
            img = cv2.imread(info_dict['data_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_size = info_dict['data_shape'][0:2]

            run_dir = info_dict['run_dir']
            save_dir = os.path.join(run_dir, 'outputs')
            os.makedirs(save_dir, exist_ok=True)

            # Convert bboxes to corners
            corners_3d = get_camera_box_corners_3d(bboxes_3d)
            num_bbox   = corners_3d.shape[0]
            points_3d  = corners_3d.reshape(-1, 3)

            # proj_camera_bbox3d_to_img
            cam2img = info_dict['intrins']
            uv_origin = points_cam2img(points_3d, cam2img)
            uv_origin = (uv_origin - 1).round()
            corners_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2)

            if img_size is not None:
                # Filter out the bbox where half of stuff is outside the image.
                # This is for the visualization of multi-view image.
                valid_point_idx = (corners_2d[..., 0] >= 0) & \
                            (corners_2d[..., 0] <= img_size[1]) & \
                            (corners_2d[..., 1] >= 0) & (corners_2d[..., 1] <= img_size[0])  # noqa: E501
                valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
                corners_2d = corners_2d[valid_bbox_idx]

            for idx, corners in enumerate(corners_2d):
                if _is_polygon_valid(corners, img_size):
                    corners = corners.astype(np.int32)
                    cv2.line(img, tuple(corners[0]), tuple(corners[1]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[1]), tuple(corners[2]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[2]), tuple(corners[3]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[3]), tuple(corners[0]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[4]), tuple(corners[5]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[5]), tuple(corners[6]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[6]), tuple(corners[7]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[7]), tuple(corners[4]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[0]), tuple(corners[4]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[1]), tuple(corners[5]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[2]), tuple(corners[6]), self.bbox_color[labels_3d[idx]], self.thickness)
                    cv2.line(img, tuple(corners[3]), tuple(corners[7]), self.bbox_color[labels_3d[idx]], self.thickness)

            save_path = os.path.join(save_dir, 'output_frame-{:04d}.png'.format(self.output_frame_idx))
            cv2.imwrite(save_path, img)

        else:
            imgs = []
            for i, img_path in enumerate(info_dict['data_path']):
                img = cv2.imread(img_path)
                imgs.append(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))

            img_size = info_dict['data_shape'][0:2]

            run_dir = info_dict['run_dir']
            save_dir = os.path.join(run_dir, 'outputs')
            os.makedirs(save_dir, exist_ok=True)

            corners_3d = get_lidar_box_corners_3d(bboxes_3d)
            if info_dict['task_name'] == 'BEVDet':
                ego2lidar = np.linalg.inv(info_dict['lidar2ego']) 
                for i in range(corners_3d.shape[0]):
                    corners_3d[i] = (ego2lidar[:3, :3] @ corners_3d[i].T).T
                    corners_3d[i] = corners_3d[i] + ego2lidar[:3, 3]
            trans2imgs = info_dict['lidar2imgs_org']
            """
            corners_3d = get_lidar_box_corners_3d(bboxes_3d)
            if info_dict['task_name'] == 'BEVDet':
                # BEVDet, BBxes is w.r.t. ego coordinate system
                trans2imgs = info_dict['ego2imgs']
            else:
                trans2imgs = info_dict['lidar2imgs_org']
            """

            for i, single_img in enumerate(imgs):
                trans2img = trans2imgs[i]
                corners_2d = proj_lidar_bbox3d_to_img(corners_3d, trans2img)

                if img_size is not None:
                    # Filter out the bbox where half of stuff is outside the image.
                    # This is for the visualization of multi-view image.
                    valid_point_idx = (corners_2d[..., 0] >= 0) & \
                                (corners_2d[..., 0] <= img_size[1]) & \
                                (corners_2d[..., 1] >= 0) & (corners_2d[..., 1] <= img_size[0])  # noqa: E501
                    valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
                    corners_2d = corners_2d[valid_bbox_idx]

                for idx, corners in enumerate(corners_2d):
                    if _is_polygon_valid(corners, img_size):
                        corners = corners.astype(np.int32)
                        cv2.line(single_img, tuple(corners[0]), tuple(corners[1]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[1]), tuple(corners[2]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[2]), tuple(corners[3]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[3]), tuple(corners[0]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[4]), tuple(corners[5]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[5]), tuple(corners[6]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[6]), tuple(corners[7]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[7]), tuple(corners[4]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[0]), tuple(corners[4]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[1]), tuple(corners[5]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[2]), tuple(corners[6]), self.bbox_color[labels_3d[idx]], self.thickness)
                        cv2.line(single_img, tuple(corners[3]), tuple(corners[7]), self.bbox_color[labels_3d[idx]], self.thickness)

                save_path = os.path.join(save_dir, 'output_frame-{:04d}_{}.png'.format(self.output_frame_idx, i))
                cv2.imwrite(save_path, single_img)

            # Visualize Bird's eye view detections
            self.visualize_LiDAR_detections(corners_3d, labels_3d, info_dict)

        self.output_frame_idx += 1
        return detections, info_dict
