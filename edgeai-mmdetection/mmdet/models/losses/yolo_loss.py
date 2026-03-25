from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss
from mmdet.registry import MODELS
from mmdet.utils import ConfigType
from einops import rearrange
import math

from mmdet.models.utils.yolo_model_utils import LossConfig, BoxMatcher, Vec2Box, Anc2Box, calculate_iou, transform_bbox

def calculate_ciou(bbox1, bbox2):
    EPS=1e-7
    bbox1 = transform_bbox(bbox1, "xycwh -> xyxy")
    bbox2 = transform_bbox(bbox2, "xycwh -> xyxy")

    # Calculate intersection coordinates
    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(ymax_inter - ymin_inter, min=0)
    # Calculate area of each bbox
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area
    # Calculate IoU
    iou = intersection_area / (union_area + EPS)

    # Calculate diagonal length of the smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(bbox1[..., 0], bbox2[..., 0])
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + EPS

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Compute aspect ratio penalty term
    arctan = torch.atan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    alpha = v / (v - iou + 1 + EPS)

    return iou - (cent_dis / diag_dis + v * alpha)  # CIoU



class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou, iou


class DFLoss(nn.Module):
    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0, self.reg_max - 1.01
        )
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


@MODELS.register_module()
class YOLOV7Loss(nn.Module):
    def __init__(self, loss_cfg: LossConfig, anc2box: Anc2Box, class_num: int = 80, reg_max: int = 16) -> None:
        super().__init__()
        self.class_num = class_num
        self.anc2box = anc2box
        self.device = anc2box.device
        self.loss_cfg = loss_cfg
        self.iou_type = loss_cfg.matcher.iou
        self.anch_topk = loss_cfg.matcher.topk

        self.cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, device=self.device))
        self.obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, device=self.device))

    def forward(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, _, predicts_box, predicts_cnf = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        batch_size = targets.shape[0]
        device = predicts_box[0].device
        obj_scale = [4.0, 1., 0.4]

        #merge targets of batch
        target = []
        for idx,target_per_image in enumerate(targets):
            target += [[idx]+(valid_target.tolist()) for valid_target in target_per_image if valid_target[0] > -1 ]
        target = torch.tensor(target).to(device=device)

        #xyxytoxycwh
        target[:,4:] = target[:,4:] - target[:,2:4]
        target[:,2:4] = target[:,2:4] + (target[:,4:]/2)

        #format predicts
        predict = []
        for i in range(len(self.anc2box.strides)):
            predict.append(torch.cat([predicts_box[i],predicts_cnf[i],predicts_cls[i]], dim=-1))

        loss_weight = self.loss_cfg['objective']
        loss_cls_wt = loss_weight['ClassLoss']
        loss_box_wt = loss_weight['BoxLoss']
        loss_obj_wt = loss_weight['ObjLoss']

        loss_cls, loss_box, loss_obj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        target_cls_idx, target_box, target_indices, anchors = self.get_targets(target, device)

        for idx, pridiction in enumerate(predict):
            batch, anchor, gridy, gridx = target_indices[idx] 
            num_targets = batch.shape[0]
            target_obj = torch.zeros_like(pridiction[..., 0], device=device)  

            if num_targets:
                pred = pridiction[batch, anchor, gridy, gridx] 
                predxy = pred[:, :2].sigmoid()
                predwh = pred[:, 2:4].sigmoid()
                predxy = predxy * 2 - 0.5
                predwh = ((predwh * 2) ** 2) * anchors[idx]
                pred_box = torch.cat((predxy, predwh), 1)  

                iou = calculate_ciou(pred_box, target_box[idx])  
                loss_box += (1.0 - iou).mean() # loss_bbox

                target_obj[batch, anchor, gridy, gridx] =  iou.detach().clamp(0).type(target_obj.dtype)

                target_cls = torch.full_like(pred[:, 5:], 0, device=device)
                target_cls[range(num_targets), target_cls_idx[idx]] = 1.0
                loss_cls += self.cls(pred[:, 5:], target_cls)  # loss_class

            obji = self.obj(pridiction[..., 4], target_obj) # loss_objectness
            loss_obj += obji * obj_scale[idx]

        loss_box *= loss_box_wt * batch_size
        loss_obj *= loss_obj_wt * batch_size
        loss_cls *= loss_cls_wt * batch_size

        return loss_box, loss_obj, loss_cls

    def get_targets(self, target, device):
        num_anchor = self.anc2box.num_anchor
        num_target = target.shape[0]
        
        anchor_indices = torch.arange(num_anchor, device=device).float().view(num_anchor, 1).repeat(1, num_target)
        target = torch.cat((target.repeat(num_anchor, 1, 1), anchor_indices[:, :, None]), 2) 
        strides = self.anc2box.strides
        
        pre_off = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, -0.5]], device=device)  
        
        target_cls, target_box, target_indices, anch = [], [], [], []
        for i in range(len(strides)):
            anchors = torch.tensor(self.anc2box.anchors[i]).view(3,2).float().to(device=self.device)
            anchors/=strides[i]
            grid_size = self.anc2box.anchor_grid[i].shape[2:4]

            norm_factor = torch.ones(7, device=device).long()
            norm_factor[2:6] = torch.tensor(strides[i])
            target_norm = target / norm_factor

            target_wh = target_norm[:, :, 4:6]
            ratio = target_wh / anchors[:, None] 
            anchor_filter = torch.max(ratio, 1. / ratio).max(2)[0] < self.anch_topk
            target_norm = target_norm[anchor_filter]

            gridxy = target_norm[:, 2:4] 
            gridx_inv = torch.tensor(grid_size).to(device=self.device) - gridxy
            grid_a, grid_b = ((gridxy % 1. < 0.5) & (gridxy > 1.)).T
            grid_inv_a, grid_inv_b = ((gridx_inv % 1. < 0.5) & (gridx_inv > 1.)).T
            j = torch.stack((torch.ones_like(grid_a), grid_a, grid_b, grid_inv_a, grid_inv_b))
            target_norm = target_norm.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gridxy)[None] + pre_off[:, None])[j]

            batch, label = target_norm[:, :2].long().T
            grid_xy = target_norm[:, 2:4]
            grid_wh = target_norm[:, 4:6]
            grid_index = (grid_xy - offsets).long()
            grid_i, grid_j = grid_index.T

            anchor_index = target_norm[:, 6].long() 
            target_indices.append((batch, anchor_index, grid_j.clamp_(0, grid_size[1] - 1), grid_i.clamp_(0, grid_size[0] - 1))) 
            target_box.append(torch.cat((grid_xy - grid_index, grid_wh), 1)) 
            anch.append(anchors[anchor_index]) 
            target_cls.append(label) 

        return target_cls, target_box, target_indices, anch


@MODELS.register_module()
class YOLOV9Loss(nn.Module):
    # Implemented from https://github.com/WongKinYiu/yolo
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
        super().__init__()
        self.class_num = class_num
        self.vec2box = vec2box

        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()

        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box.anchor_grid)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def forward(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()))

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = targets_cls.sum()
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou, iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        return loss_iou, loss_dfl, loss_cls

