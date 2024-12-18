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

from mmdet.models.utils.yolo_model_utils import LossConfig, BoxMatcher, BoxMatcherV7, Vec2Box, Anc2Box, calculate_iou

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 assing pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm

class BoxLoss2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = (1.0 - iou).mean()
        # loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou, iou

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
class YOLOV7Loss:
    def __init__(self, loss_cfg: LossConfig, anc2box: Anc2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.anc2box = anc2box


        self.cls = BCELoss()
        self.iou = BoxLoss()
        self.obj2 = nn.BCEWithLogitsLoss()
        self.obj = BCELoss()

        self.matcher = BoxMatcherV7(loss_cfg.matcher, self.class_num, anc2box.anchor_scale, anc2box.anchor_boxes,
                                    anc2box.prior_anchor_grid, anc2box.num_anchor)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.anc2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, _, predicts_box, predicts_cnf = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()), num_anc=3)

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.anc2box.scaler[None, :, None]

        

        # obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        # obj_target[pos_inds] = 1

        cls_norm = targets_cls.sum()
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- IOU -- ##
        loss_iou, iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        # loss_iou2 = self.loss_bbox(predicts_box[valid_masks], targets_bbox[valid_masks])
        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls[valid_masks], targets_cls[valid_masks], cls_norm)

        targets_obj = torch.zeros_like(targets_bbox[..., 0])
        targets_obj[valid_masks] = iou.detach().clamp(0).type(targets_obj.dtype)
        obj_norm = targets_obj.sum()

        ## -- OBJ -- ##
        loss_obj = self.obj(predicts_cnf, targets_obj.unsqueeze(-1), obj_norm)
        loss_obj2 = self.obj2(predicts_cnf, targets_obj.unsqueeze(-1))

        return loss_iou, loss_obj, loss_cls 

@MODELS.register_module()
class YOLOV7Loss2:
    def __init__(self, loss_cfg: LossConfig, anc2box: Anc2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.anc2box = anc2box

        self.iou = BoxLoss()
        self.obj = nn.BCEWithLogitsLoss()
        self.cls = nn.BCEWithLogitsLoss()

        self.matcher = BoxMatcherV7(loss_cfg.matcher, self.class_num, anc2box.anchor_scale, anc2box.anchor_boxes,
                                    anc2box.prior_anchor_grid, anc2box.num_anchor)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.anc2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, _, predicts_box, predicts_cnf = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()), num_anc=3)
        batch_size = predicts_cls.shape[0]

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.anc2box.scaler[None, :, None]
        device = predicts_box.device

        loss_iou, loss_cls, loss_obj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # loss_cls2, loss_cls3, loss_obj2 = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        obj_scale = [4.0, 1., 0.4]
        targets_cls_all = []
        targets_bbox_all = []
        predicts_bbox_all =[]
        valid_masks_all = []
        predicts_cls_all = []
        predicts_cnf_all = []
        num_cell_per_stride = [0]
        for idx, stride in enumerate(self.anc2box.strides):
            grid_size = self.anc2box.anchor_grid[idx].shape[-2]
            num_cell_per_stride.append(grid_size*grid_size*self.anc2box.num_anchor + num_cell_per_stride[-1])
            targets_cls_all.append(targets_cls[:,num_cell_per_stride[-2]:num_cell_per_stride[-1],:])
            targets_bbox_all.append(targets_bbox[:,num_cell_per_stride[-2]:num_cell_per_stride[-1],:])
            predicts_bbox_all.append(predicts_box[:,num_cell_per_stride[-2]:num_cell_per_stride[-1],:])
            valid_masks_all.append(valid_masks[:,num_cell_per_stride[-2]:num_cell_per_stride[-1]])
            predicts_cls_all.append(predicts_cls[:,num_cell_per_stride[-2]:num_cell_per_stride[-1],:])
            predicts_cnf_all.append(predicts_cnf[:,num_cell_per_stride[-2]:num_cell_per_stride[-1],:])

            cls_norm = targets_cls_all[-1].sum()
            box_norm = targets_cls_all[-1].sum(-1)[valid_masks_all[-1]]

            loss_bbox, iou = self.iou(predicts_bbox_all[-1], targets_bbox_all[-1], valid_masks_all[-1], box_norm, cls_norm)
            loss_iou += loss_bbox

            loss_cls += self.cls(predicts_cls_all[-1][valid_masks_all[-1]], targets_cls_all[-1][valid_masks_all[-1]])
            # loss_cls += self.cls(predicts_cls_all[-1][valid_masks_all[-1]], targets_cls_all[-1][valid_masks_all[-1]], cls_norm)
            # loss_cls3 += self.loss_cls(predicts_cls_all[-1][valid_masks_all[-1]], targets_cls_all[-1][valid_masks_all[-1]])

            targets_obj = torch.zeros_like(targets_bbox_all[-1][..., 0])
            targets_obj[valid_masks_all[-1]] = iou.detach().clamp(0).type(targets_obj.dtype)

            loss_objectness = self.obj(predicts_cnf_all[-1], targets_obj.unsqueeze(-1))
            loss_obj += loss_objectness * obj_scale[idx]

            # loss_objectness2 = self.loss_conf(predicts_cnf_all[-1], targets_obj.unsqueeze(-1))
            # loss_obj2 += loss_objectness2 * obj_scale[idx]
  
        loss_iou *= batch_size
        loss_obj *= batch_size
        loss_cls *= batch_size

        return loss_iou, loss_obj, loss_cls
    





@MODELS.register_module()
class YOLOV7Loss3:
    def __init__(self, loss_cfg: LossConfig, anc2box: Anc2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.anc2box = anc2box
        self.device = anc2box.device
        self.loss_cfg = loss_cfg

        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, device=self.device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, device=self.device))

        self.matcher = BoxMatcherV7(loss_cfg.matcher, self.class_num, anc2box.anchor_scale, anc2box.anchor_boxes,
                                    anc2box.prior_anchor_grid, anc2box.num_anchor)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.anc2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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

        #format predicts
        predict = []
        for i in range(len(self.anc2box.strides)):
            predict.append(torch.cat([predicts_box[i],predicts_cnf[i],predicts_cls[i]], dim=-1))
            # predict[i] = rearrange(predict[i], 'b (a g) p -> b a g p', a=self.anc2box.num_anchor)
            # predict[i] = rearrange(predict[i], 'b a (w h) p -> b a w h p', w=self.anc2box.anchor_grid[i].shape[-2])

        loss_weight = self.loss_cfg['objective']
        loss_cls_wt = loss_weight['ClassLoss']
        loss_box_wt = loss_weight['BoxLoss']
        loss_obj_wt = loss_weight['ObjLoss']

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.get_targets(predict, target)

                # Losses
        for i, pi in enumerate(predict):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                tobj[b, a, gj, gi] =  iou.detach().clamp(0).type(tobj.dtype)

                # Classification
                if True: #self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], 0, device=device)  # targets
                    t[range(n), tcls[i]] = 1.0
                    #t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE


            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * obj_scale[i]  # obj loss



        lbox *= loss_box_wt * batch_size
        lobj *= loss_obj_wt * batch_size
        lcls *= loss_cls_wt * batch_size

        return lbox, lobj, lcls

    def get_targets(self, predict, target):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.anc2box.num_anchor, target.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        shrink = torch.ones(7, device=target.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=target.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        target = torch.cat((target.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        strides = self.anc2box.strides
        
        

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=target.device).float() * g  # offsets
        

        for i in range(len(strides)):
            anchors = torch.tensor(self.anc2box.anchors[i]).view(3,2).float().to(device=self.device)
            anchors/=strides[i]
            grid_size = self.anc2box.anchor_grid[i].shape[2:4]
            shrink[2:6] = torch.tensor(strides[i])
            
            t = target / shrink
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < 8.0  # self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = torch.tensor(grid_size).to(device=self.device) - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = target[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, grid_size[1] - 1), gi.clamp_(0, grid_size[0] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch




















@MODELS.register_module()
class YOLOLoss:
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
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

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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


#not required TODO:remove
class DualLoss:
    def __init__(self, cfg: LossConfig, vec2box) -> None:
        loss_cfg = cfg.task.loss
        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]

    def __call__(
        self, aux_predicts: List[Tensor], main_predicts: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # TODO: Need Refactor this region, make it flexible!
        aux_iou, aux_dfl, aux_cls = self.loss(aux_predicts, targets)
        main_iou, main_dfl, main_cls = self.loss(main_predicts, targets)

        loss_dict = {
            "BoxLoss": self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            "DFLoss": self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            "BCELoss": self.cls_rate * (aux_cls * self.aux_rate + main_cls),
        }
        loss_sum = sum(list(loss_dict.values())) / len(loss_dict)
        return loss_sum, loss_dict


def create_loss_function(cfg: LossConfig, vec2box) -> DualLoss:
    # TODO: make it flexible, if cfg doesn't contain aux, only use SingleLoss
    loss_function = DualLoss(cfg, vec2box)
    logger.info("✅ Success load loss function")
    return loss_function





    # def get_targets(self, p, targets):
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     na, nt = 3, targets.shape[0]  # number of anchors, targets
    #     tcls, tbox, indices, anch = [], [], [], []
    #     gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
    #     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    #     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    #     g = 0.5  # bias
    #     off = torch.tensor([[0, 0],
    #                         [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
    #                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
    #                         ], device=targets.device).float() * g  # offsets

    #     for i in range(self.nl):
    #         anchors = self.anchors[i]
    #         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain   ## [80,80,80,80]

    #         # Match targets to anchors
    #         t = targets * gain
    #         if nt:
    #             # Matches
    #             r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
    #             j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
    #             # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
    #             t = t[j]  # filter

    #             # Offsets
    #             gxy = t[:, 2:4]  # grid xy
    #             gxi = gain[[2, 3]] - gxy  # inverse
    #             j, k = ((gxy % 1. < g) & (gxy > 1.)).T
    #             l, m = ((gxi % 1. < g) & (gxi > 1.)).T
    #             j = torch.stack((torch.ones_like(j), j, k, l, m))
    #             t = t.repeat((5, 1, 1))[j]
    #             offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
    #         else:
    #             t = targets[0]
    #             offsets = 0

    #         # Define
    #         b, c = t[:, :2].long().T  # image, class
    #         gxy = t[:, 2:4]  # grid xy
    #         gwh = t[:, 4:6]  # grid wh
    #         gij = (gxy - offsets).long()
    #         gi, gj = gij.T  # grid xy indices

    #         # Append
    #         a = t[:, 6].long()  # anchor indices
    #         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
    #         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
    #         anch.append(anchors[a])  # anchors
    #         tcls.append(c)  # class

    #     return tcls, tbox, indices, anch