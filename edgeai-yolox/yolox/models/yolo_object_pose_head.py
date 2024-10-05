#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger
import os

from torch._C import Size

#For debugging runtime error
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

from ..data.datasets.lmo import CADModelsLM
from ..data.datasets.ycbv import CADModelsYCBV

class YOLOXObjectPoseHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        dataset = "ycbv",
        adds = True,
        shape_loss = False,
        adds_z = True
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.export_proto = False  # Set it to True while exporting prototxt

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rot_convs = nn.ModuleList()
        self.trn_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.rot_preds = nn.ModuleList()
        self.trn_preds_xy = nn.ModuleList()
        self.trn_preds_z = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        if "lm" in dataset:
            self.cad_models = CADModelsLM()
        elif "ycbv" in dataset:
            self.cad_models = CADModelsYCBV()
        self.adds = adds
        self.shape_loss = shape_loss
        self.adds_z = adds_z
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.rot_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.trn_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.rot_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 6,
                    kernel_size=1,
                    padding=0,
                )
            )
            self.trn_preds_xy.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.trn_preds_z.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.mae_loss = nn.L1Loss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, rot_conv, trn_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.rot_convs, self.trn_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            rot_x = x #NEW
            trn_x = x #NEW

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            rot_feat = rot_conv(rot_x)  #NEW
            rot_preproc = self.rot_preds[k](rot_feat)
            # rot_c1 = F.normalize(rot_preproc[:, :3, :, :], dim=1)
            # rot_c2 = F.normalize(rot_preproc[:, 3:, :, :] - torch.sum(rot_c1 * rot_preproc[:, 3:, :, :], dim=1, keepdim=True) * rot_c1, dim=1)
            # rot_output = torch.cat([rot_c1, rot_c2], dim=1)
            
            trn_feat = trn_conv(trn_x)  #NEW
            trn_xy_output = self.trn_preds_xy[k](trn_feat)
            trn_z_output = self.trn_preds_z[k](trn_feat)

            if self.training:
                rot_c1 = F.normalize(rot_preproc[:, :3, :, :], dim=1)
                rot_c2 = F.normalize(rot_preproc[:, 3:, :, :] - torch.sum(rot_c1 * rot_preproc[:, 3:, :, :], dim=1, keepdim=True) * rot_c1, dim=1)
                rot_output = torch.cat([rot_c1, rot_c2], dim=1)
                output = torch.cat([reg_output, obj_output, cls_output,  rot_output, trn_xy_output, trn_z_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output, cls_output,  rot_preproc, trn_xy_output, trn_z_output], 1
                )
                output[:, 4:5+self.num_classes, :, :] = torch.sigmoid(output[:, 4:5+self.num_classes, :, :])
                rot_c1 = (output[:, -9:-6, :, :] / output[:, -9:-6, :, :].norm(p=2, dim=1, keepdim=True))
                rot_c2 = output[:, -6:-3, :, :] - torch.sum(rot_c1 * output[:, -6:-3, :, :], dim=1, keepdim=True) * rot_c1
                rot_c2 = rot_c2 / rot_c2.norm(p=2, dim=1, keepdim=True)
                #rot_c1 = F.normalize(output[:, -9:-6, :, :], dim=1)
                #rot_c2 = F.normalize(output[:, -6:-3, :, :] - torch.sum(rot_c1 * output[:, -6:-3, :, :], dim=1, keepdim=True) * rot_c1, dim=1)
                output[:, -9:-3, :, :] = torch.cat([rot_c1, rot_c2], dim=1)

            outputs.append(output)


        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )

        elif self.export_proto:
            return outputs
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 14 + self.num_classes  # box_params=4, objectness_score=1, pose_params=9 (rotation=6, translation=3)
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., -3:-1] = (output[..., -3:-1] + grid) * stride
        output[..., -1:] = torch.exp(output[..., -1:])
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., -3:-1] = (outputs[..., -3:-1] + grids) * strides
        outputs[..., -1:] = torch.exp(outputs[..., -1:])
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[..., :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[..., 4:5]  # [batch, n_anchors_all, 1]
        rot_preds = outputs[..., -9:-3] # [batch, n_anchors_all, 6]
        trn_xy_preds = outputs[..., -3:-1] # [batch, n_anchors_all, 2]
        trn_z_preds = outputs[..., -1:] # [batch, n_anchors_all, 1]
        cls_preds = outputs[..., 5:5+self.num_classes]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 14 #change to 14 from 5
        if mixup:
            label_cut = labels[..., :14]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        cls_targets_raw = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        rot_targets = []
        trn_targets = []
        trn_xy_targets = []
        trn_z_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                cls_target_raw = outputs.new_zeros((0,))
                reg_target = outputs.new_zeros((0, 4))
                rot_target = outputs.new_zeros((0, 6))
                trn_xy_target = outputs.new_zeros((0, 2))
                trn_z_target = outputs.new_zeros((0))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_rots_per_image = labels[batch_idx, :num_gt, -9:-3]
                gt_trns_xy_per_image = labels[batch_idx, :num_gt, -3:-1]
                gt_trns_z_per_image = labels[batch_idx, :num_gt, -1:]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                cls_target_raw = gt_matched_classes  #Preserving the raw classes for fetching class_related parameters
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                rot_target = gt_rots_per_image[matched_gt_inds]
                trn_xy_target = gt_trns_xy_per_image[matched_gt_inds]
                trn_z_target = gt_trns_z_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            cls_targets_raw.append(cls_target_raw)
            reg_targets.append(reg_target)
            rot_targets.append(rot_target)
            trn_xy_targets.append(trn_xy_target)
            trn_z_targets.append(trn_z_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        cls_targets_raw = torch.cat(cls_targets_raw, 0)
        reg_targets = torch.cat(reg_targets, 0)
        rot_targets = torch.cat(rot_targets, 0)
        trn_xy_targets = torch.cat(trn_xy_targets, 0)
        trn_z_targets = torch.cat(trn_z_targets, 0)
        #trn_z_targets = trn_z_targets.unsqueeze(-1)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)
        ).sum() / num_fg
        loss_rot = (self.mae_loss(
                rot_preds.view(-1, 6)[fg_masks], rot_targets)
        ).sum() / num_fg
        #loss_rot = 4 * (2 - (rot_preds.view(-1, 6)[fg_masks] * rot_targets).sum()/num_fg) #Cosine similarity loss
        loss_trn_xy = (self.kpts_loss(trn_xy_preds.view(-1, 2)[fg_masks], trn_xy_targets, reg_targets)
        ).sum() / num_fg
        if not self.adds_z:
            loss_trn_z = (self.mae_loss(trn_z_preds.view(-1, 1)[fg_masks], trn_z_targets)
                          ).sum() / num_fg
        else:
            loss_trn_z = self.adds_loss_z(trn_z_preds.view(-1, 1)[fg_masks], trn_z_targets, cls_targets_raw)
        if self.adds:
            pose_targets = torch.cat([trn_xy_targets, trn_z_targets, rot_targets], 1)
            pose_preds = torch.cat([trn_xy_preds.view(-1, 2)[fg_masks], trn_z_preds.view(-1, 1)[fg_masks], rot_preds.view(-1, 6)[fg_masks]], 1)
            if not self.shape_loss:
                pose_targets = self.xy2XY(pose_targets)
                pose_preds = self.xy2XY(pose_preds)
            loss_adds = self.adds_loss(
                pose_preds, pose_targets,
                cls_targets_raw,
                shape_loss = self.shape_loss)
        else:
            loss_adds = 0
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_rot + reg_weight * loss_trn_xy + loss_trn_z + loss_l1 + loss_adds

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_rot,
            loss_trn_xy,
            loss_trn_z,
            loss_l1,
            loss_adds,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


    def kpts_loss(self, kpts_preds, kpts_targets, bbox_targets):
        sigmas = torch.tensor([.26], device=kpts_preds.device) / 10.0
        kpts_preds_x, kpts_targets_x = kpts_preds[:, 0:1], kpts_targets[:, 0:1]
        kpts_preds_y, kpts_targets_y = kpts_preds[:, 1:2], kpts_targets[:, 1:2]
        # OKS based loss
        d = (kpts_preds_x - kpts_targets_x) ** 2 + (kpts_preds_y - kpts_targets_y) ** 2
        bbox_scale = torch.prod(bbox_targets[:, -2:], dim=1, keepdim=True)  #scale derived from bbox gt
        oks = torch.exp(-d / (bbox_scale * (4 * sigmas**2) + 1e-9))
        lkpt = (1 - oks).mean(axis=1)
        return lkpt


    def adds_loss_z(self, preds, targets, cls_targets):
        """
        Normalize the depth error with diameter of that object.
        """
        models_diameter = torch.tensor(list(self.cad_models.models_diameter.values()), device=targets.device)
        cls_targets = cls_targets.to(torch.int64)
        models_diameter = models_diameter[cls_targets]
        loss_trn_z = (self.mae_loss(preds, targets)*100 / models_diameter[:, None]).sum()/len(preds)
        return loss_trn_z


    def adds_loss(self, pose_preds, pose_gt, cls_targets, shape_loss=False):
        """
        Find out the actual ADD(S) score that can be used as a loss
        shape_loss: if set to True, don't use the translation component of the loss. This is called shape loss.
        """
        pose_preds[:, 2] *= 100.0
        pose_gt[:, 2] *= 100.0
        R_pred = torch.cat([pose_preds[:, 3:6, None], pose_preds[:, 6:9, None],
                            torch.cross(pose_preds[:, 3:6, None], pose_preds[:, 6:9, None], dim=1)], dim=-1)
        R_gt = torch.cat([pose_gt[:, 3:6, None], pose_gt[:, 6:9, None],
                          torch.cross(pose_gt[:, 3:6, None], pose_gt[:, 6:9, None], dim=1)], dim=-1)
        loss_adds = None
        for model_idx, sparse_model in self.cad_models.class_to_sparse_model.items():
            cls_idx = cls_targets==model_idx
            sparse_model = torch.tensor(sparse_model, device=cls_targets.device, dtype=cls_targets.dtype)
            if not shape_loss:
                pred_transformed_model = torch.matmul(R_pred[cls_idx], sparse_model.T) + pose_preds[cls_idx][:, :3, None]
                gt_transformed_model = torch.matmul(R_gt[cls_idx], sparse_model.T) + pose_gt[cls_idx][:, :3, None]
            else:
                pred_transformed_model = torch.matmul(R_pred[cls_idx], sparse_model.T)
                gt_transformed_model = torch.matmul(R_gt[cls_idx], sparse_model.T)

            #if torch.sum(cls_idx) != 0:
            if model_idx not in self.cad_models.symmetric_objects.keys():
                mse = ((pred_transformed_model - gt_transformed_model) ** 2).mean(axis=-1).sum(axis=-1)
            else:
                mse = torch.min(((pred_transformed_model[:, :, None, :] - gt_transformed_model[:, :, :, None]) ** 2).sum(axis=1), dim=1)[0]
                mse = mse.mean(axis=-1)
            adds_0p1 = torch.sqrt(mse) / (self.cad_models.models_diameter[model_idx])  #adds_0.1
            if loss_adds is None:
                loss_adds = adds_0p1
            else:
                loss_adds = torch.hstack((loss_adds, adds_0p1))
        loss_adds = loss_adds.mean()
        return loss_adds


    def xy2XY(self, pose):
        if isinstance(self.cad_models.camera_matrix, dict):  #has to be taken care of properly
            camera_matrix = self.cad_models.camera_matrix['camera_uw']
        else:
            camera_matrix = self.cad_models.camera_matrix
        fx, fy, px, py = camera_matrix[0], camera_matrix[4], camera_matrix[2], camera_matrix[5]
        pose[:, 0:1] = (pose[:, 0:1] - px) * ((100.0 * pose[:, 2:3])/fx)
        pose[:, 1:2] = (pose[:, 1:2] - py) * ((100.0 * pose[:, 2:3])/fy)
        return pose

    def kpts_loss_3d(self, preds_Z, targets_Z, preds_xy, targets_xy,  cls_targets):
        """Implement a generalized version of oks loss for 3d keypoints
            Not used curently.
        """
        sigmas = torch.tensor([.26], device=preds_Z.device) / 1.0
        preds_x, targets_x = preds_xy[:, 0:1], targets_xy[:, 0:1]
        preds_y, targets_y = preds_xy[:, 1:2], targets_xy[:, 1:2]
        preds_Z = 100.0 * preds_Z
        targets_Z = 100.0 * targets_Z
        # transform to 3D space
        camera_matrix = self.cad_models.camera_matrix
        fx, fy, px, py =  camera_matrix[0], camera_matrix[4], camera_matrix[2], camera_matrix[5]
        preds_X = (preds_x - px) * (preds_Z /fx)
        preds_Y = (preds_y - py) * (preds_Z /fy)
        targets_X = (targets_x - px) * (targets_Z / fx)
        targets_Y = (targets_y - py) * (targets_Z / fy)
        # OKS based loss calculation
        #d = ((preds_X - targets_X) ** 2 + (preds_Y - targets_Y) ** 2 + (preds_Z - targets_Z) ** 2)/10000
        d = (torch.abs(preds_X - targets_X) +torch.abs(preds_Y - targets_Y)+ torch.abs(preds_Z - targets_Z)) /100.0
        #cuboid_scale = torch.tensor([cuboid_volume**(2.0/3) for cuboid_volume in class_to_cuboid_volume], device=preds_Z.device)
        #class_id = torch.argmax(cls_targets, axis=1)
        #scale = cuboid_scale[class_id][:,None]
        #oks = torch.exp(-d / (10000  + 1e-9))
        #lkpt = (1-oks).mean(axis=1)
        return d


    def iou_loss_3d(self, kpts_preds, kpts_targets, bbox_targets, cuboid_targets):
        "Implemet a generalized version of iou loss based on top view, front view and side view of a box"
        pass