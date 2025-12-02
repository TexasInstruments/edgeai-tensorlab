from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from mmdet.models.layers.transformer import inverse_sigmoid


class PETR_export_model(nn.Module):
    def __init__(self,
                 model,
                 img_backbone,
                 img_neck,
                 pts_bbox_head):
        super().__init__()

        #self.img_backbone   = img_backbone
        #self.img_neck       = img_neck
        #self.pts_bbox_head  = pts_bbox_head
        self.img_backbone    = img_backbone.convert(make_copy=True) if hasattr(img_backbone, "convert") else img_backbone
        self.img_neck        = img_neck.convert(make_copy=True) if hasattr(img_neck, "convert") else img_neck
        if hasattr(pts_bbox_head, "new_bbox_head"):
            self.pts_bbox_head = copy.deepcopy(pts_bbox_head)
            # self.bbox_head.new_bbox_head loses the convert function after deepcopy so using the original
            setattr(self.pts_bbox_head, "new_bbox_head", pts_bbox_head.new_bbox_head.convert(make_copy=True))
            self.pts_bbox_head.cpu()
        elif hasattr(pts_bbox_head, "convert"): # bbox_head is not quantized but rest of the network is quantized
            self.pts_bbox_head = copy.deepcopy(pts_bbox_head).cpu()
        else:
            self.pts_bbox_head = pts_bbox_head

        self.img_feat_size = model.img_feat_size
        self.version = model.version
        if self.version == 'v2':
            self.memory = model.memory
            self.queue  = model.queue
            self.get_temporal_feats = model.get_temporal_feats

        # for camera frustum creation
        # Image feature size after image backbone. 
        # It may need update for different image backbone
        self.B              = 1
        self.N              = self.img_feat_size[0][0]
        self.C              = self.img_feat_size[0][1]
        self.H              = self.img_feat_size[0][2]
        self.W              = self.img_feat_size[0][3]

        self.position_level = self.pts_bbox_head.position_level
        self.with_multiview = self.pts_bbox_head.with_multiview
        self.LID            = self.pts_bbox_head.LID
        self.depth_num      = self.pts_bbox_head.depth_num
        self.depth_start    = self.pts_bbox_head.depth_start
        self.position_range = self.pts_bbox_head.position_range


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
            img_shape = meta['img_shape'][0]
            meta['img_shape'] = [img_shape] * len(meta['cam2img'])

        return batch_input_metas


    def prepare_data(self, img, img_metas):
        input_shape = img.shape[-2:]

        self.img_metas = img_metas

        # update real input shape of each single img
        for img_meta in self.img_metas:
            img_meta.update(input_shape=input_shape)


    def create_coords3d(self, img):

        batch_size = self.B
        num_cams   = len(self.img_metas[0]['lidar2img'])

        pad_h, pad_w = self.img_metas[0]['pad_shape']
        masks = img.new_ones((batch_size, num_cams, pad_h, pad_w))
        for batch_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w = self.img_metas[batch_id]['img_shape'][cam_id]
                masks[batch_id, cam_id, :img_h, :img_w] = 0
        
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=(self.H, self.W)).to(torch.bool)

        eps = 1e-5

        B, N, C, H, W = self.B, num_cams, self.C, self.H, self.W
        coords_h = torch.arange(
            H, device=img.device).float() * pad_h / H
        coords_w = torch.arange(
            W, device=img.device).float() * pad_w / W

        if self.LID:
            index = torch.arange(
                start=0,
                end=self.depth_num,
                step=1,
                device=img.device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(
                start=0,
                end=self.depth_num,
                step=1,
                device=img.device).float()
            bin_size = (self.position_range[3] -
                        self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d
                                             ])).permute(1, 2, 3,
                                                         0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3],
            torch.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        for img_meta in self.img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4,
                                     4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3,
                                    2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)

        return masks, coords3d


    def forward(self, img, coords3d, valid_prev_feats=0, prev_feats_map=0):
        B = 1
        N, C, H, W = img.size()

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        if self.version == 'v2':
            img_feats_all = []
            img_feats_all.append(
                 torch.cat((img_feats_reshaped[0], \
                    valid_prev_feats*prev_feats_map + (1.0-valid_prev_feats)*img_feats_reshaped[0]), dim=1))
            outs = self.pts_bbox_head(img_feats_all, self.img_metas, coords3d=coords3d)
        else:
            outs = self.pts_bbox_head(img_feats_reshaped, self.img_metas, masks=None, coords3d=coords3d)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, self.img_metas, rescale=False)

        if self.version == 'v2':
            return bbox_list, img_feats_reshaped[0]
        else:
            return bbox_list

