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
                 img_backbone,
                 img_neck,
                 pts_bbox_head,
                 imgfeat_size):
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

        # for camera frustum creation
        # Image feature size after image backbone. 
        # It may need update for different image backbone
        self.B              = 1
        self.N              = 6
        self.C              = 256
        self.H              = imgfeat_size[0]
        self.W              = imgfeat_size[1]

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
            img_shape = meta['img_shape'][:3]
            meta['img_shape'] = [img_shape] * len(img[0])

        return batch_input_metas


    def prepare_data(self, img, img_metas):
        input_shape = img.shape[-2:]

        self.img_metas = img_metas

        # update real input shae of each single img
        for img_meta in self.img_metas:
            img_meta.update(input_shape=input_shape)


    def create_coords3d(self, img):

        batch_size = self.B
        num_cams   = self.N

        pad_h, pad_w = self.img_metas[0]['pad_shape']
        masks = img.new_ones((batch_size, num_cams, pad_h, pad_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w = self.img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=(self.H, self.W)).to(torch.bool)

        eps = 1e-5

        B, N, C, H, W = self.B, self.N, self.C, self.H, self.W
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


    def forward(self, img, coords3d):
        B = 1
        N, C, H, W = img.size()

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        outs = self.pts_bbox_head(img_feats_reshaped, self.img_metas, masks=None, coords3d=coords3d)

        bbox_list = self.pts_bbox_head.get_bboxes_onnx(
            outs, self.img_metas, rescale=False)

        return bbox_list


class StreamPETR_export_model(nn.Module):
    def __init__(self,
                 stride,
                 use_grid_mask,
                 grid_mask,
                 img_backbone,
                 img_neck,
                 pts_bbox_head,
                 prepare_location,
                 forward_roi_head):
        super().__init__()

        self.stride           = stride
        self.use_grid_mask    = use_grid_mask
        self.img_backbone     = img_backbone
        self.img_neck         = img_neck
        self.pts_bbox_head    = pts_bbox_head
        self.grid_mask        = grid_mask

        self.aux_2d_only      = True
        self.position_level   = 0
        self.len_queue        = 1
        self.prev_scene_token = None

        # Image feature size after image backbone. 
        # It may need update for different image backbone
        self.B              = 1
        self.N              = 6
        self.C              = 256
        self.H              = 16
        self.W              = 44

    def prepare_data(self, img, img_metas):
        #input_shape = img.shape[-2:]
        self.img_metas = img_metas

        ## update real input shae of each single img
        #for img_meta in self.img_metas:
        #    img_meta.update(input_shape=input_shape)
    
    def prepare_location(self, img):
        pad_h, pad_w = self.img_metas[0]['pad_shape']
        bs, n, h, w = self.B, self.N, self.H, self.W

        device = img.device

        shifts_x = (torch.arange(
            0, self.stride*w, step=self.stride,
            dtype=torch.float32, device=device
        ) + self.stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * self.stride, step=self.stride,
            dtype=torch.float32, device=device
        ) + self.stride // 2) / pad_h

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        location = torch.stack((shift_x, shift_y), dim=1)
        location = location.reshape(h, w, 2)[None].repeat(bs*n, 1, 1, 1)

        return location


    def forward(self, img, location,
                memory_embedding, memory_reference_point,
                memory_timestamp, memory_egopose, memory_velo):
        B = 1
        N, C, H, W = img.size()

        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/self.len_queue), C, H, W)

        #return img_feats_reshaped
        topk_indexes = None
        outs = self.pts_bbox_head(location, img_feats_reshaped, self.img_metas, topk_indexes,
                                  memory_embedding, memory_reference_point, memory_timestamp, 
                                  memory_egopose, memory_velo)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, self.img_metas)

        return bbox_list


class Far3D_export_model(nn.Module):
    def __init__(self,
                 stride,
                 use_grid_mask,
                 with_img_neck,
                 single_test,
                 position_level,
                 aux_2d_only,
                 grid_mask,
                 img_backbone,
                 img_neck,
                 pts_bbox_head,
                 img_roi_head):
        super().__init__()

        self.grid_mask        = grid_mask
        self.img_backbone     = img_backbone
        self.img_neck         = img_neck
        self.img_roi_head     = img_roi_head
        self.pts_bbox_head    = pts_bbox_head

        self.stride           = stride
        self.use_grid_mask    = use_grid_mask
        self.with_img_neck    = with_img_neck
        self.single_test      = single_test
        self.position_level   = position_level
        self.aux_2d_only      = aux_2d_only

        self.depth_branch     = None
        self.prev_scene_token = None

        # For locations()
        # Depends on image backbone
        self.img_feat_sizes = [[80, 120], [40, 60], [20, 30], [10, 15]]
        self.B = 1
        self.N = 6

        self.num_propagated           = self.pts_bbox_head.num_propagated
        self.pseudo_reference_points  = self.pts_bbox_head.pseudo_reference_points
        self.memory_len               = self.pts_bbox_head.memory_len
        self.pc_range                 = self.pts_bbox_head.pc_range


    def prepare_data(self, img, img_metas):
        self.img_metas = img_metas

        intrinsics = []
        extrinsics = []
        lidar2imgs = []
        img2lidars = []
        for img_meta in img_metas:
            lidar2img = []
            img2lidar = []
            intrinsics.append(img_meta['intrinsics'])
            extrinsics.append(img_meta['extrinsics'])
            for i in range(len(img_meta['lidar2img'])):
                lidar2img.append(img_meta['lidar2img'][i])
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            lidar2imgs.append(lidar2img)
            img2lidars.append(img2lidar)

        intrinsics = np.asarray(intrinsics)
        extrinsics = np.asarray(extrinsics)
        lidar2imgs = np.asarray(lidar2imgs)
        img2lidars = np.asarray(img2lidars)
        intrinsics = img.new_tensor(intrinsics) / 1e3
        extrinsics = img.new_tensor(extrinsics)[..., :3, :]
        lidar2imgs = img.new_tensor(lidar2imgs)
        img2lidars = img.new_tensor(img2lidars)

        return intrinsics, extrinsics, lidar2imgs, img2lidars

    def locations(self, feat_size, img, stride, pad_h, pad_w):
        """
        From projects.petr.utils.locations
        """
        h, w = feat_size
        device = img.device

        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)

        locations = locations.reshape(h, w, 2)
        return locations


    def prepare_location(self, img):
        pad_h, pad_w = self.img_metas[0]['pad_shape']

        location_r = []
        for i, img_feat_size in enumerate(self.img_feat_sizes):
            bs, n = self.B, self.N
            location = self.locations(img_feat_size, img, self.stride[i], pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
            location_r.append(location)

        return location_r

    def get_memory(self, prev_exist):
        B = prev_exist.size(0)
        memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_velo = self.pts_bbox_head.init_memory(prev_exist)

        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            memory_reference_point[:, :self.num_propagated]  = memory_reference_point[:, :self.num_propagated] + (1 - prev_exist).view(B, 1, 1) * pseudo_reference_points
            memory_egopose[:, :self.num_propagated]  = memory_egopose[:, :self.num_propagated] + (1 - prev_exist).view(B, 1, 1, 1) * torch.eye(4, device=prev_exist.device)

        return memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_velo

    def forward(self,
                img,
                memory_embedding=None,
                memory_reference_point=None,
                memory_timestamp=None,
                memory_egopose=None,
                memory_velo=None,
                intrinsics=None,
                extrinsics=None,
                lidar2imgs=None,
                img2lidars=None):
        B = self.B
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for i in self.position_level:
            BN, C, H, W = img_feats[i].size()
            img_feat_reshaped = img_feats[i].view(B, int(BN/B), C, H, W)
            img_feats_reshaped.append(img_feat_reshaped)

        outs_roi  = self.img_roi_head(img_feats_reshaped)
        bbox_dict = self.img_roi_head.predict_by_feat(outs_roi)
        outs_roi.update(bbox_dict)

        outs = self.pts_bbox_head(img_feats_reshaped, self.img_metas, outs_roi,
                                  memory_embedding, memory_reference_point, memory_timestamp,
                                  memory_egopose, memory_velo,
                                  intrinsics, extrinsics, lidar2imgs, img2lidars)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, self.img_metas)

        return bbox_list


class Far3D_export_img_backbone(nn.Module):
    def __init__(self,
                 use_grid_mask,
                 with_img_neck,
                 position_level,
                 grid_mask,
                 img_backbone,
                 img_neck):
        super().__init__()

        self.grid_mask        = grid_mask
        self.img_backbone     = img_backbone
        self.img_neck         = img_neck

        self.use_grid_mask    = use_grid_mask
        self.with_img_neck    = with_img_neck

        self.position_level   = position_level
        self.B = 1

    def prepare_data(self, img, img_metas):
        self.img_metas = img_metas

        intrinsics = []
        extrinsics = []
        for img_meta in img_metas:
            intrinsics.append(img_meta['intrinsics'])
            extrinsics.append(img_meta['extrinsics'])
        intrinsics = np.asarray(intrinsics)
        extrinsics = np.asarray(extrinsics)
        intrinsics = torch.from_numpy(intrinsics).to(img.device)
        extrinsics = torch.from_numpy(extrinsics).to(img.device)

        return intrinsics, extrinsics


    def forward(self,
                img):
        B = self.B
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for i in self.position_level:
            BN, C, H, W = img_feats[i].size()
            img_feat_reshaped = img_feats[i].view(B, int(BN/B), C, H, W)
            img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped


class Far3D_export_img_roi(nn.Module):
    def __init__(self,
                 img_roi_head):
        super().__init__()
        self.img_roi_head     = img_roi_head

    def prepare_data(self, img, img_metas):
        self.img_metas = img_metas

        intrinsics = []
        extrinsics = []
        for img_meta in img_metas:
            intrinsics.append(img_meta['intrinsics'])
            extrinsics.append(img_meta['extrinsics'])
        intrinsics = np.asarray(intrinsics)
        extrinsics = np.asarray(extrinsics)
        intrinsics = torch.from_numpy(intrinsics).to(img.device)
        extrinsics = torch.from_numpy(extrinsics).to(img.device)

        return intrinsics, extrinsics


    def forward(self,
                img_feats):
        outs_roi  = self.img_roi_head(img_feats)
        bbox_dict = self.img_roi_head.predict_by_feat(outs_roi)
        bbox_roi  = bbox_dict['bbox_list']
        outs_roi.update(bbox_dict)

        # outs_roi have the following fields
        # 1. enc_cls_scores        - list of 4 [6, 10, H, W] tensors
        # 2. enc_bbox_preds        - list of 4 [6, 4, H, W] tensors
        # 3. pred_centers2d_offset - list of 4 [6, 2, H, W] tensors
        # 4. objectnesses          - list of 4 [6, 1, H, W] tensors
        # 5. TokP_indexes          - None
        # 6. depth_logit           - 6x51x80x120 tensor
        # 7. pred_detph            - 6x51x80x120 tensor
        # 8. bbox_list             - list of 6 [X, 4] tensors
        # 9. bbox2d_scores         - [K, 1] tensor
        # 10. valid_indices        - 6x12750x1 tensor
        # But only pred_depth, bbox_list, bbox2d_scores, valid_indies are used in pts_bbox_head
        return outs_roi['pred_depth'], outs_roi['bbox_list'], \
               outs_roi['bbox2d_scores'], outs_roi['valid_indices']


class Far3D_export_pts_bbox(nn.Module):
    def __init__(self,
                 pts_bbox_head):
        super().__init__()

        self.pts_bbox_head    = pts_bbox_head

        # For locations()
        # Depends on image backbone
        self.B = 1
        self.N = 6

        self.num_propagated           = self.pts_bbox_head.num_propagated
        self.pseudo_reference_points  = self.pts_bbox_head.pseudo_reference_points
        self.memory_len               = self.pts_bbox_head.memory_len
        self.pc_range                 = self.pts_bbox_head.pc_range


    def prepare_data(self, img_feats, img_metas):
        self.img_metas = img_metas

        intrinsics = []
        extrinsics = []
        lidar2imgs = []
        img2lidars = []
        for img_meta in img_metas:
            lidar2img = []
            img2lidar = []
            intrinsics.append(img_meta['intrinsics'])
            extrinsics.append(img_meta['extrinsics'])
            for i in range(len(img_meta['lidar2img'])):
                lidar2img.append(img_meta['lidar2img'][i])
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            lidar2imgs.append(lidar2img)
            img2lidars.append(img2lidar)

        intrinsics = np.asarray(intrinsics)
        extrinsics = np.asarray(extrinsics)
        lidar2imgs = np.asarray(lidar2imgs)
        img2lidars = np.asarray(img2lidars)
        intrinsics = img_feats[0].new_tensor(intrinsics) / 1e3
        extrinsics = img_feats[0].new_tensor(extrinsics)[..., :3, :]
        lidar2imgs = img_feats[0].new_tensor(lidar2imgs)
        img2lidars = img_feats[0].new_tensor(img2lidars)

        return intrinsics, extrinsics, lidar2imgs, img2lidars


    def forward(self,
                img_feats,
                outs_roi,
                memory_embedding=None,
                memory_reference_point=None,
                memory_timestamp=None,
                memory_egopose=None,
                memory_velo=None,
                intrinsics=None,
                extrinsics=None,
                lidar2imgs=None,
                img2lidars=None):
        outs = self.pts_bbox_head(img_feats, self.img_metas, outs_roi,
                                  memory_embedding, memory_reference_point, memory_timestamp,
                                  memory_egopose, memory_velo,
                                  intrinsics, extrinsics, lidar2imgs, img2lidars)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, self.img_metas)

        return bbox_list

