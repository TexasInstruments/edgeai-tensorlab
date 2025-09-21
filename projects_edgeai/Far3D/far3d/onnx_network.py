from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np

class StreamPETR_export_model(nn.Module):
    def __init__(self,
                 stride,
                 use_grid_mask,
                 grid_mask,
                 img_backbone,
                 img_neck,
                 pts_bbox_head):
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

        # Image feature size after image backbone. 
        # It may need update for different image backbone
        self.B              = 1
        self.N              = 6
        self.C              = 256
        self.H              = 16
        self.W              = 44

        self.num_propagated           = self.pts_bbox_head.num_propagated
        self.pseudo_reference_points  = self.pts_bbox_head.pseudo_reference_points
        self.pc_range                 = self.pts_bbox_head.pc_range
        self.position_range           = self.pts_bbox_head.position_range
        self.coords_d                 = self.pts_bbox_head.coords_d

    def prepare_data(self, img_metas):
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

    def create_coords3d(self, location):
        eps = 1e-5
        BN, H, W, _ = location.shape

        intrinsics = []
        img2lidars = []
        for img_meta in self.img_metas:
            intrinsic = []
            img2lidar = []
            for i in range(len(img_meta['intrinsics'])):
                intrinsic.append(img_meta['intrinsics'][i])
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            intrinsics.append(np.asarray(intrinsic))
            img2lidars.append(np.asarray(img2lidar))
        intrinsics = np.asarray(intrinsics)
        intrinsics = location.new_tensor(intrinsics)  # (B, N, 4, 4)
        img2lidars = np.asarray(img2lidars)
        img2lidars = location.new_tensor(img2lidars)  # (B, N, 4, 4)

        B = img2lidars.size(0)

        intrinsic = torch.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)
        num_sample_tokens = LEN

        pad_h, pad_w  = self.img_metas[0]['pad_shape']
        location[..., 0] = location[..., 0] * pad_w
        location[..., 1] = location[..., 1] * pad_h

        D = self.coords_d.shape[0]

        location = location.detach().view(B, num_sample_tokens, 1, 2)
        topk_centers = location.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)
        #img2lidars = topk_gather(img2lidars, topk_indexes)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)

        #intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        return coords3d, cone


    def get_memory(self, prev_exist):
        B = prev_exist.size(0)
        memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_velo = self.pts_bbox_head.init_memory(prev_exist)

        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            memory_reference_point[:, :self.num_propagated]  = memory_reference_point[:, :self.num_propagated] + (1 - prev_exist).view(B, 1, 1) * pseudo_reference_points
            memory_egopose[:, :self.num_propagated]  = memory_egopose[:, :self.num_propagated] + (1 - prev_exist).view(B, 1, 1, 1) * torch.eye(4, device=prev_exist.device)

        return memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_velo


    def get_ego_pose_and_timestamp(self):
        ego_pose = []
        timestamp = []
        for img_meta in self.img_metas:
            ego_pose.append(img_meta['ego_pose'])
            timestamp.append(img_meta['timestamp'])
        ego_pose = np.asarray(ego_pose)
        ego_pose = torch.from_numpy(ego_pose).to(torch.float32)
        timestamp = np.asarray(timestamp)
        timestamp = torch.from_numpy(timestamp)

        return ego_pose, timestamp


    def forward(self, img,
                memory_embedding, memory_reference_point,
                memory_timestamp, memory_egopose, memory_velo,
                coords_3d, cone,
                ego_pose, timestamp):
        B = self.B
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/self.len_queue), C, H, W)

        # location = None, instead send coord_3d directly
        # We can send cood_3d directly only when topk_indexes is None (i.e. self.aux_2d_only = True)
        # if not, we have to send location intead of coord_3d
        location = None
        topk_indexes = None
        outs, out_memory_embedding, out_memory_reference_point, \
            out_memory_timestamp, out_memory_egopose, out_memory_velo = \
            self.pts_bbox_head(location, img_feats_reshaped, self.img_metas, topk_indexes,
                               memory_embedding, memory_reference_point, memory_timestamp,
                               memory_egopose, memory_velo, coords_3d, cone,
                               ego_pose, timestamp)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, self.img_metas)

        return bbox_list, out_memory_embedding, out_memory_reference_point, \
               out_memory_timestamp, out_memory_egopose, out_memory_velo


class StreamPETR_export_img_backbone(nn.Module):
    def __init__(self,
                 use_grid_mask,
                 grid_mask,
                 img_backbone,
                 img_neck):
        super().__init__()

        self.use_grid_mask    = use_grid_mask
        self.img_backbone     = img_backbone
        self.img_neck         = img_neck
        self.grid_mask        = grid_mask

        self.position_level   = 0
        self.len_queue        = 1
        self.B               = 1

    def forward(self, img):
        B = self.B
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/self.len_queue), C, H, W)

        return img_feats_reshaped


class StreamPETR_export_pts_bbox(nn.Module):
    def __init__(self,
                 stride,
                 pts_bbox_head):
        super().__init__()

        self.stride           = stride
        self.pts_bbox_head    = pts_bbox_head
        self.img_metas        = None

        # Image feature size after image backbone. 
        # It may need update for different image backbone
        self.B              = 1
        self.N              = 6
        self.C              = 256
        self.H              = 16
        self.W              = 44

        self.num_propagated           = self.pts_bbox_head.num_propagated
        self.pseudo_reference_points  = self.pts_bbox_head.pseudo_reference_points
        self.pc_range                 = self.pts_bbox_head.pc_range
        self.position_range           = self.pts_bbox_head.position_range
        self.coords_d                 = self.pts_bbox_head.coords_d


    def prepare_data(self, img_metas):
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

    def create_coords3d(self, location):
        eps = 1e-5
        BN, H, W, _ = location.shape

        intrinsics = []
        img2lidars = []
        for img_meta in self.img_metas:
            intrinsic = []
            img2lidar = []
            for i in range(len(img_meta['intrinsics'])):
                intrinsic.append(img_meta['intrinsics'][i])
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            intrinsics.append(np.asarray(intrinsic))
            img2lidars.append(np.asarray(img2lidar))
        intrinsics = np.asarray(intrinsics)
        intrinsics = location.new_tensor(intrinsics)  # (B, N, 4, 4)
        img2lidars = np.asarray(img2lidars)
        img2lidars = location.new_tensor(img2lidars)  # (B, N, 4, 4)

        B = img2lidars.size(0)

        intrinsic = torch.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)
        num_sample_tokens = LEN

        pad_h, pad_w  = self.img_metas[0]['pad_shape']
        location[..., 0] = location[..., 0] * pad_w
        location[..., 1] = location[..., 1] * pad_h

        D = self.coords_d.shape[0]

        location = location.detach().view(B, num_sample_tokens, 1, 2)
        topk_centers = location.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)

        # for spatial alignment in focal petr
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        return coords3d, cone


    def get_memory(self, prev_exist):
        B = prev_exist.size(0)
        memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_velo = self.pts_bbox_head.init_memory(prev_exist)

        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            memory_reference_point[:, :self.num_propagated]  = memory_reference_point[:, :self.num_propagated] + (1 - prev_exist).view(B, 1, 1) * pseudo_reference_points
            memory_egopose[:, :self.num_propagated]  = memory_egopose[:, :self.num_propagated] + (1 - prev_exist).view(B, 1, 1, 1) * torch.eye(4, device=prev_exist.device)

        return memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_velo


    def get_ego_pose_and_timestamp(self):
        ego_pose = []
        timestamp = []
        for img_meta in self.img_metas:
            ego_pose.append(img_meta['ego_pose'])
            timestamp.append(img_meta['timestamp'])
        ego_pose = np.asarray(ego_pose)
        ego_pose = torch.from_numpy(ego_pose).to(torch.float32)
        timestamp = np.asarray(timestamp)
        timestamp = torch.from_numpy(timestamp)

        return ego_pose, timestamp


    def forward(self, img_feats_reshaped,
                memory_embedding, memory_reference_point,
                memory_timestamp, memory_egopose, memory_velo,
                coords_3d, cone,
                ego_pose, timestamp):

        # location = None, instead send coord_3d directly
        # We can send cood_3d directly only when topk_indexes is None (i.e. self.aux_2d_only = True)
        # if not, we have to send location intead of coord_3d
        location = None
        topk_indexes = None

        if torch.onnx.is_in_onnx_export():
            outs, out_memory_embedding, out_memory_reference_point, \
                out_memory_timestamp, out_memory_egopose, out_memory_velo = \
                    self.pts_bbox_head(location, img_feats_reshaped, self.img_metas, topk_indexes,
                                       memory_embedding, memory_reference_point, memory_timestamp,
                                       memory_egopose, memory_velo, coords_3d, cone,
                                       ego_pose, timestamp)
        else:
            outs =  self.pts_bbox_head(location, img_feats_reshaped, self.img_metas, topk_indexes,
                                       memory_embedding, memory_reference_point, memory_timestamp,
                                       memory_egopose, memory_velo, coords_3d, cone,
                                       ego_pose, timestamp)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, self.img_metas)

        if torch.onnx.is_in_onnx_export():
            return bbox_list, out_memory_embedding, out_memory_reference_point, \
                   out_memory_timestamp, out_memory_egopose, out_memory_velo
        else:
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

    def get_ego_pose_and_timestamp(self):
        ego_pose = []
        timestamp = []
        for img_meta in self.img_metas:
            ego_pose.append(img_meta['ego_pose'])
            timestamp.append(img_meta['timestamp'])
        ego_pose = np.asarray(ego_pose)
        ego_pose = torch.from_numpy(ego_pose).to(torch.float32)
        timestamp = np.asarray(timestamp)
        timestamp = torch.from_numpy(timestamp)

        return ego_pose, timestamp

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
                img2lidars=None,
                ego_pose=None,
                timestamp=None):
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

        outs, out_memory_embedding, out_memory_reference_point, \
           out_memory_timestamp, out_memory_egopose, out_memory_velo = \
            self.pts_bbox_head(img_feats_reshaped, self.img_metas, outs_roi,
                               memory_embedding, memory_reference_point, memory_timestamp,
                               memory_egopose, memory_velo,
                               intrinsics, extrinsics, lidar2imgs, img2lidars,
                               ego_pose, timestamp)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, self.img_metas)

        return bbox_list, out_memory_embedding, out_memory_reference_point, \
               out_memory_timestamp, out_memory_egopose, out_memory_velo


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


    def forward(self,
                img_feats):
        outs_roi  = self.img_roi_head(img_feats)
        bbox_dict = self.img_roi_head.predict_by_feat(outs_roi)
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


    def get_memory(self, prev_exist):
        B = prev_exist.size(0)
        memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_velo = self.pts_bbox_head.init_memory(prev_exist)

        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            memory_reference_point[:, :self.num_propagated]  = memory_reference_point[:, :self.num_propagated] + (1 - prev_exist).view(B, 1, 1) * pseudo_reference_points
            memory_egopose[:, :self.num_propagated]  = memory_egopose[:, :self.num_propagated] + (1 - prev_exist).view(B, 1, 1, 1) * torch.eye(4, device=prev_exist.device)

        return memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_velo

    def get_ego_pose_and_timestamp(self):
        ego_pose = []
        timestamp = []
        for img_meta in self.img_metas:
            ego_pose.append(img_meta['ego_pose'])
            timestamp.append(img_meta['timestamp'])
        ego_pose = np.asarray(ego_pose)
        ego_pose = torch.from_numpy(ego_pose).to(torch.float32)
        timestamp = np.asarray(timestamp)
        timestamp = torch.from_numpy(timestamp)

        return ego_pose, timestamp

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
                img2lidars=None,
                ego_pose=None,
                timestamp=None):

        if torch.onnx.is_in_onnx_export():
            outs, out_memory_embedding, out_memory_reference_point, \
                out_memory_timestamp, out_memory_egopose, out_memory_velo = \
                    self.pts_bbox_head(img_feats, self.img_metas, outs_roi,
                                       memory_embedding, memory_reference_point, memory_timestamp,
                                       memory_egopose, memory_velo,
                                       intrinsics, extrinsics, lidar2imgs, img2lidars,
                                       ego_pose, timestamp)
        else:
            outs = self.pts_bbox_head(img_feats, self.img_metas, outs_roi,
                                      memory_embedding, memory_reference_point, memory_timestamp,
                                      memory_egopose, memory_velo,
                                      intrinsics, extrinsics, lidar2imgs, img2lidars,
                                      ego_pose, timestamp)


        bbox_list = self.pts_bbox_head.get_bboxes(outs, self.img_metas)

        if torch.onnx.is_in_onnx_export():
            return bbox_list, out_memory_embedding, out_memory_reference_point, \
                   out_memory_timestamp, out_memory_egopose, out_memory_velo
        else:
            return bbox_list

