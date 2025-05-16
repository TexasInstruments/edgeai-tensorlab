from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from mmdet3d.structures.bbox_3d.utils import get_lidar2img

from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

import torchvision.transforms.functional as tF
import torchvision.transforms._functional_tensor as tF_t


# BEVDet_R50 ONNX exporting model
class BEVDet_export_model(nn.Module):

    def __init__(self, 
                 img_backbone,
                 img_neck,
                 img_view_transformer,
                 img_bev_encoder_backbone,
                 img_bev_encoder_neck,
                 pts_bbox_head):
        super().__init__()

        # Image feature map
        self.img_backbone              = img_backbone
        self.img_neck                  = img_neck

        # View transform
        self.img_view_transformer      = img_view_transformer
        self.grid_size                 = img_view_transformer.grid_size.int()
        self.grid_lower_bound          = img_view_transformer.grid_lower_bound
        self.grid_interval             = img_view_transformer.grid_interval
        self.D                         = img_view_transformer.D             # 59
        self.C                         = img_view_transformer.out_channels  # 64

        # BEV encoder
        self.img_bev_encoder_backbone  = img_bev_encoder_backbone
        self.img_bev_encoder_neck      = img_bev_encoder_neck

        # Detection
        self.pts_bbox_head             = pts_bbox_head

        # Batch to multiple branches
        self.enable_multi_branch = False


    def prepare_sensor_transform(self, img, img_metas):
        sensor2ego, cam2img, lidar2cam, ego2global = [], [], [], []
        post_rts, bda = [], []

        # post_rts accounts for scaling and translation.
        # Therefore we send the original camera intrinsic for view transform
        for i, meta in enumerate(img_metas):
            sensor2ego.append(meta['sensor2ego'])
            cam2img.append(meta['ori_cam2img'])
            lidar2cam.append(meta['lidar2cam'])
            ego2global.append(meta['ego2global'])
            post_rts.append(meta['post_rts'])
            bda.append(meta['bda_mat'])

        sensor2ego = img.new_tensor(np.asarray(sensor2ego))
        cam2img = img.new_tensor(np.asarray(cam2img))
        lidar2cam = img.new_tensor(np.array(lidar2cam))
        ego2global = img.new_tensor(np.asarray(ego2global))
        post_rts = img.new_tensor(np.asarray(post_rts))
        bda = img.new_tensor(np.asarray(bda))

        return [sensor2ego, cam2img, lidar2cam, ego2global, post_rts, bda]


    def prepare_data(self, img, img_metas):
        self.img_metas = img_metas
        
        transforms = self.prepare_sensor_transform(img, img_metas)
        return transforms


    def precompute_geometry(self, transforms):
        """
        Pre-compute the voxel info 
        """
        coor = self.img_view_transformer.get_lidar_coor(*transforms)
        return self.precompute_voxel_info(coor)


    def precompute_voxel_info(self, coor):
        B, N, D, H, W, _ = coor.shape

        num_points = B * N * D * H * W

        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))

        coor = coor.long().reshape(num_points, 3)
        batch_idx = torch.arange(0, B).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])

        if kept.size() == 0:
            return None, None

        # for our BEV pooling - coor in 1D tensor
        num_grids = B*self.grid_size[2]*self.grid_size[1]*self.grid_size[0]
        bev_feat = torch.zeros((num_grids + 1, self.C), device=coor.device)

        coor_1d = torch.zeros(num_points, device=coor.device)
        coor_1d  = coor[:, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0]) + \
                   coor[:, 2] * (self.grid_size[1] * self.grid_size[0]) + \
                   coor[:, 1] *  self.grid_size[0] + coor[:, 0]
        coor_1d[(kept==False).nonzero().squeeze()] = (B * self.grid_size[2] * self.grid_size[1] * self.grid_size[0]).long()

        return bev_feat, coor_1d.long().contiguous()


    # Single image view Transform for TIDL with the pre-computed 1D coor info
    def view_transform_branch_TIDL(self, inputs):
        feat          = inputs[0]
        lidar_coor_1d = inputs[1]
        bev_feat      = inputs[2]

        num_grids = self.grid_size[2]*self.grid_size[1]*self.grid_size[0]
        # accumulate=True is not exported correctly.
        # So set accumulate=False, and modify the onnx model by adding attrs["reduction"]='add' 
        # using the onnx surgery tool
        #bev_feat = bev_feat.index_put_(tuple([lidar_coor_1d]), feat, accumulate=True)
        bev_feat.index_put_(tuple([lidar_coor_1d]), feat, accumulate=False)

        bev_feat = bev_feat[:num_grids, :]
        bev_feat = bev_feat.reshape(1, self.grid_size[2], self.grid_size[1], self.grid_size[0], self.C)
        bev_feat = bev_feat.permute(0, 4, 1, 2, 3)

        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    # Batch view Transform for TIDL with the pre-computed 1D coor info
    def view_transform_TIDL(self, inputs):
        
        B = 1
        x = inputs[0]
        N, C, H, W = x.shape # [1, 6, 256, 16, 44]
        x = x.view(B*N, C, H, W)
        x = self.img_view_transformer.depth_net(x)

        depth = x[:, :self.D].softmax(dim=1)   # depth: [6, 59, 16, 44]
        tran_feat = x[:, self.D:self.D + self.C] # tran_feat: [6, 64, 16, 44]

        feat = depth.unsqueeze(1) * tran_feat.unsqueeze(2)
        # It is not necesasry, but it resolved shape inferencing error while simplifying the model
        # Should use numbers, shouln't use B, N, self.C, self.D
        #feat = feat.view(6, 64, 59, 16, 44)
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = feat.reshape(B*N*self.D*H*W, self.C)

        lidar_coor_1d = inputs[1]
        bev_feat = inputs[2]
                
        num_grids = B*self.grid_size[2]*self.grid_size[1]*self.grid_size[0]
        # accumulate=True is not exported correctly.
        # So set accumulate=False, and modify the onnx model by adding attrs["reduction"]='add' 
        # using the onnx surgery tool
        #bev_feat = bev_feat.index_put_(tuple([lidar_coor_1d]), feat, accumulate=True)
        bev_feat.index_put_(tuple([lidar_coor_1d]), feat, accumulate=False)

        bev_feat = bev_feat[:num_grids, :]
        bev_feat = bev_feat.reshape(B, self.grid_size[2], self.grid_size[1], self.grid_size[0], self.C)
        bev_feat = bev_feat.permute(0, 4, 1, 2, 3)

        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat, depth


    def forward(self, imgs, bev_feat, lidar_coor_1d):

        # Image encoder
        # Assume batch size = 1
        B = 1

        if self.enable_multi_branch is True:
            N = len(imgs)
            _, C, imH, imW = imgs[0].shape
            xv = []
            for i in range(N):
                #x = self.img_backbone(imgs[i].view(1, C, imH, imW))
                x = self.img_backbone(imgs[i])
                x = self.img_neck(x)
                x = self.img_view_transformer.depth_net(x)

                _, C, H, W  = x.shape # [_, 256, 16, 44]

                depth     = x[:, :self.D].softmax(dim=1)                     # depth: [1, 59, 16, 44]
                tran_feat = x[:, self.D:self.D + self.C].permute(1, 0, 2, 3) # tran_feat: [64, 1, 16, 44]
                feat      = depth * tran_feat
                #feat = feat.permute(1, 2, 3, 0)
                # reshape in batch dimension is not supported in TIDL
                #feat = feat.reshape(self.D*H*W, self.C)
                #feat = feat.flatten(start_dim=0, end_dim=2)
                xv.append(feat)

            x = torch.cat((xv[0], xv[1], xv[2], xv[3], xv[4], xv[5]), dim=1)
            x = x.permute(1, 2, 3, 0)

            #x = x.flatten(start_dim=0, end_dim=2)
            x = x.reshape(N*self.D*H*W, self.C)
            x = self.view_transform_branch_TIDL([x, lidar_coor_1d, bev_feat])
        else:
            N, C, H, W = imgs.size()
            x = self.img_backbone(imgs)
            x = self.img_neck(x)

            _, output_dim, ouput_H, output_W = x.shape # x : [6, 256, 16, 44]
            x = x.view(N, output_dim, ouput_H, output_W)
            x, depth = self.view_transform_TIDL([x] + [lidar_coor_1d, bev_feat])

        # BEV encoder
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)

        outs = self.pts_bbox_head([x])  # outs: tuple of length = 1
        bbox_out = self.pts_bbox_head.get_bboxes_onnx(outs, self.img_metas, rescale=False)

        return bbox_out


class DETR3D_export_model(nn.Module):
    def __init__(self,
                 img_backbone,
                 img_neck,
                 pts_bbox_head,
                 add_pred_to_datasample):
        super().__init__()
        self.img_backbone           = img_backbone
        self.img_neck               = img_neck
        self.pts_bbox_head          = pts_bbox_head
        self.add_pred_to_datasample = add_pred_to_datasample


    def add_lidar2img(self, batch_input_metas: List[Dict]) -> List[Dict]:
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            l2i = list()
            for i in range(len(meta['cam2img'])):
                c2i = torch.tensor(meta['cam2img'][i]).double()
                l2c = torch.tensor(meta['lidar2cam'][i]).double()
                l2i.append(get_lidar2img(c2i, l2c).float().numpy())
            meta['lidar2img'] = l2i

        return batch_input_metas

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
        """

    def prepare_data(self, img, img_metas):
        self.img_metas = img_metas

        input_shape = img.shape[-2:]
        # update real input shae of each single img
        for img_meta in self.img_metas:
            img_meta.update(input_shape=input_shape)


    #def forward(self, img, batch_data_samples):
    def forward(self, img):
        B, N, C, H, W = img.size()

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)

        img_feats = self.img_backbone(img)      
        img_feats = self.img_neck(img_feats)
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        outs = self.pts_bbox_head(img_feats_reshaped, self.img_metas)

        return outs

        # post-processing
        '''        
        results_list_3d = self.pts_bbox_head.predict_by_feat(
            outs, self.img_metas)
        return results_list_3d

        detsamples = self.add_pred_to_datasample(
            batch_data_samples, results_list_3d)

        return detsamples
        '''


class BEVFormer_export_model(nn.Module):
    def __init__(self,
                 img_backbone,
                 img_neck,
                 pts_bbox_head,
                 add_pred_to_datasample,
                 video_test_mode):
        super().__init__()
        # Somehow, the model and input should be loaded on cpu to export BEVFormer
        # To run the model again after exporing, this model should be on gpu.
        # So we deepcopy each sub network for model export
        self.img_backbone           = img_backbone
        self.img_neck               = img_neck
        self.pts_bbox_head          = pts_bbox_head
        self.add_pred_to_datasample = add_pred_to_datasample
        self.fp16_enabled           = False
        self.video_test_mode        = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.img_metas = None

        self.bev_h = self.pts_bbox_head.bev_h
        self.bev_w = self.pts_bbox_head.bev_w
        self.pc_range = self.pts_bbox_head.transformer.encoder.pc_range
        self.num_points_in_pillar = self.pts_bbox_head.transformer.encoder.num_points_in_pillar
        self.bev_embedding = self.pts_bbox_head.bev_embedding

        self.real_h = self.pts_bbox_head.real_h
        self.real_w = self.pts_bbox_head.real_w

        self.rotate_prev_bev = self.pts_bbox_head.transformer.rotate_prev_bev
        self.rotate_center = self.pts_bbox_head.transformer.rotate_center


    def prepare_data(self, img_metas):
        self.img_metas = img_metas


    def get_reference_points(self, H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


    def point_sampling(self, reference_points, pc_range,  img_metas):
        # NOTE: close tf32 here.
        #allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        #torch.backends.cuda.matmul.allow_tf32 = False
        #torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3) # 4x1x2500x4
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        #torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        #torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask


    def precompute_bev_info(self, img_metas):
        """
        Pre-compute the voxel info 
        """
        bev_query = self.bev_embedding.weight.to(torch.float32)
        bev_query = bev_query.unsqueeze(1)

        ref_3d = self.get_reference_points(
            self.bev_h, self.bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar,
            dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        #ref_2d = self.get_reference_points(
        #    self.bev_h, self.bev_w, dim='2d', bs=bev_query.size(1),
        #    device=bev_query.device, dtype=bev_query.dtype)
        
        # Get image coors corresponding to ref_3d. bev_mask indicates valid coors
        # bev_mask: 6x1x2500x4
        # reference_points_cam: 6x1x2500x4x2
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, self.img_metas)

        # bev_valid_indices has valid BEV grid indices. Invalid BEV grids is bev_h*bev_w
        # bev_valid_indices_count has # of valid BEV grids
        bev_valid_indices = []
        bev_valid_indices_count = []
        for mask_per_img in bev_mask:
            nzindex = mask_per_img[0].sum(-1).nonzero().squeeze(-1).to(torch.int32)
            index_query_per_img = nzindex.new_ones(self.bev_h * self.bev_w)*self.bev_h * self.bev_w
            index_query_per_img[:len(nzindex)] = nzindex
            bev_valid_indices.append(index_query_per_img)
            bev_valid_indices_count.append(len(nzindex))

        # Get bev_mask_count from bev_mask for encoder spatial_cross_attention
        bev_mask_count = bev_mask.sum(-1) > 0
        bev_mask_count = bev_mask_count.permute(1, 2, 0).sum(-1)
        bev_mask_count = torch.clamp(bev_mask_count, min=1.0)
        bev_mask_count = bev_mask_count[..., None]

        can_bus = bev_query.new_tensor([each['can_bus'] for each in self.img_metas])

        delta_x = np.array([each['can_bus'][0] for each in self.img_metas])
        delta_y = np.array([each['can_bus'][1] for each in self.img_metas])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in self.img_metas])
        grid_length_y = self.real_h / self.bev_h
        grid_length_x = self.real_w / self.bev_w
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w

        shift_xy =  torch.tensor([[shift_x[0],shift_y[0]]]).to(torch.float32)

        #return ref_3d, ref_2d, reference_points_cam, bev_mask, torch.tensor([shift_y[0],shift_x[0]]), can_bus
        return reference_points_cam, bev_mask_count, torch.cat(bev_valid_indices, dim=0), \
            torch.Tensor(bev_valid_indices_count).to(torch.int64), shift_xy, can_bus


    def compute_rotation_matrix(self, prev_bev, img_metas):
        height = self.bev_h
        width  = self.bev_w
        oh = height
        ow = width
        dtype = prev_bev.dtype if torch.is_floating_point(prev_bev) else torch.float32

        center_f = [0.0, 0.0]
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(self.rotate_center, [width, height])]

        angle = img_metas[0]['can_bus'][-1]
        matrix = tF._get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

        theta = torch.tensor(matrix, dtype=dtype, device=prev_bev.device).reshape(1, 2, 3)
        grid = tF_t._gen_affine_grid(theta, w=width, h=height, ow=ow, oh=oh)

        return grid


    def forward(self, img, shift_xy=None, rotation_grid=None, \
        reference_points_cam=None, bev_mask_count=None, bev_valid_indices=None, \
        bev_valid_indices_count=None,  can_bus=None, prev_bev=None):
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        # to check len_queue values
        len_queue = None

        img_feats_reshaped = []
        for img_feat in img_feats:
            B = 1
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        outs = self.pts_bbox_head(img_feats_reshaped, self.img_metas, prev_bev=prev_bev,
                                  rotation_grid=rotation_grid,
                                  reference_points_cam=reference_points_cam,
                                  bev_mask_count=bev_mask_count,
                                  bev_valid_indices=bev_valid_indices,
                                  bev_valid_indices_count=bev_valid_indices_count,
                                  shift_xy=shift_xy, can_bus=can_bus)
        self.prev_frame_info['prev_bev'] = outs['bev_embed']  # outs['bev_embed']: 2500x1x256

        bbox_list = self.pts_bbox_head.get_bboxes_onnx(
            outs, self.img_metas, rescale=False)

        return bbox_list, outs['bev_embed']


class FCOS3D_export_model(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 add_pred_to_datasample):
        super().__init__()
        self.backbone   = backbone.convert(make_copy=True) if hasattr(backbone, "convert") else backbone
        self.neck       = neck.convert(make_copy=True) if hasattr(neck, "convert") else neck
        if hasattr(bbox_head, "new_bbox_head"):
            self.bbox_head  = copy.deepcopy(bbox_head)
            # self.bbox_head.new_bbox_head loses the convert function after deepcopy so using the original
            setattr(self.bbox_head, "new_bbox_head", bbox_head.new_bbox_head.convert(make_copy=True))
            self.bbox_head.cpu()
        elif hasattr(backbone, "convert"): # bbox_head is not quantized but rest of the network is quantized
            self.bbox_head  = copy.deepcopy(bbox_head).cpu()
        else:
            self.bbox_head = bbox_head
        self.add_pred_to_datasample = add_pred_to_datasample

    def prepare_data(self, batch_img_metas):
        self.batch_img_metas = batch_img_metas


    def forward(self, img, pad_cam2img, inv_pad_cam2img):
        x = self.backbone(img)
        x = self.neck(x)

        outs = self.bbox_head(x)
        #return outs

        predictions = self.bbox_head.predict_by_feat_onnx(
            *outs, batch_img_metas=self.batch_img_metas, rescale=True, 
            pad_cam2img=pad_cam2img, inv_pad_cam2img=inv_pad_cam2img)

        return predictions