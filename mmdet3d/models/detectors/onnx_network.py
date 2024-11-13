from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.structures.bbox_3d.utils import get_lidar2img

from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

import torchvision.transforms.functional as tF
import torchvision.transforms._functional_tensor as tF_t

from mmseg.models.utils import resize


class PETR_export_model(nn.Module):
    def __init__(self,
                 img_backbone,
                 img_neck,
                 pts_bbox_head):
        super().__init__()

        self.img_backbone   = img_backbone
        self.img_neck       = img_neck
        self.pts_bbox_head  = pts_bbox_head

        # for camera frustum creation
        # Image feature size after image backbone. 
        # It may need update for different image backbone
        self.B              = 1
        self.N              = 6
        self.C              = 256
        self.H              = 20
        self.W              = 50

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
        self.enable_multi_branch = True


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
            _, N, C, imH, imW = imgs.shape

            imgs = imgs.view(B * N, C, imH, imW)
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
        self.img_backbone           = copy.deepcopy(img_backbone)
        self.img_neck               = copy.deepcopy(img_neck)
        self.pts_bbox_head          = copy.deepcopy(pts_bbox_head)
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
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, self.img_metas)

        bev_valid_indices = []
        bev_valid_indices_count = []
        for mask_per_img in bev_mask:
            nzindex = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
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
        return reference_points_cam, bev_mask_count, torch.stack(bev_valid_indices, dim=0), \
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
        self.backbone   = backbone
        self.neck       = neck
        self.bbox_head  = bbox_head
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

"""
@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points
"""

def backproject_tidl(features, xy_coor, n_voxels):
    """
    function: 2d feature + predefined point cloud -> 3d volume
    """
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = n_voxels

    features = features.permute(0, 2, 3, 1)
    features = features.reshape(-1, n_channels)
    features = F.pad(features,(0,0,0,1),"constant",0)

    #volume   = torch.index_select(features, 0, xy_coor.to(features.device))
    volume = features[xy_coor]

    volume   = volume.permute(1,0)
    #volume   = volume.view(1, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume


class FastBEV_export_model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone        = model.backbone
        self.neck            = model.neck
        self.neck_fuse_0     = model.neck_fuse_0
        self.neck_3d         = model.neck_3d
        self.bbox_head       = model.bbox_head

        self.multi_scale_id  = model.multi_scale_id
        self.n_voxels        = model.n_voxels
        self.backproject     = model.backproject
        self.style           = model.style
        self.extrinsic_noise = model.extrinsic_noise
        self.voxel_size      = model.voxel_size

        self._compute_projection  = model._compute_projection
        self.get_temporal_feats   = model.get_temporal_feats
        self.precompute_proj_info_for_inference = model.precompute_proj_info_for_inference

        self.num_temporal_feats = model.num_temporal_feats
        self.feats_size         = model.feats_size

        self.memory          = model.memory
        self.queue           = model.queue


    def prepare_data(self, img_metas):
        self.img_metas = img_metas


    def extract_feat(self, img):
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        x = self.backbone(
            img
        )  # [6, 256, 232, 400]; [6, 512, 116, 200]; [6, 1024, 58, 100]; [6, 2048, 29, 50]

        mlvl_feats = self.neck(x)
        mlvl_feats = list(mlvl_feats)

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i],
                            size=mlvl_feats[msid].size()[2:],
                            mode="bilinear",
                            align_corners=False)
                        fuse_feats.append(resized_feat)

                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_

        # v3 bev ms
        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        return mlvl_feats

    def extract_feat_neck3d(self, img, img_metas, mlvl_feats, xy_coors):
        batch_size = img.shape[0]

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):
            # to reduce redundant operator
            if batch_size == 1:
                mlvl_feat_split = torch.split(mlvl_feat, 6, dim=0)
            else:
                # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
                mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
                # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
                mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id, mlvl_feat_i in enumerate(mlvl_feat_split):
                volumes = []
                
                for batch_id, seq_img_meta in enumerate(img_metas):
                    if batch_size == 1:
                        feat_i = mlvl_feat_i
                    else:
                        feat_i = mlvl_feat_i[batch_id]  # [nv, c, h, w]

                    volume = backproject_tidl(
                        feat_i, xy_coors[seq_id], self.n_voxels[0])  # [c, vx, vy, vz]
                    if batch_size == 1:
                        volume = volume.view([1, feat_i.shape[1]] + self.n_voxels[0])
                    else:
                        volume = volume.view([feat_i.shape[1]] + self.n_voxels[0])
                        volumes.append(volume)

                # to reduce redundant operator
                if batch_size ==1:
                    volume_list.append(volume)
                else:
                    volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])

        mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
        x = self.neck_3d(mlvl_volumes)
        return x


    def forward(self,
                img,
                xy_coors,
                prev_feats_map=None):

        mlvl_feats = self.extract_feat(img)
        if prev_feats_map is not None:
            mlvl_feats_all = [torch.cat((mlvl_feats[0], prev_feats_map), dim=0)]
        else:
            mlvl_feats_all = mlvl_feats

        feature_bev = self.extract_feat_neck3d(img, self.img_metas, mlvl_feats_all, xy_coors)
        x = self.bbox_head(feature_bev)

        bbox_list = self.bbox_head.get_bboxes(*x, self.img_metas, valid=None)
        return mlvl_feats, bbox_list
