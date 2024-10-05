from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.structures.bbox_3d.utils import get_lidar2img

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
        bbox_out  = self.pts_bbox_head.get_bboxes_onnx(outs, self.img_metas, rescale=False)

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


    def prepare_data(self, img, img_metas):
        self.img_metas = img_metas

    def forward(self, img, prev_bev):
        B, N, C, H, W = img.size()

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        # to check len_queue values
        len_queue = None

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        outs = self.pts_bbox_head(img_feats_reshaped, self.img_metas, prev_bev=prev_bev)
        
        self.prev_frame_info['prev_bev'] = outs['bev_embed']
        #return outs

        bbox_list = self.pts_bbox_head.get_bboxes_onnx(
            outs, self.img_metas, rescale=False)

        return bbox_list
