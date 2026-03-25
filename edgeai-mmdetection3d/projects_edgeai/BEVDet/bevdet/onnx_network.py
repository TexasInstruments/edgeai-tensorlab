import torch
import torch.nn as nn
import numpy as np


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


