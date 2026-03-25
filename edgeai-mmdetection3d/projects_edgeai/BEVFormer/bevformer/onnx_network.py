import torch
import torch.nn as nn
import numpy as np

from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

import torchvision.transforms.functional as tF
import torchvision.transforms._functional_tensor as tF_t


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

        # clip to [0, 1] to help quantization
        reference_points_cam = torch.clip(reference_points_cam, min=0.0, max=1.0)

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
        return reference_points_cam, bev_mask_count, torch.unsqueeze(torch.cat(bev_valid_indices, dim=0), dim=1), \
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

