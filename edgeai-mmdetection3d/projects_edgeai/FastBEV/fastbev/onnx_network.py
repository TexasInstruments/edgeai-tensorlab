from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from mmseg.models.utils import resize

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
        self.backbone    = model.backbone.convert(make_copy=True) if hasattr(model.backbone, "convert") else model.backbone
        self.neck        = model.neck.convert(make_copy=True) if hasattr(model.neck, "convert") else model.neck
        self.neck_fuse_0 = model.neck_fuse_0.convert(make_copy=True) if hasattr(model.neck_fuse_0, "convert") else model.neck_fuse_0
        self.neck_3d     = model.neck_3d.convert(make_copy=True) if hasattr(model.neck_3d, "convert") else model.neck_3d
        if hasattr(model.bbox_head, "new_bbox_head"):
            self.bbox_head  = copy.deepcopy(model.bbox_head)
            # self.bbox_head.new_bbox_head loses the convert function after deepcopy so using the original
            setattr(self.bbox_head, "new_bbox_head", model.bbox_head.new_bbox_head.convert(make_copy=True))
            self.bbox_head.cpu()
        elif hasattr(model.bbox_head, "convert"): # bbox_head is not quantized but rest of the network is quantized
            self.bbox_head  = copy.deepcopy(model.bbox_head).cpu()
        else:
            self.bbox_head = model.bbox_head

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

        if self.num_temporal_feats > 0:
            self.memory          = model.memory
            self.queue           = model.queue


    def prepare_data(self, img_metas):
        self.img_metas = img_metas


    def extract_feat(self, img):
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
        batch_size = img.shape[0] // 6
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

                    if len(mlvl_feat_split) > 1:
                        volume = backproject_tidl(
                            feat_i, xy_coors[seq_id], self.n_voxels[0])  # [c, vx, vy, vz]
                    else:
                        volume = backproject_tidl(
                            feat_i, xy_coors, self.n_voxels[0])  # [c, vx, vy, vz]
                    
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
        if prev_feats_map is None:
            mlvl_feats_all = mlvl_feats
        else:
            concat_mlvl_feats = torch.cat((mlvl_feats[0], mlvl_feats[0]), dim=1)
            mlvl_feats_all    = [torch.cat((concat_mlvl_feats[:, 0:64], prev_feats_map), dim=0)]

        feature_bev = self.extract_feat_neck3d(img, self.img_metas, mlvl_feats_all, xy_coors)
        x = self.bbox_head(feature_bev)

        bbox_list = self.bbox_head.get_bboxes(*x, self.img_metas, valid=None)

        if prev_feats_map is None:
            return bbox_list
        else:
            return bbox_list, concat_mlvl_feats[:, 64:128]