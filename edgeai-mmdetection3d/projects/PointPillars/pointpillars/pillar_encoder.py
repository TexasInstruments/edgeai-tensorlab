# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator
from .utils import PPPFNLayer

""" PillarFeatureNet for PointPillars"""

@MODELS.register_module()
class CustomPillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PPPFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PPPFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels: Optional[int] = 4,
                 feat_channels: Optional[tuple] = (64, ),
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 point_color_dim=0,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
                 point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
                                                              40, 1),
                 replace_mat_mul=False,
                 feat_scale_fact=1.0,
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super().__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        in_channels += point_color_dim
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PPPFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    replace_mat_mul = replace_mat_mul,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.pfn_layers.replace_mat_mul = replace_mat_mul
        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.feat_scale_fact   = feat_scale_fact # features are scaled by this value
        self.max_feat = -6535
        self.min_feat = 6535

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                dump_raw_voxel_feat=False) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        #  kind of quantizing the feature vectors
        print_max_min_feat = False

        if print_max_min_feat == True:
            if features.max() > self.max_feat:
                self.max_feat = features.max()
                print (self.max_feat)
                print (self.min_feat)

            if features.min() < self.min_feat:
                self.min_feat = features.min()
                print (self.max_feat)
                print (self.min_feat)

        if self.feat_scale_fact != 1.0 :
            #If user provided scale fact then scale and do clipping and float conversion
            # if user has not provided any value then feat_scale_fact will be 1 and so nothing to be done
            features = features*self.feat_scale_fact
            features = features.to(torch.int32)
            features = features.to(torch.float32)

        if dump_raw_voxel_feat == True:
            f = open("voxel_raw_features.txt",'w')
            features_np = features.cpu().detach().numpy()
            for i, voxel_ft in  enumerate(features_np):
                f.write("voxel no {} \n".format(i))
                for j in range(32):
                    f.write(" {:.2f} {:.2f} {:.2f} ".format(voxel_ft[j][0],voxel_ft[j][1],voxel_ft[j][2]))
                    f.write(" {:.2f} {:.2f} {:.2f} ".format(voxel_ft[j][3],voxel_ft[j][4],voxel_ft[j][5]))
                    f.write(" {:.2f} {:.2f} {:.2f} {:.2f} \n".format(voxel_ft[j][6],voxel_ft[j][7],voxel_ft[j][8], voxel_ft[j][9]))

            f.close()

        if dump_raw_voxel_feat == True:
            inputs = features.permute(2,1,0)
            inputs = inputs.unsqueeze(0)
            x = self.pfn_layers._modules['0'].linear(inputs)
            f = open("mlp.txt",'w')
            x_np = x.cpu().detach().numpy()
            for c, x_ch in  enumerate(x_np):
                for i, x_i in enumerate(x_ch):
                    for j, pt_w in enumerate(x_i):
                        for j, pt in enumerate(pt_w):
                            f.write("{:.2f} ".format(pt))
                        f.write("\n")
            f.close

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze() # changed by TI to squeeze all dimension here, that is what is the expectation asfter this. #features.squeeze(1)


