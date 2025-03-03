import torch
from mmdet3d.registry import TRANSFORMS



@TRANSFORMS.register_module()
class DownsampleQuantizeInstanceDepthmap():
    """
    Given downsample stride, downsample depthmap to (N, H, W), and make Quantization to return onehot format
    Input depthmap in numpy array, Output in torch array.
    """

    def __init__(self, downsample=8, depth_config={}):
        '''
        default: LID style
        '''
        self.downsample = downsample
        self.depth_min, self.depth_max, self.num_bins = [depth_config.get(key) for key in
                                                         ['depth_min', 'depth_max', 'num_depth_bins']]

    def __call__(self, results):
        """
        """
        # depth_maps = torch.from_numpy(np.stack(results['depthmap'])) / 100.0    # (N, Hi, Wi)
        # gt_depths = self._get_downsampled_gt_depth(depth_maps)  # (N, H/downsample, W/downsample)
        # gt_depths_ = self._quantize_gt_depth(gt_depths)          # (N, Hd, Wd, self.D=150)
        # results['depthmap'] = gt_depths_

        gt_boxes2d, gt_center_depth = results['gt_bboxes'], results['depths']   # np array, N*(M,4), N*(M)
        H, W = results['img_shape'][0][:2]
        H, W = int(H / self.downsample), int(W / self.downsample)
        depth_maps, fg_mask = self.build_target_depth_from_3dcenter_argo(gt_boxes2d, gt_center_depth, (H, W))
        depth_target = self.bin_depths(depth_maps, "LID", self.depth_min, self.depth_max, self.num_bins, target=True)
        results['ins_depthmap'] = depth_target  # (N, H, W)
        results['ins_depthmap_mask'] = fg_mask  # (N, H, W)
        return results


    def build_target_depth_from_3dcenter_argo(self, gt_boxes2d, gt_center_depth, HW):
        H, W = HW
        B = len(gt_boxes2d) # B is N indeed
        depth_maps = torch.zeros((B, H, W), dtype=torch.float)
        fg_mask = torch.zeros_like(depth_maps).bool()

        for b in range(B):
            center_depth_per_batch = torch.from_numpy(gt_center_depth[b])
            # Set box corners
            if gt_boxes2d[b].shape[0] > 0:
                gt_boxes_per_batch = torch.from_numpy(gt_boxes2d[b])
                gt_boxes_per_batch = gt_boxes_per_batch / self.downsample   # downsample is necessary
                gt_boxes_per_batch[:, :2] = torch.floor(gt_boxes_per_batch[:, :2])
                gt_boxes_per_batch[:, 2:] = torch.ceil(gt_boxes_per_batch[:, 2:])
                gt_boxes_per_batch = gt_boxes_per_batch.long()

                for n in range(gt_boxes_per_batch.shape[0]):
                    u1, v1, u2, v2 = gt_boxes_per_batch[n]
                    depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]
                    fg_mask[b, v1:v2, u1:u2] = True

        return depth_maps, fg_mask

    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        # elif mode == "SID":
        #     indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
        #               (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)

        return indices
