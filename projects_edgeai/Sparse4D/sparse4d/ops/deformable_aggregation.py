import torch
from torch.autograd.function import Function, once_differentiable

from mmengine.model.base_module import BaseModule

try:
    from . import deformable_aggregation_ext
    has_deformable_aggregation_ext = True
except:
    has_deformable_aggregation_ext = False
import torch
import torch.nn.functional as F

def bilinear_sampling(feature_map:torch.Tensor, sampling_locations:torch.Tensor):
    """
    PyTorch implementation of bilinear sampling
    
    Args:
        feature_map: tensor of shape [batch_size, channels, height, width]
        sampling_locations: tensor of shape [batch_size, n_points, 2] with values in range [0, height/width]
                           where each point is (x, y) in pixel coordinates
    
    Returns:
        sampled_features: tensor of shape [batch_size, channels, n_points]
    """
    batch_size, channels, height, width = feature_map.shape
    n_points = sampling_locations.shape[1]
    
    # Convert sampling locations from [0, height/width] to [-1, 1] format required by grid_sample
    normalized_locations = sampling_locations.clone()
    normalized_locations[:, :, 0] = 2.0 * normalized_locations[:, :, 0] / (width - 1) - 1.0  # x coordinate
    normalized_locations[:, :, 1] = 2.0 * normalized_locations[:, :, 1] / (height - 1) - 1.0  # y coordinate
    
    # Reshape grid for grid_sample
    grid = normalized_locations.unsqueeze(1)  # [batch_size, 1, n_points, 2]
    
    # Perform sampling
    sampled = F.grid_sample(
        feature_map, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    # Output shape: [batch_size, channels, 1, n_points]
    return sampled.squeeze(2)  # [batch_size, channels, n_points]

class DeformableAggregationFunction(BaseModule):
    def forward(self,mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights):
        if has_deformable_aggregation_ext:
            return DeformableAggregationFunction1.apply(mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights)
        
        batch_size, num_cams, num_feat, num_embeds = mc_ms_feat.shape
        num_scale = spatial_shape.shape[0]
        num_pts = sample_location.shape[1]
        num_groups = weights.shape[-1]
        group_size = num_embeds // num_groups
        
        # Initialize output tensor
        output = torch.zeros(batch_size, num_pts, num_embeds, device=mc_ms_feat.device)
        
        # Create a mask for valid sampling locations (in range (0,1))
        valid_mask = ((sample_location > 0) & (sample_location < 1)).all(dim=-1)  # [batch, pts, cams]
        
        # Process each camera
        for c in range(num_cams):
            # Skip if no valid points for this camera
            if not valid_mask[:, :, c].any():
                continue
            
            # Get locations for this camera
            cam_locations = sample_location[:, :, c]  # [batch, pts, 2]
            cam_valid_mask = valid_mask[:, :, c]  # [batch, pts]
            
            # Process each scale
            for s in range(num_scale):
                h, w = spatial_shape[s]
                scale_start = scale_start_index[s]
                scale_end = scale_start_index[s+1] if s < num_scale-1 else num_feat
                
                # Get features for this camera and scale
                scale_features = mc_ms_feat[:, c, scale_start:scale_end]
                
                # Reshape if we have multiple feature layers for this scale
                feat_layers = scale_end - scale_start
                if feat_layers > 1:
                    # If multiple layers, we need to handle them separately
                    for layer in range(feat_layers):
                        layer_features = scale_features[:, layer].reshape(batch_size, num_embeds, h, w)
                        
                        # Convert to pixel coordinates
                        pixel_loc = torch.zeros_like(cam_locations)
                        pixel_loc[:, :, 0] = cam_locations[:, :, 0] * w - 0.5  # w_im
                        pixel_loc[:, :, 1] = cam_locations[:, :, 1] * h - 0.5  # h_im
                        # Sample features using bilinear interpolation
                        sampled = bilinear_sampling(layer_features, pixel_loc)  # [batch, embeds, pts]
                        
                        # Apply weights for each group
                        for g in range(num_groups):
                            start_ch = g * group_size
                            end_ch = (g + 1) * group_size
                            
                            # Get weights for this camera, scale, group
                            group_weights = weights[:, :, c, s, g].unsqueeze(1)  # [batch, 1, pts]
                            
                            # Apply weights and mask
                            mask = cam_valid_mask.unsqueeze(1).float()  # [batch, 1, pts]
                            weighted = sampled[:, start_ch:end_ch] * group_weights * mask
                            
                            # Transpose and add to output
                            output[:, :, start_ch:end_ch] += weighted.transpose(1, 2)
                else:
                    # Single feature layer for this scale
                    layer_features = scale_features.reshape(batch_size, num_embeds, h, w)
                    
                    # Convert to pixel coordinates
                    pixel_loc = torch.zeros_like(cam_locations)
                    pixel_loc[:, :, 0] = cam_locations[:, :, 0] * w - 0.5  # w_im
                    pixel_loc[:, :, 1] = cam_locations[:, :, 1] * h - 0.5  # h_im
                    
                    # Sample features using bilinear interpolation
                    sampled = bilinear_sampling(layer_features, pixel_loc)  # [batch, embeds, pts]
                    
                    # Apply weights for each group
                    for g in range(num_groups):
                        start_ch = g * group_size
                        end_ch = (g + 1) * group_size
                        
                        # Get weights for this camera, scale, group
                        group_weights = weights[:, :, c, s, g].unsqueeze(1)  # [batch, 1, pts]
                        
                        # Apply weights and mask
                        mask = cam_valid_mask.unsqueeze(1).float()  # [batch, 1, pts]
                        weighted = sampled[:, start_ch:end_ch] * group_weights * mask
                        
                        # Transpose and add to output
                        output[:, :, start_ch:end_ch] += weighted.transpose(1, 2)
        
        return output
    
    def feature_maps_format(self,feature_maps, inverse=False):
        return DeformableAggregationFunction1.feature_maps_format(feature_maps, inverse)


class DeformableAggregationFunction1(Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):

        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )

    @staticmethod
    def feature_maps_format(feature_maps, inverse=False):
        bs, num_cams = feature_maps[0].shape[:2]
        if not inverse:
            spatial_shape = []
            scale_start_index = [0]

            col_feats = []
            for i, feat in enumerate(feature_maps):
                spatial_shape.append(feat.shape[-2:])
                scale_start_index.append(
                    feat.shape[-1] * feat.shape[-2] + scale_start_index[-1]
                )
                col_feats.append(torch.reshape(
                    feat, (bs, num_cams, feat.shape[2], -1)
                ))
            scale_start_index.pop()
            col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2)
            feature_maps = [
                col_feats,
                torch.tensor(
                    spatial_shape,
                    dtype=torch.int64,
                    device=col_feats.device,
                ),
                torch.tensor(
                    scale_start_index,
                    dtype=torch.int64,
                    device=col_feats.device,
                ),
            ]
        else:
            spatial_shape = feature_maps[1].int()
            split_size = (spatial_shape[:, 0] * spatial_shape[:, 1]).tolist()
            feature_maps = feature_maps[0].permute(0, 1, 3, 2)
            feature_maps = list(torch.split(feature_maps, split_size, dim=-1))
            for i, feat in enumerate(feature_maps):
                feature_maps[i] = feat.reshape(
                    feat.shape[:3] + (spatial_shape[i, 0], spatial_shape[i, 1])
                )
        return feature_maps
