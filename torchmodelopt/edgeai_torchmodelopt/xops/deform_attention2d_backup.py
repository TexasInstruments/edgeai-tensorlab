#################################################################################
# Copyright (c) 2018-2024, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleFixedSizeDeformAttnOnnx(nn.Module):
    """This is a single-layer version of MultiScaleDeformAttn, mainly for ONNX export.

       Assuming single layer can export a clean onnx model.
       This may not be needed any more as the implementation in deform_attention2d.py
         can handle both variable size and fixed size.
    """
    def __init__(self, value_spatial_shapes_list=None, mode='bilinear'):
        super().__init__(value_spatial_shapes_list, mode)
        self.mode = mode
        self.value_spatial_shapes_list = value_spatial_shapes_list
        if value_spatial_shapes_list is None:
            assert False, 'value_spatial_shapes_list must be provided'

    # Based on mmcv.ops.mutli_scale_deformable_attn_pytorch
    def deformable_attn_pytorch(self,
                                value,
                                value_spatial_shapes,
                                sampling_locations,
                                attention_weights):
        """CPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ =\
            sampling_locations.shape

        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []

        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        _value_spatial_shapes = self.value_spatial_shapes_list[0] if self.value_spatial_shapes_list is not None \
            else value_spatial_shapes
        value_l_ = value.flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, _value_spatial_shapes[0], _value_spatial_shapes[1])
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          0].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode=self.mode,
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)

        # (bs, num_queries, num_heads, num_levels, num_points) ->
        # (bs, num_heads, num_queries, num_levels, num_points) ->
        # (bs, num_heads, 1, num_queries, num_levels*num_points)
        attention_weights = attention_weights.transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_levels * num_points)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
                  attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                                  num_queries)
        return output.transpose(1, 2).contiguous()

    def forward(self,
                value,
                value_spatial_shapes,
                sampling_locations,
                attention_weights):
        output = self.deformable_attn_pytorch(
                    value, value_spatial_shapes, sampling_locations, attention_weights)
        return output
