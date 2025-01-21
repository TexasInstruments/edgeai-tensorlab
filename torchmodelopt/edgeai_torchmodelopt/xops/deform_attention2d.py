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


class MultiScaleDeformAttn(nn.Module):
    def __init__(self, value_spatial_shapes_list=None, mode='bilinear', verbose=True):
        '''
        If value_spatial_shapes_list here in the constructor,
        value_spatial_shapes provided in the forward call will be ignored.
        Providing here is recommended for ONNX export only.
        '''
        super().__init__()
        self.mode = mode
        self.value_spatial_shapes_list = value_spatial_shapes_list
        if verbose and value_spatial_shapes_list is not None:
            print('value_spatial_shapes_list has been provided in constructor. '
                  '\nThe value provided in forward call will be ignored')

    def forward(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        output = self.multi_scale_deformable_attn(
                    value, value_spatial_shapes, sampling_locations, attention_weights, self.mode)
        return output

    # Based on mmcv.ops.mutli_scale_deformable_attn_pytorch
    def multi_scale_deformable_attn(self, value, value_spatial_shapes, sampling_locations, attention_weights, mode):
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

        _value_spatial_shapes = self.value_spatial_shapes_list if self.value_spatial_shapes_list is not None \
            else value_spatial_shapes

        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ =\
            sampling_locations.shape
        value_list = value.split([int(H_) * int(W_) for H_, W_ in _value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (H_, W_) in enumerate(_value_spatial_shapes):
            H_ = int(H_)
            W_ = int(W_)
            # bs, H_*W_, num_heads, embed_dims ->
            # bs, H_*W_, num_heads*embed_dims ->
            # bs, num_heads*embed_dims, H_*W_ ->
            # bs*num_heads, embed_dims, H_, W_
            value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H_, W_)
            # bs, num_queries, num_heads, num_points, 2 ->
            # bs, num_heads, num_queries, num_points, 2 ->
            # bs*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = sampling_grids[:, :, :,
                                              level].transpose(1, 2).flatten(0, 1)
            # bs*num_heads, embed_dims, num_queries, num_points
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode=mode,
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


class SingleScaleDeformAttn(MultiScaleDeformAttn):
    def __init__(self, value_spatial_shapes_list=None, mode='bilinear', verbose=False):
        super().__init__(value_spatial_shapes_list, mode, verbose=verbose)
        if value_spatial_shapes_list is not None:
            if len(value_spatial_shapes_list) != 1:
                raise RuntimeError('value_spatial_shapes_list if provided, must have only a single shape/scale')

    def forward(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        if value_spatial_shapes is not None:
            if value_spatial_shapes.size(0) != 1:
                raise RuntimeError('value_spatial_shapes_list if provided, must have only a single shape/scale')

        return super().forward(value, value_spatial_shapes, sampling_locations, attention_weights)


###########################################################################################
# unit tests
###########################################################################################

def run_test_deform_attn():
    # Set test params
    B = 1
    NUM_LEVELS = 1
    FEATURE_HEIGHT = 50
    FEATURE_WIDTH = 50
    NUM_QUERIES = FEATURE_HEIGHT * FEATURE_WIDTH
    NUM_HEADS = 8
    EMBED_DIMS = 256
    NUM_POINTS = 4
    INTP_MODE = 'bilinear'

    # Input to the model:
    # value, value_spatial_shape, sampling_location, attention_weight
    value = torch.randn(2*B, NUM_QUERIES, NUM_HEADS, EMBED_DIMS // NUM_HEADS)

    # value_spatial_shapes = torch.zeros(1,2).to(torch.int64)
    # value_spatial_shapes[:, 0] = FEATURE_HEIGHT
    # value_spatial_shapes[:, 1] = FEATURE_WIDTH
    value_spatial_shapes_list = [[FEATURE_HEIGHT, FEATURE_WIDTH]]
    value_spatial_shapes = torch.tensor(value_spatial_shapes_list).to(torch.int64)

    sampling_locations = torch.rand(
        2*B, NUM_QUERIES, NUM_HEADS, NUM_LEVELS, NUM_POINTS, 2)

    attention_weights = torch.randn(
        2*B, NUM_QUERIES, NUM_HEADS, NUM_LEVELS, NUM_POINTS)
    attention_weights = attention_weights.softmax(-1)

    # Run deform_attn
    deform_attn_ss = SingleScaleDeformAttn(mode=INTP_MODE)
    deform_attn_ss.eval()
    out_deform_attn = deform_attn_ss(value, value_spatial_shapes, sampling_locations, attention_weights)

    # Export FixedSizeDeformAttn
    input_names  = ["value", "value_shape", "samp_loc", "attn_weight"]
    output_names = ["output"]
    model_input = (
        value,
        value_spatial_shapes,  # this will be ignored, since shape was provided through constructor
        sampling_locations,
        attention_weights)
    torch.onnx.export(deform_attn_ss,
                      model_input,
                      "deform_attn_ss.onnx",
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16)

    # simplify the onnx model
    from onnxsim import simplify
    import onnx
    onnx_model, simplify_ok = simplify("deform_attn_ss.onnx")
    onnx.save(onnx_model, "deform_attn_ss.onnx")


if __name__ == "__main__":
    run_test_deform_attn()
