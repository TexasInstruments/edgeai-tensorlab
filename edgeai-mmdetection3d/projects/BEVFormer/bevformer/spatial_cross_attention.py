
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model import xavier_init, constant_init
from mmdet3d.registry import MODELS


from mmcv.cnn.bricks.transformer import build_attention
from mmcv.utils import ext_loader


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@MODELS.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                bev_mask_count=None,
                bev_valid_indices=None,
                bev_valid_indices_count=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        D = reference_points_cam.size(3)

        # Add zero query
        query = torch.cat((query, query.new_zeros(bs, 1, self.embed_dims)), dim=1)
        reference_points_cam = torch.cat((reference_points_cam, reference_points_cam.new_zeros(self.num_cams, bs, 1, D, 2)), dim=2)

        if bev_valid_indices is None:
            indexes = []
            indexes_count = []
            for i, mask_per_img in enumerate(bev_mask):
                #index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                nzindex = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                index_query_per_img = nzindex.new_ones(num_query)*num_query
                index_query_per_img[:len(nzindex)] = nzindex

                indexes.append(index_query_per_img.unsqueeze(1))
                indexes_count.append(len(nzindex))
            # Need to create bev_valid_indices for later use
            bev_valid_indices = torch.cat(indexes)
        else:
            # bev_valid_indecse : [150000x1]
            # indexes : list of six [2500x1] tensors
            indexes = list(torch.split(bev_valid_indices, num_query, dim=0))
            indexes_count = bev_valid_indices_count
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        
        # Notes: bs == 1 is for batch size = 1 in order to simpliy the exported ONNX model
        if bs == 1:
            #queries_rebatch = query.new_zeros(
            #    [self.num_cams, max_len, self.embed_dims])
            #reference_points_rebatch = reference_points_cam.new_zeros(
            #    [self.num_cams, max_len, D, 2])
            query = query.squeeze(0)

            if False:
                queries_rebatch = query.new_zeros(
                    [self.num_cams*max_len, self.embed_dims])
                reference_points_rebatch = reference_points_cam.new_zeros(
                    [self.num_cams*max_len, D, 2])

                for i, reference_points_per_img in enumerate(reference_points_cam):
                    index_query_per_img = indexes[i]
                    #queries_rebatch[i, :len(index_query_per_img)] = query[index_query_per_img]
                    #reference_points_rebatch[i, :len(index_query_per_img)] = reference_points_per_img.squeeze(0)[index_query_per_img]
                    queries_rebatch[i*max_len:i*max_len + max_len] = query[index_query_per_img]
                    reference_points_rebatch[i*max_len:i*max_len + max_len] = reference_points_per_img.squeeze(0)[index_query_per_img]
            else:
                # Replace ScatterND with Concat
                for i, reference_points_per_img in enumerate(reference_points_cam):
                    index_query_per_img = indexes[i].squeeze(1)
                    if i == 0:
                        queries_rebatch = query[index_query_per_img]
                        reference_points_rebatch = reference_points_per_img.squeeze(0)[index_query_per_img]
                    else:
                        queries_rebatch = torch.cat((queries_rebatch, query[index_query_per_img]), dim=0)
                        reference_points_rebatch = torch.cat((reference_points_rebatch, reference_points_per_img.squeeze(0)[index_query_per_img]), dim=0)

            queries_rebatch = queries_rebatch.reshape(self.num_cams, max_len, self.embed_dims)
            reference_points_rebatch = reference_points_rebatch.reshape(self.num_cams,max_len, D, 2)
        else:
            queries_rebatch = query.new_zeros(
                [bs, self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs, self.num_cams, max_len, D, 2])

            for j in range(bs):
                for i, reference_points_per_img in enumerate(reference_points_cam):
                    index_query_per_img = indexes[i].squeeze(1)
                    queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        if bs == 1:
            queries = self.deformable_attention(query=queries_rebatch, key=key, value=value,
                                                reference_points=reference_points_rebatch, spatial_shapes=spatial_shapes,
                                                level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)

            # Note: The following codes could result in the different outcome when compared to original code
            #       since there are redundant query indices (i.e. 0), and the last query is picked up
            #       Because it is finally averaged using count, the difference could be negligible
            slots = slots.squeeze(0)
            queries = queries.squeeze(0)

            # To remove indices - Add one more (2501-th) tensor
            slots = torch.cat((slots,  slots.new_zeros(1, self.embed_dims)), dim = 0)

            # Just reshape bev_valid_indices, instead concat squeezed indexes again
            # bev_valid_indices should be type cated to int64 for ScatterND
            all_indices = bev_valid_indices.to(torch.int64).squeeze(1)
            all_queries = queries.view(-1, self.embed_dims)
            """
            for i, index_query_per_img in enumerate(indexes):
                if i == 0:
                    all_queries = queries[i]
                else:
                    all_queries = torch.cat((all_queries, queries[i]), dim = 0)
            """

            slots.index_put_(tuple([all_indices]), all_queries, accumulate=True)
            slots = slots[:-1].unsqueeze(0)
            """
            for i, index_query_per_img in enumerate(indexes):
                # Note: When len(index_query_per_img) == 2500 (i.e. max query num),
                #       we don't need slicing like  queries[i, :len(index_query_per_img)]
                #slots[index_query_per_img] += queries[i, :len(index_query_per_img)]
                if i == 0:
                    all_indices = index_query_per_img[:indexes_count[i]]
                    all_queries = queries[i, :indexes_count[i]]
                else:
                    all_indices = torch.cat((all_indices, index_query_per_img[:indexes_count[i]]), dim = 0)
                    all_queries = torch.cat((all_queries, queries[i, :indexes_count[i]]), dim = 0)

            slots.index_put_(tuple([all_indices]), all_queries, accumulate=True)
            slots = slots.unsqueeze(0)
            """
        else:
            queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                                reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                                level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)

            for j in range(bs):
                for i, index_query_per_img in enumerate(indexes):
                    index_query_per_img = index_query_per_img.squeeze(1)
                    slots[j, index_query_per_img[:indexes_count[i]]] += queries[j, i, :indexes_count[i]]

        if bev_mask_count is None:
            count = bev_mask.sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots = slots / count[..., None]
        else:
            slots = slots / bev_mask_count
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@MODELS.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        self.sampling_offsets = self.sampling_offsets.to(self.sampling_offsets.weight.device)

        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape # [N, 50x50, 256]
        bs, num_value, _ = value.shape # [N, 375, 256]
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)   # [N, 375, 8, 32]
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2) # [N, 50x50, 8, 1, 8, 2]
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)  # [N, 50x50, 8, 8]

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)  # [N, 50x50, 8, 1, 8]

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            """
            # 7D tensor operation
            bs, num_query, num_Z_anchors, xy = reference_points.shape  # [6, 50x50, 4 ,2]
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
            """
            # 6D tensor operation
            bs, num_query, num_Z_anchors, xy = reference_points.shape  # [6, 50x50, 4 ,2]
            reference_points = reference_points[:, :, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels * (num_all_points // num_Z_anchors), num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == (num_levels_points // num_levels) * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
