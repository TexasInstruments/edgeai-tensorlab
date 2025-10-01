import copy
import warnings
import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.utils.checkpoint as cp

import onnx
from onnxsim import simplify

from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.config import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

from mmdet3d.registry import MODELS
from mmdet3d.models.utils.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch


# Disable warnings
warnings.filterwarnings("ignore")

def get_global_pos(points, pc_range):
    points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
    return points

import numpy as np
class SelfAttentionONNX(nn.Module):
    def __init__(self, self_attn_module):
        super(SelfAttentionONNX, self).__init__()
        self.self_attn_module = copy.deepcopy(self_attn_module)

    def forward(self, query, key, value,
                query_pos=None, key_pos=None,
                identity=None,
                attn_mask=None, key_padding_mask=None):
        return self.self_attn_module(
            query,
            key,
            value,
            identity,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)


class CrossAttentionONNX(nn.Module):
    def __init__(self, cross_attn_module):
        super(CrossAttentionONNX, self).__init__()
        self.cross_attn_module = copy.deepcopy(cross_attn_module)

    def forward(self, query, query_pos,
                mlvl_feats, reference_points,
                spatial_flatten, level_start_index,
                pc_range, lidar2img, pad_shape):
        return self.cross_attn_module(
            query,
            query_pos,
            mlvl_feats,
            reference_points,
            spatial_flatten,
            level_start_index,
            pc_range,
            lidar2img,
            pad_shape)

class SinleLayerONNX(nn.Module):
    def __init__(self, layer_module, layer_id, img_metas, spatial_flatten):
        super(SinleLayerONNX, self).__init__()
        self.layer_module = copy.deepcopy(layer_module)
        self.layer_id = layer_id
        self.img_metas = img_metas
        self.spatial_flatten = spatial_flatten
        
    def forward(self,
                query,
                query_pos,
                mlvl_feats,
                temp_memory,
                temp_pos,
                reference_points,
                #spatial_flatten,
                level_start_index,
                pc_range,
                lidar2img,
                attn_masks=None):

        return self.layer_module(
            self.layer_id,
            query,
            query_pos,
            mlvl_feats,
            temp_memory,
            temp_pos,
            reference_points,
            #spatial_flatten,
            self.spatial_flatten,
            level_start_index,
            pc_range,
            lidar2img,
            self.img_metas,
            attn_masks
        )

@MODELS.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = MODELS.build(decoder)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(self,
                query,
                query_pos,
                feat_flatten,
                spatial_flatten,
                level_start_index,
                temp_memory,
                temp_pos,
                attn_masks,
                reference_points,
                pc_range,
                lidar2imgs,
                img_metas,):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        #lidar2img = data['lidar2img']
        inter_states = self.decoder(
            query=query,
            query_pos=query_pos,
            mlvl_feats=feat_flatten,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            reference_points=reference_points,
            spatial_flatten=spatial_flatten,
            level_start_index=level_start_index,
            pc_range=pc_range, 
            lidar2img=lidar2imgs,
            img_metas=img_metas,
            attn_masks=attn_masks)

        return inter_states

@MODELS.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, embed_dims, *args, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.set_layer_export = False

    def export_single_layer(self,
                            layer_id,
                            layer_onnx,
                            query,
                            query_pos,
                            mlvl_feats,
                            temp_memory,
                            temp_pos,
                            reference_points,
                            spatial_flatten,
                            level_start_index,
                            pc_range,
                            lidar2img,
                            attn_masks=None):

        # Save inputs
        query_np = query.to('cpu').numpy()
        query_pos_np = query_pos.to('cpu').numpy()
        mlvl_feats_np = mlvl_feats.to('cpu').numpy()
        temp_memory_np = temp_memory.to('cpu').numpy()
        temp_pos_np = temp_pos.to('cpu').numpy()
        reference_points_np = reference_points.to('cpu').numpy()
        #spatial_flatten_np = spatial_flatten.to('cpu').numpy()
        level_start_index_np = level_start_index.to('cpu').numpy()
        pc_range_np = pc_range.to('cpu').numpy()
        lidar2img_np = lidar2img.to('cpu').numpy()
        attn_masks_np = attn_masks.to('cpu').numpy() if attn_masks is not None else None

        np.savez(f'single_layer_input_{layer_id}.npz',
                 query=query_np, query_pos=query_pos_np,
                 mlvl_feats=mlvl_feats_np, 
                 temp_memory=temp_memory_np, temp_pos=temp_pos_np,
                 reference_points=reference_points_np,
                 #spatial_flatten=spatial_flatten_np, 
                 level_start_index=level_start_index_np,
                 pc_range=pc_range_np, lidar2img=lidar2img_np,
                 attn_masks=attn_masks_np)

        model_input=[]
        input_names=[]
        model_input.append(copy.deepcopy(query))
        input_names.append('query')
        model_input.append(copy.deepcopy(query_pos))
        input_names.append('query_pos')
        model_input.append(copy.deepcopy(mlvl_feats))
        input_names.append('mlvl_feats')
        model_input.append(copy.deepcopy(temp_memory))
        input_names.append('temp_memory')
        model_input.append(copy.deepcopy(temp_pos))
        input_names.append('temp_pos')
        model_input.append(copy.deepcopy(reference_points))
        input_names.append('reference_points')
        #model_input.append(copy.deepcopy(spatial_flatten))
        #input_names.append('spatial_flatten')
        model_input.append(copy.deepcopy(level_start_index))
        input_names.append('level_start_index')
        model_input.append(copy.deepcopy(pc_range))
        input_names.append('pc_range')
        model_input.append(copy.deepcopy(lidar2img))
        input_names.append('lidar2img')
        model_input.append(copy.deepcopy(attn_masks))
        input_names.append('attn_masks')

        model_name = f"single_layer_{layer_id}.onnx"
        torch.onnx.export(
            layer_onnx,
            tuple(model_input),
            model_name,
            opset_version=18,
            input_names=input_names,
            output_names=['output'])
        onnx_model, _ = simplify(model_name)
        onnx.save(onnx_model, model_name)


    def forward(self,
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range,
                lidar2img,
                img_metas,
                attn_masks):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            if self.set_layer_export is True:
                layer_module = layer
                layer_onnx = SinleLayerONNX(layer_module, lid, img_metas, spatial_flatten)
                self.export_single_layer(lid,
                                         layer_onnx,
                                         query,
                                         query_pos,
                                         mlvl_feats,
                                         temp_memory,
                                         temp_pos,
                                         reference_points,
                                         spatial_flatten,
                                         level_start_index,
                                         pc_range,
                                         lidar2img,
                                         attn_masks)
                print("Successfully export a single decoder layer to onnx!")

            query = layer(
                lid,
                query,
                query_pos,
                mlvl_feats,
                temp_memory,
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range,
                lidar2img,
                img_metas,
                attn_masks)

            intermediate.append(query)

        self.set_layer_export=False
        return torch.stack(intermediate)


@MODELS.register_module()
class Detr3DTemporalDecoderLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 use_reentrant=False,
                 **kwargs):
        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        self.use_checkpoint = with_cp
        self.use_reentrant = use_reentrant

        # For onnx model validation only
        self.layer_export = False


    def export_self_attn(self,
                         layer_id,
                         self_attn_onnx,
                         query,
                         key,
                         value,
                         identity,
                         query_pos=None,
                         key_pos=None,
                         attn_mask=None,
                         key_padding_mask=None):
        # Save inputs
        query_np = query.to('cpu').numpy()
        key_np   = key.to('cpu').numpy()
        value_np = value.to('cpu').numpy()
        if identity is None:
            identity_np = None
        else:
            identity_np = identity.to('cpu').numpy()
        query_pos_np    = query_pos.to('cpu').numpy()
        key_pos_np      = key_pos.to('cpu').numpy()
        if attn_mask is None:
            attn_mask_np = None
        else:
            attn_mask_np = attn_mask.to('cpu').numpy()
        if key_padding_mask is None:
            key_padding_mask_np = None
        else:
            key_padding_mask_np = key_padding_mask.to('cpu').numpy()

        np.savez(f"self_attn_input_{layer_id}.npz",
                 query=query_np, key=key_np,
                 value=value_np, identity=identity_np,
                 query_pos=query_pos_np, key_pos=key_pos_np,
                 attn_mask=attn_mask_np, key_padding_mask=key_padding_mask_np)

        model_input=[]
        input_names=[]
        model_input.append(copy.deepcopy(query))
        input_names.append('query')
        model_input.append(copy.deepcopy(key))
        input_names.append('key')
        model_input.append(copy.deepcopy(value))
        input_names.append('value')
        model_input.append(copy.deepcopy(query_pos))
        input_names.append('query_pos')
        model_input.append(copy.deepcopy(key_pos))
        input_names.append('key_pos')
        # put possible None variable in the end of input list
        if identity is not None:
            model_input.append(copy.deepcopy(identity))
            input_names.append('identity')
        if attn_mask is not None:
            model_input.append(copy.deepcopy(attn_mask))
            input_names.append('attn_mask')
        if key_padding_mask is not None:
            model_input.append(copy.deepcopy(key_padding_mask))
            input_names.append('key_padding_mask')

        model_name = f"self_attention_{layer_id}.onnx"
        torch.onnx.export(
            self_attn_onnx,
            tuple(model_input),
            model_name,
            opset_version=18,
            input_names=input_names,
            output_names=['output']
        )
        onnx_model, _ = simplify(model_name)
        onnx.save(onnx_model, model_name)

    def export_cross_attn(self,
                          layer_id,
                          cross_attn_onnx,
                          query,
                          query_pos,
                          mlvl_feats,
                          reference_points,
                          spatial_flatten,
                          level_start_index,
                          pc_range,
                          lidar2img,
                          pad_shape):

        # Save inputs
        query_np = query.to('cpu').numpy()
        query_pos_np   = query_pos.to('cpu').numpy()
        mlvl_feats_np = mlvl_feats.to('cpu').numpy()
        reference_points_np = reference_points.to('cpu').numpy()
        spatial_flatten_np    = spatial_flatten.to('cpu').numpy()
        level_start_index_np      = level_start_index.to('cpu').numpy()
        pc_range_np = pc_range.to('cpu').numpy()
        lidar2img_np = lidar2img.to('cpu').numpy()
        pad_shape_np = np.array(pad_shape)

        np.savez(f'cross_attn_input_{layer_id}.npz',
                 query=query_np, query_pos=query_pos_np,
                 mlvl_feats=mlvl_feats_np, reference_points=reference_points_np,
                 spatial_flatten=spatial_flatten_np, level_start_index=level_start_index_np,
                 pc_range=pc_range_np, lidar2img=lidar2img_np, pad_shape=pad_shape_np)

        model_input=[]
        input_names=[]
        model_input.append(copy.deepcopy(query))
        input_names.append('query')
        model_input.append(copy.deepcopy(query_pos))
        input_names.append('query_pos')
        model_input.append(copy.deepcopy(mlvl_feats))
        input_names.append('mlvl_feats')
        model_input.append(copy.deepcopy(reference_points))
        input_names.append('reference_points')
        model_input.append(copy.deepcopy(spatial_flatten))
        input_names.append('spatial_flatten')
        model_input.append(copy.deepcopy(level_start_index))
        input_names.append('level_start_index')
        model_input.append(copy.deepcopy(pc_range))
        input_names.append('pc_range')
        model_input.append(copy.deepcopy(lidar2img))
        input_names.append('lidar2img')
        model_input.append(torch.tensor(pad_shape))
        input_names.append('pad_shape')

        model_name = f"cross_attention_{layer_id}.onnx"
        torch.onnx.export(
            cross_attn_onnx,
            tuple(model_input),
            model_name,
            opset_version=18,
            input_names=input_names,
            output_names=['output'])
        onnx_model, _ = simplify(model_name)
        onnx.save(onnx_model, model_name)


    def _forward(self,
                layer_id,
                query,
                query_pos,
                mlvl_feats,
                temp_memory,
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range,
                lidar2img,
                img_metas,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
                    temp_pos = torch.cat([query_pos, temp_pos], dim=1)
                else:
                    temp_key = temp_value = query
                    temp_pos = query_pos

                # For onnx model validation only
                if self.layer_export is True:
                    self_attn_module = self.attentions[attn_index]
                    self_attn_onnx = SelfAttentionONNX(self_attn_module)
                    self.export_self_attn(layer_id,
                                          self_attn_onnx,
                                          query,
                                          temp_key,
                                          temp_value,
                                          identity if self.pre_norm else None,
                                          query_pos=query_pos,
                                          key_pos=temp_pos,
                                          attn_mask=attn_masks[attn_index],
                                          key_padding_mask=query_key_padding_mask)
                    print("Successfully export self attention to onnx!")

                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                # For onnx model validation only
                if self.layer_export is True:
                    cross_attn_module = self.attentions[attn_index]
                    cross_attn_onnx = CrossAttentionONNX(cross_attn_module)
                    self.export_cross_attn(layer_id,
                                           cross_attn_onnx,
                                           query,
                                           query_pos,
                                           mlvl_feats,
                                           reference_points,
                                           spatial_flatten,
                                           level_start_index,
                                           pc_range,
                                           lidar2img,
                                           img_metas[0]['pad_shape'])
                    print("Successfully export cross attention to onnx!")

                query = self.attentions[attn_index](
                    query,
                    query_pos,
                    mlvl_feats,
                    reference_points,
                    spatial_flatten,
                    level_start_index,
                    pc_range,
                    lidar2img,
                    img_metas[0]['pad_shape'],
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        self.layer_export=False
        return query

    def forward(self,
                layer_id,
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward,
                query,
                query_pos,
                mlvl_feats,
                temp_memory,
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range,
                lidar2img,
                img_metas,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                use_reentrant=self.use_reentrant,
                )
        else:
            x = self._forward(
            layer_id,
            query,
            query_pos,
            mlvl_feats,
            temp_memory, 
            temp_pos,
            reference_points,
            spatial_flatten,
            level_start_index,
            pc_range, 
            lidar2img, 
            img_metas,
            attn_masks,
            query_key_padding_mask,
            key_padding_mask
        )
        return x


@MODELS.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            num_pts=13,
            im2col_step=64,
            batch_first=True,
            bias=1.,
            grid_sample_mode='bilinear',
            ):
        super(DeformableFeatureAggregation, self).__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.num_levels * num_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.learnable_fc = nn.Linear(self.embed_dims, num_pts * 3)
        self.cam_embed = nn.Sequential(
            nn.Linear(12, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),
        )
        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.bias = bias
        self.grid_sample_mode = grid_sample_mode

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        nn.init.uniform_(self.learnable_fc.bias.data, -self.bias, self.bias)

    def forward(self, instance_feature, query_pos,feat_flatten, reference_points, spatial_flatten, level_start_index, pc_range, lidar2img_mat, pad_shape):
        bs, num_anchor = reference_points.shape[:2]
        reference_points = get_global_pos(reference_points, pc_range)
        key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature).reshape(bs, num_anchor, -1, 3)

        weights = self._get_weights(instance_feature, query_pos, lidar2img_mat)

        features = self.feature_sampling(feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, pad_shape)

        output = self.output_proj(features)
        output = self.drop(output) + instance_feature
        return output

    def _get_weights(self, instance_feature, anchor_embed, lidar2img_mat):
        bs, num_anchor = instance_feature.shape[:2]
        lidar2img = lidar2img_mat[..., :3, :].flatten(-2)
        cam_embed = self.cam_embed(lidar2img) # B, N, C
        feat_pos = (instance_feature + anchor_embed).unsqueeze(2) + cam_embed.unsqueeze(1)
        weights = self.weights_fc(feat_pos).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2)
        weights = weights.reshape(bs, num_anchor, self.num_cams, -1, self.num_groups).permute(0, 2, 1, 4, 3).contiguous()
        return weights.flatten(end_dim=1)

    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, pad_shape):
        bs, num_anchor, _ = key_points.shape[:3]

        pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)

        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        points_2d[..., 0:1] = points_2d[..., 0:1] / pad_shape[1]
        points_2d[..., 1:2] = points_2d[..., 1:2] / pad_shape[0]

        points_2d = points_2d.flatten(end_dim=1) #[b*6, 900, 13, 2]
        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
        # attention_weights = weights * mask
        if 0:
            output = MultiScaleDeformableAttnFunction.apply(
                    feat_flatten, spatial_flatten, level_start_index, points_2d,
                    weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                    feat_flatten, spatial_flatten, points_2d, weights, self.grid_sample_mode)

        output = output.reshape(bs, self.num_cams, num_anchor, -1)

        return output.sum(1)