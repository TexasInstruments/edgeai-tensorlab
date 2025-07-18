import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from torch import Tensor
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet3d.models.task_modules.builder import build_bbox_coder

from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmdet.models.utils import multi_apply
from mmdet.models.layers import inverse_sigmoid

from mmcv.cnn import Linear, build_activation_layer
from mmengine.model import bias_init_with_prob
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmdet3d.registry import MODELS, TASK_UTILS

from .util import normalize_bbox, denormalize_bbox

from mmengine.structures import InstanceData

import numpy as np


@MODELS.register_module()
class BEVFormerHead(AnchorFreeHead):
    """Head of BEVFormer.

    Args:

        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        bbox_coder (obj:`ConfigDict`): Configs to build the bbox coder
        num_cls_fcs (int) : the number of layers in cls and reg branch
        code_weights (List[double]) : loss weights of
            (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y)
        bev_h, bev_w (int): spatial shape of BEV queries.
        num_query (int): Number of query in Transformer. Defaults to 100.
        num_reg_fcs (int): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head.
            Defaults to 2.
        transformer (:obj:`ConfigDict` or dict, optional): Config for
            transformer. Defaults to None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of all
            ranks. Defaults to False.
        positional_encoding (:obj:`ConfigDict` or dict): Config for position
            encoding.
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_bbox (:obj:`ConfigDict` or dict): Config of the regression loss.
            Defaults to `L1Loss`.
        loss_iou (:obj:`ConfigDict` or dict): Config of the regression iou
            loss. Defaults to `GIoULoss`.
        tran_cfg (:obj:`ConfigDict` or dict): Training config of transformer
            head.
        test_cfg (:obj:`ConfigDict` or dict): Testing config of transformer
            head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            with_box_refine=False,
            as_two_stage=False,
            bbox_coder=None,
            num_cls_fcs=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            bev_h=30,
            bev_w=30,
            num_query=100,
            num_reg_fcs=2,
            transformer=None,
            sync_cls_avg_factor=False,
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            loss_cls=dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='ClassificationCost', weight=1.),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ])),
            test_cfg=dict(max_per_img=100),
            init_cfg=None,
            **kwargs) -> None:

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1


        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is BEVFormerHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('BEVFormer do not build sampler.')

            # Use PseudoSampler, format the result
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = TASK_UTILS.build(sampler_cfg)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = TASK_UTILS.build(
            positional_encoding)
        self.transformer = MODELS.build(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'

        self._init_layers()
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

   
    def forward(self, mlvl_feats, img_metas, prev_bev=None,
                rotation_grid = None,
                reference_points_cam=None,
                bev_mask_count=None,
                bev_valid_indices=None,
                bev_valid_indices_count=None,
                shift_xy=None, 
                can_bus=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype) # bev_mask: 1x50x50
        bev_pos = self.positional_encoding(bev_mask).to(dtype)      # bev_pos:  1x256x50x50

        # These tensors are constant
        #bev_queries.cpu().numpy().tofile("./temp/bev_queries.dat")
        #bev_pos.cpu().numpy().tofile("./temp/bev_pos.dat")
        #object_query_embeds.cpu().numpy().tofile("./temp/object_query_embeds.dat")

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
                rotation_grid=rotation_grid,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                rotation_grid=rotation_grid,
                reference_points_cam=reference_points_cam,
                bev_mask_count=bev_mask_count,
                bev_valid_indices=bev_valid_indices,
                bev_valid_indices_count=bev_valid_indices_count,
                shift_xy=shift_xy,
                can_bus=can_bus
        )

        bev_embed, hs, init_reference, inter_references = outputs
        outputs_classes = []
        outputs_coords = []

        if torch.onnx.is_in_onnx_export():
            batch_hs = hs.reshape(len(self.cls_branches), -1, hs.shape[-1])
            outputs_classes = self.run_cls_branch(batch_hs)
            outputs_coords = self.run_reg_branch(batch_hs, init_reference, inter_references)
        else:
            hs = hs.permute(0, 2, 1, 3)
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])

                # TODO: check the shape of reference
                assert reference.shape[-1] == 3
                tmp[..., 0:1] = ((tmp[..., 0:1] + reference[..., 0:1]).sigmoid() * (self.pc_range[3] -
                                 self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = ((tmp[..., 1:2] + reference[..., 1:2]).sigmoid() * (self.pc_range[4] -
                                 self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = ((tmp[..., 4:5] + reference[..., 2:3]).sigmoid() * (self.pc_range[5] -
                                 self.pc_range[2]) + self.pc_range[2])

                # TODO: check if using sigmoid
                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs


    def run_reg_branch(self, batch_hs, init_reference, inter_references):
        #reference = []
        #for lvl in range(hs.shape[0]):
        #    if lvl == 0:
        #        reference.append(init_reference)
        #    else:
        #        reference.append(inter_references[lvl - 1])
        #reference=torch.cat(reference, dim=0)
        reference=torch.cat((init_reference, inter_references[0:5]), dim=0)
        reference = inverse_sigmoid(reference)

        #batch_hs = hs.squeeze(1)

        reg_linear1_wgt = []
        reg_linear1_bias = []
        reg_linear2_wgt = []
        reg_linear2_bias = []
        reg_linear3_wgt = []
        reg_linear3_bias = []

        for lvl in range(len(self.reg_branches)):
            reg_linear1_wgt.append(self.reg_branches[lvl][0].weight.transpose(1, 0))
            reg_linear1_bias.append(self.reg_branches[lvl][0].bias.unsqueeze(0))
            reg_linear2_wgt.append(self.reg_branches[lvl][2].weight.transpose(1, 0))
            reg_linear2_bias.append(self.reg_branches[lvl][2].bias.unsqueeze(0))
            reg_linear3_wgt.append(self.reg_branches[lvl][4].weight.transpose(1, 0))
            reg_linear3_bias.append(self.reg_branches[lvl][4].bias.unsqueeze(0))

        reg_linear1_wgt  = torch.stack(reg_linear1_wgt)
        reg_linear1_bias = torch.stack(reg_linear1_bias)
        reg_linear2_wgt  = torch.stack(reg_linear2_wgt)
        reg_linear2_bias = torch.stack(reg_linear2_bias)
        reg_linear3_wgt  = torch.stack(reg_linear3_wgt)
        reg_linear3_bias = torch.stack(reg_linear3_bias)

        # Run reg_branch
        # 1. Linear
        out  = torch.bmm(batch_hs, reg_linear1_wgt)
        out  = torch.add(out, reg_linear1_bias)

        # 2. ReLU
        out = torch.relu(out)

        # 3. Linear
        out  = torch.bmm(out, reg_linear2_wgt)
        out  = torch.add(out, reg_linear2_bias)

        # 4. ReLU
        out = torch.relu(out)

        # 5. Linear
        out  = torch.bmm(out, reg_linear3_wgt)
        out  = torch.add(out, reg_linear3_bias)

        """
        out[..., 0:1] = ((out[..., 0:1] + reference[..., 0:1]).sigmoid() * (self.pc_range[3] -
                         self.pc_range[0]) + self.pc_range[0])
        out[..., 1:2] = ((out[..., 1:2] + reference[..., 1:2]).sigmoid() * (self.pc_range[4] -
                         self.pc_range[1]) + self.pc_range[1])
        out[..., 4:5] = ((out[..., 4:5] + reference[..., 2:3]).sigmoid() * (self.pc_range[5] -
                         self.pc_range[2]) + self.pc_range[2])
        """
        # Use index_put_ directly to have a single ScatterND
        temp = torch.cat((out[..., 0:2], out[...,4:5]), dim=-1)
        temp = (temp + reference).sigmoid()
        m0 = torch.Tensor([self.pc_range[3] - self.pc_range[0],
                           self.pc_range[4] - self.pc_range[1], 
                           self.pc_range[5] - self.pc_range[2]]).to(temp.device)
        a0 = torch.Tensor([self.pc_range[0], self.pc_range[1], self.pc_range[2]]).to(temp.device)
        temp  = temp*m0 + a0

        d0 = temp.size(0)
        d1 = temp.size(1)
        d2 = temp.size(2)
        p0 = torch.arange(0, d0).to(temp.device).view(-1, 1, 1).expand(d0, d1, d2)
        p1 = torch.arange(0, d1).to(temp.device).view(1, -1, 1).expand(d0, d1, d2)
        p2 = torch.Tensor([0,1,4]).to(torch.int64).to(temp.device).view(1, 1, -1).expand(d0, d1, d2)
        indices = tuple([p0, p1, p2])
        out.index_put_(indices, temp)

        out = out.unsqueeze(1)
        return out


    def run_cls_branch(self, batch_hs):
        #batch_hs = hs.squeeze(1)

        cls_linear1_wgt = []
        cls_linear1_bias = []
        cls_linear2_wgt = []
        cls_linear2_bias = []
        cls_linear3_wgt = []
        cls_linear3_bias = []
        layer_norm1_wgt = []
        layer_norm1_bias = []
        layer_norm2_wgt = []
        layer_norm2_bias = []
        for lvl in range(len(self.cls_branches)):
            cls_linear1_wgt.append(self.cls_branches[lvl][0].weight.transpose(1, 0))
            cls_linear1_bias.append(self.cls_branches[lvl][0].bias.unsqueeze(0))
            cls_linear2_wgt.append(self.cls_branches[lvl][3].weight.transpose(1, 0))
            cls_linear2_bias.append(self.cls_branches[lvl][3].bias.unsqueeze(0))
            cls_linear3_wgt.append(self.cls_branches[lvl][6].weight.transpose(1, 0))
            cls_linear3_bias.append(self.cls_branches[lvl][6].bias.unsqueeze(0))

            layer_norm1_wgt.append(self.cls_branches[lvl][1].weight.unsqueeze(0))
            layer_norm1_bias.append(self.cls_branches[lvl][1].bias.unsqueeze(0))
            layer_norm2_wgt.append(self.cls_branches[lvl][4].weight.unsqueeze(0))
            layer_norm2_bias.append(self.cls_branches[lvl][4].bias.unsqueeze(0))

        cls_linear1_wgt  = torch.stack(cls_linear1_wgt)
        cls_linear1_bias = torch.stack(cls_linear1_bias)
        cls_linear2_wgt  = torch.stack(cls_linear2_wgt)
        cls_linear2_bias = torch.stack(cls_linear2_bias)
        cls_linear3_wgt  = torch.stack(cls_linear3_wgt)
        cls_linear3_bias = torch.stack(cls_linear3_bias)
        layer_norm1_wgt  = torch.stack(layer_norm1_wgt)
        layer_norm1_bias = torch.stack(layer_norm1_bias)
        layer_norm2_wgt  = torch.stack(layer_norm2_wgt)
        layer_norm2_bias = torch.stack(layer_norm2_bias)

        # Run cls_branch
        # 1. Linear
        out  = torch.bmm(batch_hs, cls_linear1_wgt)
        out  = torch.add(out, cls_linear1_bias)

        # 2. LayerNorm
        out1 = torch.mean(out, dim=-1, keepdim=True)
        out  = torch.sub(out, out1)
        out1 = torch.pow(out, 2.0)
        out1 = torch.mean(out1, dim=-1, keepdim=True)
        out1 = out1 + 1e-5
        out1 = torch.sqrt(out1)
        out  = torch.div(out, out1)
        out  = torch.mul(out, layer_norm1_wgt)
        out  = torch.add(out, layer_norm1_bias)

        # 3. ReLU
        out  = torch.relu(out)

        # 4. Linear
        out  = torch.bmm(out, cls_linear2_wgt)
        out  = torch.add(out, cls_linear2_bias)

        # 5. LayerNorm
        out1 = torch.mean(out, dim=-1, keepdim=True)
        out  = torch.sub(out, out1)
        out1 = torch.pow(out, 2.0)
        out1 = torch.mean(out1, dim=-1, keepdim=True)
        out1 = out1 + 1e-5
        out1 = torch.sqrt(out1)
        out  = torch.div(out, out1)
        out  = torch.mul(out, layer_norm2_wgt)
        out  = torch.add(out, layer_norm2_bias)

        # 6. ReLU
        out  = torch.relu(out)

        # 7. Linear
        out  = torch.bmm(out, cls_linear3_wgt)
        out  = torch.add(out, cls_linear3_bias)

        out  = out.unsqueeze(1)
        return out

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        # force to gpu
        gt_labels = gt_labels.to(bbox_pred.device)
        gt_bboxes = gt_bboxes.to(bbox_pred.device)

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, InstanceData(priors=bbox_pred),
                                              InstanceData(bboxes_3d=gt_bboxes))
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox


    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict


    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list

    def get_bboxes_onnx(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions for onnx model exports
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            # for onnx export
            bboxes = bboxes.tensor
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list

    # It is not used, but needed for abstract method
    # Can be modifed so that it is identical to loss()
    def loss_by_feat(
        self,
        all_cls_scores_list: List[Tensor],
        all_bbox_preds_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        assert batch_gt_instances_ignore is None, \
            'Only supports for batch_gt_instances_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        batch_img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single, all_cls_scores, all_bbox_preds,
            batch_gt_instances_list, batch_img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1

        return loss_dict
