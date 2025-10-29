# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch
from mmengine.structures import InstanceData

# from mmengine.model.base_module import force_fp32, auto_fp16
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures.det3d_data_sample import ForwardResults, OptSampleList

from projects_edgeai.edgeai_mmdet3d.grid_mask import GridMask

try:
    from .ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

from .onnx_export import export_Sparse4D, export_Sparse4D_subnets

__all__ = ["Sparse4D"]

@MODELS.register_module()
class Sparse4D(MVXTwoStageDetector):
    def __init__(
        self,
        use_grid_mask=True,
        use_deformable_func=False,
        save_onnx_model=False,
        onnx_subnets=False,
        data_preprocessor=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        depth_branch=None,
    ):
        super(Sparse4D, self).__init__(data_preprocessor=data_preprocessor,
                                       img_backbone=img_backbone,
                                       img_neck=img_neck,
                                       pts_bbox_head=pts_bbox_head,
                                       init_cfg=init_cfg,
                                       train_cfg=train_cfg,
                                       test_cfg=test_cfg)
        if pretrained is not None:
            img_backbone.pretrained = pretrained

        self.use_grid_mask = use_grid_mask
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func

        if depth_branch is not None:
            self.depth_branch = MODELS.build(depth_branch)
        else:
            self.depth_branch = None

        self.save_onnx_model    = save_onnx_model
        self.onnx_subnets       = onnx_subnets
        self.onnx_model         = None
        self.onnx_img_backbone  = None
        self.onnx_pts_bbox_head = None


    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    # @force_fp32(apply_to=("img",))
    def _forward(self, inputs, data_samples, **kwargs):
        raise NotImplementedError('tensor mode is yet to add')


    def loss(self, inputs, data_samples, **kwargs):

        rec_img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]
        
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.pts_bbox_head(feature_maps, data)
        output = self.pts_bbox_head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def predict(self, inputs, data_samples,  **kwargs):
        img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]
        for var, name in [(batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        feature_maps = self.extract_feat(img)

        model_outs = self.pts_bbox_head(feature_maps, batch_img_metas)
        results = self.pts_bbox_head.post_process(model_outs, batch_img_metas)

        bbox_list = [dict() for i in range(len(batch_img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict['pts_bbox'] = pts_bbox

        for i, data_sample in enumerate(data_samples):
            bbox_list_i = InstanceData(
                metainfo=bbox_list[i]['pts_bbox'])
            data_sample.pred_instances_3d = bbox_list_i
            data_sample.pred_instances = InstanceData()
        return data_samples

    def aug_test(self, inputs, data_samples, **kwargs):
        # fake test time augmentation
        for key in kwargs.keys():
            if isinstance(kwargs[key], list):
                kwargs[key] = kwargs[key][0]
        return self.predict(inputs, data_samples, **kwargs)

    def forward(self,
                inputs,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and kwargs samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation kwargs of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            if isinstance(data_samples[0], list):
                # aug test
                assert len(data_samples[0]) == 1, 'Only support ' \
                                                  'batch_size 1 ' \
                                                  'in mmdet3d when ' \
                                                  'do the test' \
                                                  'time augmentation.'
                return self.aug_test(inputs, data_samples, **kwargs)
            else:
                if self.save_onnx_model is True:
                    if self.onnx_subnets:
                        export_Sparse4D_subnets(self, inputs, data_samples, opset_version=18, **kwargs)
                    else:
                        export_Sparse4D(self, inputs, data_samples, opset_version=18, **kwargs)
                    self.save_onnx_model = False

                return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
