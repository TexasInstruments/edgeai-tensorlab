# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch
import numpy as np

# from mmengine.model.base_module import force_fp32, auto_fp16
from mmengine.registry import build_from_cfg
from mmengine.model.base_module import BaseModule
from mmengine.registry import MODELS
from .grid_mask import GridMask
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures.det3d_data_sample import Det3DDataSample 

try:
    from .ops import DeformableAggregationFunction as DAF
except:
    DAF = None

__all__ = ["Sparse4D"]

def format_value(val):
    if val is None:
        return val
    if isinstance(val, (list, tuple)) and all(isinstance(x, (list,tuple)) for x in val):
        try:
            new_val = np.array(val)
            temp = np.max(val).astype(np.float32)
            val = new_val
        except:
            pass
    if isinstance(val, (list, tuple)) and all(isinstance(x, np.ndarray) for x in val):
        val = np.stack(val)
    try:
        new_val = np.array(val)
        temp = np.max(val).astype(np.float32)
        val = new_val
    except:
        pass
    if isinstance(val, np.ndarray):
        val = torch.from_numpy(val)
    return val


def get_metas(img,data_Samples: list[Det3DDataSample], move_to_img_device_keys=[]):
    metas = {}
    keys = list(data_Samples[0].keys()) + list(data_Samples[0].metainfo.keys())
    for key in keys:
        if key not in metas:
            metas[key] = []
        for data_sample in data_Samples:
            if key in data_sample.metainfo:
                val = data_sample.metainfo[key]
            else:
                val = getattr(data_sample,key)
            val = format_value(val)
            metas[key].append(val)
        if all(isinstance(t, torch.Tensor) for t in metas[key]):
            metas[key] = torch.stack(metas[key])
        
        if key in move_to_img_device_keys:
            try:
                metas[key] = metas[key].to(img[0].device) # comment the below line and uncomment this if no stacking(torch.stack(metas[key])) is used
                # metas[key] = [t.to(img.device) for t in metas[key]] # comment the above line and uncomment this if no stacking(torch.stack(metas[key])) is not used
            except Exception as e:
                print(f"Got error: for key({key}): {e}")
                print(f"please make sure that all the entries for {key} in datasample are tensors at this point")
                
    return metas


@MODELS.register_module()
class Sparse4D(MVXTwoStageDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            img_backbone.pretrained = pretrained
        self.img_backbone = build_from_cfg(img_backbone, MODELS)
        if img_neck is not None:
            self.img_neck = build_from_cfg(img_neck, MODELS)
        self.head = build_from_cfg(head, MODELS)
        self.use_grid_mask = use_grid_mask
        self.use_deformable_func = use_deformable_func and DAF is not None
        if self.use_deformable_func:
            self.deformable_func = DAF()
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, MODELS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    # @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        if isinstance(img, (list,tuple)):
            img = torch.stack(img)
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
            feature_maps = self.deformable_func.feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    # @force_fp32(apply_to=("img",))
    def _forward(self,inputs, data_samples,  return_depth=False, **kwargs):
        img = inputs.pop("img")
        move_to_img_device_keys = ["projection_mat", "image_wh"]
        metas = []
        metas = get_metas(img, data_samples, move_to_img_device_keys)
        if return_depth:
            feature_maps, depths = self.extract_feat(img, return_depth, metas)
        else:
            feature_maps = self.extract_feat(img, )
        
        if "data_queue" in kwargs or "future_data_queue" in kwargs:
            feature_queue = []
            meta_queue = []
            with torch.no_grad():
                for d in kwargs.get("data_queue", []) + kwargs.get(
                    "future_data_queue", []
                ):
                    img = d["inputs"].pop("img")
                    feature_queue.append(self.extract_feat(img))
                    meta_queue.append(get_metas(img, d['data_samples'], move_to_img_device_keys))
        else:
            feature_queue = None
            meta_queue = None

        cls_scores, reg_preds = self.head(
            feature_maps, metas, feature_queue, meta_queue
        )
        if return_depth:
            return cls_scores, reg_preds, depths
        else:
            return cls_scores, reg_preds
    
    def loss(self, inputs, data_samples, **kwargs):
        cls_scores, reg_preds, depths = self(inputs, data_samples, return_depth=True,**kwargs)
        if self.use_deformable_func:
            feature_maps = self.deformable_func.feature_maps_format(feature_maps, inverse=True)
        output = self.head.loss(cls_scores, reg_preds, kwargs, feature_maps)
        if depths is not None and "gt_depth" in kwargs:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, kwargs["gt_depth"]
            )
        return output

    def predict(self,inputs, data_samples,  **kwargs):
        cls_scores, reg_preds = self(inputs, data_samples, **kwargs)
        results = self.head.post_process(cls_scores, reg_preds)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, inputs, data_samples, **kwargs):
        # fake test time augmentation
        for key in kwargs.keys():
            if isinstance(kwargs[key], list):
                kwargs[key] = kwargs[key][0]
        return self.predict(inputs, data_samples, **kwargs)

    def forward(self,
                inputs, data_samples, 
                # OptSampleList = None,
                mode: str = 'tensor',
                **kwargs):
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
                    # export_PETR(self, inputs, data_samples,
                        # quantized_model=self.quantized_model, opset_version=18, **kwargs)
                    # Export onnx only once
                    self.save_onnx_model = False

                return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
