from inspect import signature

import torch
import torch.nn as nn
import numpy as np
import copy
import onnx
from onnxsim import simplify
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures.det3d_data_sample import ForwardResults, OptSampleList

from projects_edgeai.edgeai_mmdet3d.grid_mask import GridMask

try:
    from .ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

from .onnx_export import export_SparseDrive, export_SparseDrive_subnets

__all__ = ["SparseDrive"]

# To export pts_bbox_head.post_process
# Recommend to diable save_onnx_model, and
# manually set SparseDrive.postprocess_export = True
def export_post_process(post_process_onnx, model_outs, data):
    model_input = []
    model_input.append(data['gt_ego_fut_cmd'])
    model_input.append(model_outs)

    model_name = "head_post_process.onnx"
    input_names = ["gt_ego_fut_cmd",
                   "det_out_classification_0", "det_out_classification_1", "det_out_classification_2", "det_out_classification_3", "det_out_classification_4", "det_out_classification_5",
                   "det_out_prediction_0", "det_out_prediction_1", "det_out_prediction_2", "det_out_prediction_3", "det_out_prediction_4", "det_out_prediction_5",
                   "det_out_quality_0", "det_out_quality_1", "det_out_quality_2", "det_out_quality_3", "det_out_quality_4", "det_out_quality_5",
                   "det_out_instance_feature", "det_out_anchor_embed", "det_out_instance_id",
                   "map_out_classification_0", "map_out_classification_1", "map_out_classification_2", "map_out_classification_3", "map_out_classification_4", "map_out_classification_5",
                   "map_out_prediction_0", "map_out_prediction_1", "map_out_prediction_2", "map_out_prediction_3", "map_out_prediction_4", "map_out_prediction_5",
                   "map_out_instance_feature", "map_out_anchor_embed",
                   "motion_out_classification", "motion_out_prediction", "motion_out_period", "motion_out_anchor_queue",
                   "planning_out_classification", "planning_out_prediction", "planning_out_anchor_staus", "planning_out_period", "planning_out_anchor_queue"]
    output_names = ["bboxes_3d", "scores_3d", "labels_3d", "cls_scores", "instance_ids",
                    "map_vectors", "map_scores", "map_labels",
                    "trajs_3d", "trajs_score", "anchor_queue", "period",
                    "planning_score", "planning", "final_planning", "ego_period", "ego_anchor_queue"]

    torch.onnx.export(
        post_process_onnx,
        tuple(model_input),
        model_name,
        opset_version=16,
        input_names=input_names,
        output_names=output_names)
    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("\n{} is exported".format(model_name))


class PostProcessONNX(nn.Module):
    def __init__(self, post_process_module, data):
        super(PostProcessONNX, self).__init__()
        self.post_process_module = copy.deepcopy(post_process_module)
        self.data = data

    def forward(self, gt_ego_fut_cmd, model_outs):
        data = {
            'return_loss': False,
            'rescale': True,
            'img_metas': self.data['img_metas'],
            'timestamp': self.data['timestamp'],
            'projection_mat': self.data['projection_mat'],
            'image_wh': self.data['image_wh'],
            'gt_ego_fut_cmd': gt_ego_fut_cmd,
        }

        results = self.post_process_module(model_outs, data)
        return results


@MODELS.register_module()
class SparseDrive(MVXTwoStageDetector):
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
        super(SparseDrive, self).__init__(data_preprocessor=data_preprocessor,
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
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = MODELS.build(depth_branch)
        else:
            self.depth_branch = None

        self.save_onnx_model = save_onnx_model
        self.onnx_subnets = onnx_subnets

        self.onnx_model = None
        self.onnx_img_backbone = None
        self.onnx_pts_bbox_head = None

        # enable pts_bbox_head post_process export
        self.postprocess_export = False


    #@auto_fp16(apply_to=("img",), out_fp32=True)
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

    def _forward(self, inputs, data_samples, **kwargs):
        raise NotImplementedError('tensor mode is yet to add')


    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    """
    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output
    """

    def predict(self, inputs, data_samples,  **kwargs):
        img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]
        for var, name in [(batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # Re-foramt some meta infos
        metas={}
        metas['img_metas'] = batch_img_metas

        projection_mat = []
        image_wh = []
        timestamp = []
        gt_ego_fut_cmd = []
        for x in batch_img_metas:
            projection_mat.append(x["projection_mat"])
            image_wh.append(x["image_wh"])
            timestamp.append(torch.DoubleTensor([x["timestamp"]]))
            gt_ego_fut_cmd.append(x["gt_ego_fut_cmd"])
        projection_mat = torch.stack(projection_mat, dim=0).to(img.device)
        image_wh = torch.stack(image_wh, dim=0).to(img.device)
        timestamp = torch.cat(timestamp, dim=0).to(img.device)
        gt_ego_fut_cmd = torch.cat(gt_ego_fut_cmd, dim=0).to(img.device)

        metas["projection_mat"] = projection_mat
        metas["image_wh"]       = image_wh
        metas["timestamp"]      = timestamp
        metas["gt_ego_fut_cmd"] = gt_ego_fut_cmd
        # End re-format

        feature_maps = self.extract_feat(img)
        model_outs = self.pts_bbox_head(feature_maps, metas)

        # Export head.post_process
        if self.postprocess_export is True:
            post_process_module = self.pts_bbox_head.post_process
            post_process_onnx = PostProcessONNX(post_process_module, metas)
            post_process_onnx.eval()
            export_post_process(post_process_onnx, model_outs, metas)
            self.postprocess_export = False

        results = self.pts_bbox_head.post_process(model_outs, metas)
        bbox_list = [dict() for i in range(len(metas["img_metas"]))]
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
                        export_SparseDrive_subnets(self, inputs, data_samples, opset_version=18, **kwargs)
                    else:
                        export_SparseDrive(self, inputs, data_samples, opset_version=18, **kwargs)
                    self.save_onnx_model = False
                return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
