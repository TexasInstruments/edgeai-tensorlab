# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

from mmdet.models import BaseDetector
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)
from mmdet3d.utils.typing_utils import (OptConfigType, OptInstanceList,
                                        OptMultiConfig)

from .onnx_network import PETR_export_model, DETR3D_export_model, BEVFormer_export_model

import torch
import copy

export_onnx = False 
model_to_export = 'BEVFormer' # 'PETR', 'DETR3D', 'BEVFormer'

def export_PETR(model, inputs=None, data_samples=None, mode=None, **kwargs):

        onnxModel = PETR_export_model(model.img_backbone,
                                      model.img_neck,
                                      model.pts_bbox_head)
        onnxModel.eval()

        # Should clone. Otherwise, when we run both export_model and self.predict,
        # we have error PETRHead forward() - Don't know why
        img = inputs['imgs'].clone()
        batch_img_metas = [ds.metainfo for ds in data_samples]

        batch_img_metas = onnxModel.add_lidar2img(img, batch_img_metas)
        onnxModel.prepare_data(img, batch_img_metas)
         
        modelInput = []
        modelInput.append(img)
        
        torch.onnx.export(onnxModel,
                          tuple(modelInput),
                         'petrv2.onnx',
                          opset_version=11,
                          verbose=False)

        print("!! ONNX model has been exported for PETR! !!\n\n")

def export_DETR3D(model, inputs=None, data_samples=None, mode=None, **kwargs):

        onnxModel = DETR3D_export_model(model.img_backbone, 
                                        model.img_neck,
                                        model.pts_bbox_head,
                                        model.add_pred_to_datasample)
        onnxModel.eval()

        # Should clone. Otherwise, when we run both export_model and self.predict,
        # we have error PETRHead forward() - Don't know why
        img = inputs['imgs'].clone()                
        batch_img_metas = [ds.metainfo for ds in data_samples]

        batch_img_metas = onnxModel.add_lidar2img(batch_img_metas)
        onnxModel.prepare_data(img, batch_img_metas)    

        modelInput = []
        modelInput.append(img)
        #modelInput.append(data_samples)

        torch.onnx.export(onnxModel,                        
                          tuple(modelInput),
                         'detr3d.onnx',
                          opset_version=16,
                          verbose=False)

        print("!! ONNX model has been exported for DETR3D! !!\n\n")


def export_BEVFormer(model, inputs=None, data_samples=None, mode=None, **kwargs):

        onnxModel = BEVFormer_export_model(model.img_backbone, 
                                           model.img_neck,
                                           model.pts_bbox_head,
                                           model.add_pred_to_datasample,
                                           model.video_test_mode)
        onnxModel = onnxModel.cpu()
        onnxModel.eval()

        # Should clone. Otherwise, when we run both export_model and self.predict,
        img = inputs['imgs'].clone()
        img = img.cpu()

        copy_data_samples = copy.deepcopy(data_samples)      
        batch_img_metas = [ds.metainfo for ds in copy_data_samples]

        # For temporal info
        if batch_img_metas[0]['scene_token'] != onnxModel.prev_frame_info['scene_token']:
            onnxModel.prev_frame_info['prev_bev'] = None

        onnxModel.prev_frame_info['scene_token'] = batch_img_metas[0]['scene_token']

        # do not use temporal information
        if not onnxModel.video_test_mode:
            onnxModel.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(batch_img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(batch_img_metas[0]['can_bus'][-1])

        if onnxModel.prev_frame_info['prev_bev'] is not None:
            batch_img_metas[0]['can_bus'][:3] -= onnxModel.prev_frame_info['prev_pos']
            batch_img_metas[0]['can_bus'][-1] -= onnxModel.prev_frame_info['prev_angle']
        else:
            batch_img_metas[0]['can_bus'][-1] = 0
            batch_img_metas[0]['can_bus'][:3] = 0

        # copy batch_img_metas
        onnxModel.prepare_data(img, batch_img_metas)

        modelInput = []
        modelInput.append(img)        
        modelInput.append(onnxModel.prev_frame_info['prev_bev'])

        torch.onnx.export(onnxModel,                        
                          tuple(modelInput),
                         'bevFormer.onnx',
                          opset_version=16,
                          verbose=False)

        onnxModel.prev_frame_info['prev_pos']   = tmp_pos
        onnxModel.prev_frame_info['prev_angle'] = tmp_angle
        #onnxModel.prev_frame_info['prev_bev']   = outs['bev_embed']

        print("!!ONNX model has been exported for BEVFormer!\n\n")



@MODELS.register_module()
class Base3DDetector(BaseDetector):
    """Base class for 3D detectors.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    def forward(self,
                inputs: Union[dict, List[dict]],
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
        inputs and data samples.

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
                annotation data of every samples. When it is a list[list], the
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
                if export_onnx == True:
                    if model_to_export == 'PETR':
                        export_PETR(self, inputs, data_samples, **kwargs)
                    elif model_to_export == 'DETR3D':
                        export_DETR3D(self, inputs, data_samples, **kwargs)
                    elif model_to_export == 'BEVFormer':
                        export_BEVFormer(self, inputs, data_samples, **kwargs)
                    else:
                        print("Unsupported model to export!")

                return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples
