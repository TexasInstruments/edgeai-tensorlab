# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.
from typing import List, Tuple, Union

from torch import Tensor
import torch
from mmdet.structures import OptSampleList, SampleList
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.utils.yolo_model_utils import AnchorConfig, Vec2Box, NMSConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class YOLOV9(SingleStageDetector):
    r"""Implementation of `Yolov3: An incremental improvement
    <https://arxiv.org/abs/1804.02767>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOX. Default: None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOX. Default: None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional):
            Model preprocessing config for processing the input data.
            it usually includes ``to_rgb``, ``pad_size_divisor``,
            ``pad_value``, ``mean`` and ``std``. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 aux_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
            )
        self.nms_cfg : NMSConfig = test_cfg['nms_cfg']
        self.aux_head = MODELS.build(aux_head)
    
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        # if self.with_neck:
        #     x = self.neck(x)
        return x
    

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        image_size = batch_inputs.shape[-2:]
        strides = self.bbox_head.strides
        device = batch_inputs.device

        vec2box = Vec2Box(image_size, strides, device)

        #features from backbone only
        backbone_feat = self.extract_feat(batch_inputs)

        x = self.neck(backbone_feat)

        losses = self.bbox_head.loss(self.aux_head, x, backbone_feat, batch_data_samples, vec2box)
        return losses
    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """


        image_size = batch_inputs.shape[-2:]
        strides = self.bbox_head.strides
        device = batch_inputs.device

        vec2box = Vec2Box(image_size, strides, device)

        # fake_input = torch.ones(1,3,640,640).to(batch_inputs.device)  ###
        # test_predicts = self.extract_feat(fake_input)  ###
        # test_predicts = self.neck(test_predicts) ###
        # x = test_predicts    ####


        x = self.extract_feat(batch_inputs)
        x = self.neck(x)

        results_list = self.bbox_head.predict(
            x, batch_data_samples, vec2box, self.nms_cfg, rescale=rescale)
        
        torch.save(results_list, 'work_dirs/onnx_exports/yolov9/tensors/predict.pt') ###

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
