
from typing import Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.ops.nms import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)
from torch import Tensor
# from mmdeploy.mmcv.ops import multiclass_nms

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmdet.models.layers.yolo_layers import Detection 
from mmengine.model import BaseModule
from .base_dense_head import BaseDenseHead
from mmdet.models.utils.yolo_model_utils import LossConfig, MatcherConfig, Vec2Box, NMSConfig, PostProccess

@MODELS.register_module()
class YOLOV9Head(BaseDenseHead):
    def __init__(self, 
                num_classes: int = 80, 
                in_channels: Sequence[int] =[128, 192, 256],
                strides: Sequence[int] = (8, 16, 32),
                reg_max: int = 16,
                # norm_cfg: OptConfigType = dict(
                #     type='GN', num_groups=32, requires_grad=True),
                # train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                loss_yolo: ConfigType = dict(
                    type='YOLOLoss',
                    loss_cfg = LossConfig(
                        objective=dict(
                            BCELoss=0.5,
                            BoxLoss=7.5,
                            DFLoss=1.5
                        ),
                        aux=0.25,
                        matcher=MatcherConfig(
                            iou='CIoU',
                            topk=10,
                            factor=dict(
                                iou=6.0,
                                cls=0.5
                            )
                        )
                    )
                    ),
                    init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),
                **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.loss_config = loss_yolo
        self.in_channels = in_channels
        self.strides = strides
        self.test_cfg = test_cfg
        self.postprocess_class = PostProccess

        self.heads = nn.ModuleList(
            [Detection((in_channels[0], in_channel), num_classes, reg_max=reg_max) for in_channel in in_channels]
        )
        
    def forward(self, x_lists: List[Tensor]|List[List[Tensor]]) -> List[Tensor]:
        outs = []
        outs.append([head(x) for x, head in zip(x_lists, self.heads)])
        
        return outs

    def loss(self,aux_head, x: Tuple[Tensor], backbone_feat: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        predicts = self(x)
        predicts.extend(aux_head(backbone_feat))

        image_size = batch_data_samples[0].batch_input_shape
        strides = self.strides
        device = x[0].device
        vec2box = Vec2Box(image_size, strides, device)

        self.loss_config['vec2box'] = vec2box
        self.loss_yolo: nn.Module = MODELS.build(self.loss_config)

        # TODO:load targets
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        
        batch_size = len(batch_gt_instances)
        target_sizes = [item.labels.size(0) for item in batch_gt_instances]
        batch_targets = torch.zeros(batch_size, max(target_sizes), 5).to(batch_gt_instances[0].labels.device)
        batch_targets[:, :, 0] = -1
        for idx, target_size in enumerate(target_sizes):
            label = batch_gt_instances[idx].labels.view(-1,1)
            bbox = batch_gt_instances[idx].bboxes
            target = torch.cat([label,bbox], dim=-1)
            batch_targets[idx][:target_size] = target

        aux_predicts = vec2box(predicts[1])
        main_predicts = vec2box(predicts[0])

        return self.loss_by_feat(aux_predicts, main_predicts, batch_targets)

    def loss_by_feat(self, aux_predicts, main_predicts, batch_targets):

        aux_rate = self.loss_config['loss_cfg']['aux']
        iou_rate = self.loss_config['loss_cfg'].objective['BoxLoss']
        dfl_rate = self.loss_config['loss_cfg'].objective['DFLoss']
        cls_rate = self.loss_config['loss_cfg'].objective['BCELoss']

        aux_iou, aux_dfl, aux_cls = self.loss_yolo(aux_predicts, batch_targets)
        main_iou, main_dfl, main_cls = self.loss_yolo(main_predicts, batch_targets)

        loss_dict = {
            "loss_box": iou_rate * (aux_iou * aux_rate + main_iou),
            "loss_df": dfl_rate * (aux_dfl * aux_rate + main_dfl),
            "loss_bce": cls_rate * (aux_cls * aux_rate + main_cls),
        }

        return loss_dict


    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """

        image_size = batch_data_samples[0].batch_input_shape
        strides = self.strides
        device = x[0].device
        vec2box = Vec2Box(image_size, strides, device)

        outs = self(x)
        post_proccess = self.postprocess_class(vec2box)
        cls_scores, preds = post_proccess(outs)

        return self.predict_by_feat(cls_scores, preds, batch_data_samples, rescale=False)

    def predict_by_feat(self,
                cls_scores: Tensor,
                preds: Tensor,
                batch_data_samples: SampleList,
                cfg: Optional[ConfigDict] = None,
                rescale: bool = False,
                with_nms: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """

        assert len(cls_scores) == len(preds)
        cfg = self.test_cfg if cfg is None else cfg


        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            max_scores, labels = torch.max(cls_scores[img_id], 1)
            valid_mask = max_scores >= cfg.score_thr
            results = InstanceData(
                bboxes=preds[img_id][valid_mask],
                scores=max_scores[valid_mask],
                labels=labels[valid_mask])
            
            result_list.append(
                self._bbox_post_process(
                    results=results,
                    cfg=cfg,
                    rescale=rescale,
                    with_nms=with_nms,
                    img_meta=img_meta))

        return result_list

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        if rescale:
            assert img_meta.get('scale_factor') is not None
            results.bboxes /= results.bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
        return results





