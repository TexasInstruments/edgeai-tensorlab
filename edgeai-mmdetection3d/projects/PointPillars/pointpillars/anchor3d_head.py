# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from torch import Tensor

from mmdet3d.utils import InstanceList
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead
from mmdet3d.structures.det3d_data_sample import SampleList

from .voxelnet import dumpTensor
import os

@MODELS.register_module()
class PPAnchor3DHead(Anchor3DHead):
    """Anchor-based head for PointPillars
    """

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)

        if os.path.split(batch_input_metas[0]['lidar_path'])[1] == '00000x.bin':
            dump_bbox_head = True
        else:
            dump_bbox_head = False

        if dump_bbox_head is True:
            dumpTensor('conf.txt', outs[0][0][0])
            dumpTensor('loc.txt',  outs[1][0][0])
            dumpTensor('dir.txt',  outs[2][0][0])

        predictions = self.predict_by_feat(
            *outs, batch_input_metas=batch_input_metas, rescale=rescale)
        return predictions
