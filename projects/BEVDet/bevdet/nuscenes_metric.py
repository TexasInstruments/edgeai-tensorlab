# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
import torch
from mmengine import Config, load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)

from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmdet3d.evaluation.metrics.nuscenes_metric import output_to_nusc_box


@METRICS.register_module()
class CustomNuScenesMetric(NuScenesMetric):
    """Nuscenes evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        jsonfile_prefix (str, optional): The prefix of json files including the
            file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        eval_version (str): Configuration version of evaluation.
            Defaults to 'detection_cvpr_2019'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 eval_version: str = 'detection_cvpr_2019',
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        super(CustomNuScenesMetric, self).__init__(
            data_root, ann_file, metric, modality, prefix, format_only,
            jsonfile_prefix, eval_version, collect_device, backend_args)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']
        # load annotations
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)['data_list']

        # sort data_infos
        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))

        result_dict, tmp_dir = self.format_results(results, classes,
                                                   self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            return metric_dict

        for metric in self.metrics:
            ap_dict = self.nus_evaluate(
                result_dict, classes=classes, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict


    def _format_lidar_bbox(self,
                           results: List[dict],
                           sample_idx_list: List[int],
                           classes: Optional[List[str]] = None,
                           jsonfile_prefix: Optional[str] = None) -> str:
        """Convert the results to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_idx],
                                             boxes, classes,
                                             self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print(f'Results writes to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path


def output_to_nusc_box(
        detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], np.ndarray or None]: List of standard
        NuScenesBoxes and attribute labels.
    """
    bbox3d = detection['bboxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    attrs = None
    if 'attr_labels' in detection:
        attrs = detection['attr_labels'].numpy()

    box_center = bbox3d.center.numpy()
    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if isinstance(bbox3d, LiDARInstance3DBoxes):
        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (
            # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    elif isinstance(bbox3d, CameraInstance3DBoxes):
        # our Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        nus_box_yaw = -box_yaw
        for i in range(len(bbox3d)):
            q1 = pyquaternion.Quaternion(
                axis=[0, 0, 1], radians=nus_box_yaw[i])
            q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            quat = q2 * q1
            velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes '
            'to standard NuScenesBoxes.')

    return box_list, attrs


# BEVDet register 3D info on the ego CS, not lidar CS.
# So we don't need to transform from Lidar CS to Ego CS. 
def lidar_nusc_box_to_global(
        info: dict, boxes: List[NuScenesBox], classes: List[str],
        eval_configs: DetectionConfig) -> List[NuScenesBox]:
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

    Returns:
        List[:obj:`DetectionConfig`]: List of standard NuScenesBoxes in the
        global coordinate.
    """
    box_list = []
    for box in boxes:
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list
