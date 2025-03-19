import io as sysio

import pyquaternion
import numpy as np
import os

from typing import Dict, List, Tuple

from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.utils.data_classes import Box as NuScenesBox


# Convert the output to NuScenesBox class
# https://github.com/open-mmlab/mmdetection3d/
# Based on output_to_nusc_box() 
def output_to_nusc_box(detection, task_name, bbox3d_type='lidar'):
    bbox3d = detection['bboxes_3d']
    scores = detection['scores_3d']
    labels = detection['labels_3d']
    attrs = None
    if 'attr_labels' in detection:
        attrs = detection['attr_labels']

    #dim = bbox3d.shape[-1]  # 9
    box_list = []

    if bbox3d_type == 'lidar':
        box_dims = bbox3d[:, 3:6]
        box_yaw = bbox3d[:, 6]
        center = bbox3d[:, :3]
        bottom_center = bbox3d[:, :3]
        gravity_center = np.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + bbox3d[:, 5] * 0.5

        # LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d[i, 7:9], 0.0)

            box = NuScenesBox(
                center[i] if task_name == 'BEVDet' else gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)

    else:
        #if origin != [0.5, 1.0, 0.5]:
        #    dst = np.array([0.5, 1.0, 0.5])
        #    src = np.array(origin)
        #    bbox3d[:, :3] += bbox3d[:, 3:6]*(dst-src)

        box_dims = bbox3d[:, 3:6]
        box_yaw = bbox3d[:, 6]
        center = bbox3d[:, :3]
        bottom_center = bbox3d[:, :3]
        gravity_center = np.zeros_like(bottom_center)
        gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]]
        gravity_center[:, 1] = bottom_center[:, 1] - bbox3d[:, 4] * 0.5
         
        # Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        nus_box_yaw = -box_yaw
        for i in range(len(bbox3d)):
            q1 = pyquaternion.Quaternion(
                axis=[0, 0, 1], radians=nus_box_yaw[i])
            q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            quat = q2 * q1
            velocity = (bbox3d[i, 7], 0.0, bbox3d[i, 8])
            box = NuScenesBox(
                gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)

    return box_list, attrs


# Convert the box from ego to global coordinate.
# https://github.com/open-mmlab/mmdetection3d/
# Based on lidar_nusc_box_to_global()
def lidar_nusc_box_to_global(data_info:dict, boxes: List[NuScenesBox],
    classes: List[str], eval_configs: DetectionConfig,  task_name) -> List[NuScenesBox]:

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        # BEVDet doesn't need it since it is already in ego coord system.
        if task_name != 'BEVDet':
            lidar2ego = np.array(data_info['lidar2ego'])
            box.rotate(
                pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
            box.translate(lidar2ego[:3, 3])

        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue

        # Move box to global coord system
        ego2global = np.array(data_info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)

    return box_list


# Convert the box from camera to global coordinate.
# https://github.com/open-mmlab/mmdetection3d/
# Based on cam_nusc_box_to_global()
def cam_nusc_box_to_global(
    info: dict,
    boxes: List[NuScenesBox],
    attrs: np.ndarray,
    classes: List[str],
    eval_configs: DetectionConfig,
    camera_type: str = 'CAM_FRONT',
    front_cam: bool = False
) -> Tuple[List[NuScenesBox], List[int]]:

    box_list = []
    attr_list = []
    for (box, attr) in zip(boxes, attrs):
        # Move box to ego vehicle coord system
        if front_cam is True:
            cam2ego = np.array(info['images'][camera_type]['front_cam2ego'])
        else:
            cam2ego = np.array(info['images'][camera_type]['cam2ego'])
        box.rotate(
            pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05, atol=1e-07))
        box.translate(cam2ego[:3, 3])
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
        attr_list.append(attr)
    return box_list, attr_list


# Convert the box from global to (FRONT) camera coordinate.
# https://github.com/open-mmlab/mmdetection3d/
# Based on global_nusc_box_to_cam()
def global_nusc_box_to_cam(info: dict, boxes: List[NuScenesBox],
                           classes: List[str],
                           eval_configs: DetectionConfig,
                           camera_type: str = 'CAM_FRONT') -> List[NuScenesBox]:
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        ego2global = np.array(info['ego2global'])
        box.translate(-ego2global[:3, 3])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05,
                                    atol=1e-07).inverse)
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to camera coord system
        cam2ego = np.array(info['images'][camera_type]['front_cam2ego'])
        box.translate(-cam2ego[:3, 3])
        box.rotate(
            pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05,
                                    atol=1e-07).inverse)
        box_list.append(box)
    return box_list


# Convert boxes from :obj:`NuScenesBox` to `CameraInstance3DBoxes` tensor array
# https://github.com/open-mmlab/mmdetection3d/
# Based on nusc_box_to_cam_box3d()
def nusc_box_to_cam_box3d(
    boxes: List[NuScenesBox]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
    velocity = np.array([b.velocity[0::2] for b in boxes]).reshape(-1, 2)

    # convert nusbox to cambox convention
    dims[:, [0, 1, 2]] = dims[:, [1, 2, 0]]
    rots = -rots

    boxes_3d = np.concatenate([locs, dims, rots, velocity], axis=1)
    #cam_boxes3d = CameraInstance3DBoxes(
    #    boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
    dst = np.array([0.5, 1.0, 0.5])
    src = np.array([0.5, 0.5, 0.5])
    boxes_3d[:, :3] += boxes_3d[:, 3:6]*(dst-src)

    scores = np.array([b.score for b in boxes])
    labels = np.array([b.label for b in boxes], dtype=np.long)
    nms_scores = np.zeros([scores.shape[0], 10 + 1], dtype=scores.dtype)
    indices = np.array(list(range(scores.shape[0])), dtype=labels.dtype)
    nms_scores[indices, labels] = scores

    return boxes_3d, nms_scores, labels


# Convert a rotated boxes in XYWHR format to XYXYR format.
# https://github.com/open-mmlab/mmdetection3d/
# Based on xywhr2xyxyr()
def xywhr2xyxyr(
        boxes_xywhr: np.ndarray) -> np.ndarray:
    boxes = np.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]

    return boxes


# Convert detection results to a list of numpy arrays.
# https://github.com/open-mmlab/mmdetection3d/
# Based on bbox3d2result()
def bbox3d2result(bboxes, scores, labels, attrs=None):
    result_dict = dict(
        bboxes_3d=bboxes,
        scores_3d=scores,
        labels_3d=labels)

    if attrs is not None:
        result_dict['attr_labels'] = attrs

    return result_dict
