from .nuscenes_dataset import CustomNuScenesDataset
from mmdet3d.datasets  import NuScenesDataset
from mmdet3d.datasets import PandaSetDataset
import torch
import numpy as np
from mmdet3d.registry import DATASETS
from mmengine.logging import print_log
import logging
import cv2
from mmdet3d.structures import LiDARInstance3DBoxes
try:
    from tools.dataset_converters.nuscenes_converter import get_2d_boxes
except:
    print('import error')


@DATASETS.register_module()
class CustomPandaSetDataset(CustomNuScenesDataset, PandaSetDataset):
    """
    def __init__(self, 
                 queue_length=4,
                 bev_size=(200, 200),
                 overlap_test=False,
                 *args,
                 **kwargs):
        super().__init__(queue_length,
                         bev_size,
                         overlap_test,
                         *args,
                         **kwargs)
    """

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if not self.label_mapping_changed:
            for k in self.label_mapping:
                self.label_mapping[k] = self.get_label_func(k)
            self.label_mapping_changed = True

        # CustomNuScenesDataset::full_init()
        super().full_init() 
        self.num_ins_per_cat = self.new_num_ins_per_cat


    def parse_ann_info(self, info):
        instances = info['instances']
        for instance in instances:
            instance['velocity'] = instance['velocity'] [::2] if self.load_type == 'mv_image_based' else instance['velocity'][:2]
        cam_instances = info.get('cam_instances',{})
        for name, instances in cam_instances.items():
            for instance in instances:
                instance['velocity'] = instance['velocity'] [::2]
        ann_info =  super(NuScenesDataset, self).parse_ann_info(info)

        if ann_info is not None:
            ann_info =  super()._filter_with_mask(ann_info)

            #gt_bboxes_3d, gt_labels_3d = info['ann_infos']
            #gt_bboxes_3d, gt_labels_3d = np.array(gt_bboxes_3d), np.array(gt_labels_3d)
            gt_bboxes_3d, _ = info['ann_infos']
            gt_bboxes_3d = np.array(gt_bboxes_3d)

            if len(gt_bboxes_3d) == 0:
                gt_bboxes_3d = np.zeros((0, 9), dtype=np.float32)

            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
            #ann_info['gt_labels_3d'] = gt_labels_3d
        else:
            # empty instance
            ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['attr_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.load_type in ['fov_image_based', 'mv_image_based']:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.new_num_ins_per_cat[label] += 1
        return ann_info
