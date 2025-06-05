from .nuscenes_dataset import CustomNuScenesDataset
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
