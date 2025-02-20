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


def tofloat(x):
    return x.astype(np.float32) if x is not None else None

@DATASETS.register_module()
class CustomPandaSetDataset(CustomNuScenesDataset, PandaSetDataset):
    def __init__(self, 
                 with_box2d=False,
                 sequential=False,
                 n_times=1,
                 speed_mode='relative_dis',
                 prev_only=False,
                 next_only=False,
                 train_adj_ids=None,
                 test_adj='prev',
                 test_adj_ids=None,
                 test_time_id=1,
                 max_interval=3,
                 min_interval=0,
                 fix_direction=False,
                 **kwargs):
        super().__init__(with_box2d,
                         sequential,
                         n_times,
                         speed_mode,
                         prev_only,
                         next_only,
                         train_adj_ids,
                         test_adj,
                         test_adj_ids,
                         test_time_id,
                         max_interval,
                         min_interval,
                         fix_direction,
                         **kwargs)