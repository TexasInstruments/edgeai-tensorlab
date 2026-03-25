# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa

import glob
import os
import os.path as osp
import tempfile
from collections import OrderedDict
import copy
from typing import List, Union

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

import mmengine
from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS
from .coco import CocoDataset
from .api_wrappers import COCO


@DATASETS.register_module()
class ModelMakerDataset(CocoDataset):

    # CLASSES = None
    METAINFO = {
        'classes': ('human', 'trafficsign', 'vehicle'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)