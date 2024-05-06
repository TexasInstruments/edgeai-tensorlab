# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa

import glob
import os
import os.path as osp
import tempfile
from collections import OrderedDict
import json

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.utils import print_log

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class ModelMakerDataset(CocoDataset):

    CLASSES = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

