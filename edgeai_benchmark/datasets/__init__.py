# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings

from .. import constants

from .onnx_backend_dataset import *
from .tidl_unit_dataset import *

from .image_cls import *
from .image_seg import *
from .image_det import *

from .coco_det import *
from .coco_seg import *
from .imagenet import *
from .imagenetv2 import *
from .cityscapes import *
from .ade20k import *
from .voc_seg import *
from .nyudepthv2 import *
from .ycbv import *
from .modelmaker_datasets import *

from .coco_kpts import *
from .widerface_det import *

from .robokit_seg import *
from .robokit_visloc import *

from .kitti_2015 import *

try:
    from .nuscenes_dataset import *
except ImportError as e:
    warnings.warn(f'WARNING: nuscenes_dataset could not be imported - {str(e)}')
    nuscenes_dataset = None
    NuScenesDataset = None

try:
    from .pandaset_dataset import *
except ImportError as e:
    warnings.warn(f'WARNING: pandaset_dataset could not be imported - {str(e)}')
    pandaset_dataset = None
    PandaSetDataset = None

try:
    from .kitti_lidar_det import KittiLidar3D
except ImportError as e:
    warnings.warn(f'WARNING: kitti_lidar_det could not be imported - {str(e)}')
    KittiLidar3D = None


DATASET_CATEGORY_IMAGENET = 'imagenet'
DATASET_CATEGORY_COCO = 'coco'
DATASET_CATEGORY_WIDERFACE = 'widerface'
DATASET_CATEGORY_ADE20K32 = 'ade20k32'
DATASET_CATEGORY_ADE20K = 'ade20k'
DATASET_CATEGORY_VOC2012 = 'voc2012'
DATASET_CATEGORY_COCOSEG21 = 'cocoseg21'
DATASET_CATEGORY_COCOKPTS = 'cocokpts'
DATASET_CATEGORY_NYUDEPTHV2 = 'nyudepthv2'
DATASET_CATEGORY_CITYSCAPES = 'cityscapes'
DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD = 'ti-robokit_semseg_zed1hd'
DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD = 'ti-robokit_visloc_zed1hd'
DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS = 'kitti_lidar_det_1class'
DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS = 'kitti_lidar_det_3class'
DATASET_CATEGORY_KITTI_2015 = 'kitti_2015'
DATASET_CATEGORY_YCBV = 'ycbv'
DATASET_CATEGORY_NUSCENES_FRAME = 'nuscenes_frame'
DATASET_CATEGORY_NUSCENES_MV_IMAGE = 'nuscenes_mv_image'
DATASET_CATEGORY_PANDASET_FRAME = 'pandaset_frame'
DATASET_CATEGORY_PANDASET_MV_IMAGE = 'pandaset_mv_image'

dataset_info_dict = {
    #------------------------image classification datasets--------------------------#
    # Original ImageNet
    'imagenet':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetCls, 'size':50000, 'split':'val'},
    'imagenetv1':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetCls, 'size':50000, 'split':'val'},
    # ImageNetV2 as explained in imagenet_v2.py
    'imagenetv2c':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetV2C, 'size':10000, 'split':'val'},
    'imagenetv2b':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetV2B, 'size':10000, 'split':'val'},
    'imagenetv2a':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetV2A, 'size':10000, 'split':'val'},
    #------------------------object detection datasets--------------------------#
    'coco': {'task_type':'detection', 'category':DATASET_CATEGORY_COCO, 'type':COCODetection, 'size':5000, 'split':'val2017'},
    'widerface': {'task_type':'detection', 'category':DATASET_CATEGORY_WIDERFACE, 'type':WiderFaceDetection, 'size':3226, 'split':'val'},
    #------------------------semantic segmentation datasets--------------------------#
    'ade20k32': {'task_type':'segmentation', 'category':DATASET_CATEGORY_ADE20K32, 'type':ADE20KSegmentation, 'size':2000, 'split':'validation'},
    'ade20k': {'task_type':'segmentation', 'category':DATASET_CATEGORY_ADE20K, 'type':ADE20KSegmentation, 'size':2000, 'split':'validation'},
    'voc2012': {'task_type':'segmentation', 'category':DATASET_CATEGORY_VOC2012, 'type':VOC2012Segmentation, 'size':1449, 'split':'val'},
    'cocoseg21': {'task_type':'segmentation', 'category':DATASET_CATEGORY_COCOSEG21, 'type':COCOSegmentation, 'size':5000, 'split':'val2017'},
    'ti-robokit_semseg_zed1hd': {'task_type':'segmentation', 'category':DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD, 'type':RobokitSegmentation, 'size':49, 'split':'val'},
    'ti-robokit_visloc_zed1hd': {'task_type':'visual_localization', 'category':DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD, 'type':RobokitVisualLocalization, 'size':49, 'split':'val'},
    #------------------------human pose estimation datasets--------------------------#
    'cocokpts': {'task_type':'keypoint_detection', 'category':DATASET_CATEGORY_COCOKPTS, 'type':COCOKeypoints, 'size':5000, 'split':'val2017'},
    #------------------------depth estimation datasets--------------------------#
    'nyudepthv2': {'task_type':'depth_estimation', 'category':DATASET_CATEGORY_NYUDEPTHV2, 'type':NYUDepthV2, 'size':654, 'split':'val'},
    #------------------------object 6d pose estimation datasets--------------------------#
    'ycbv': {'task_type':'object_6d_pose_estimation', 'category':DATASET_CATEGORY_YCBV, 'type': YCBV, 'size':900, 'split':'test'},
    #----------------------- BEV Detection datasets for PandaSet --------------------------#
    'pandaset_frame': {'task_type':'bev-detection', 'category':DATASET_CATEGORY_PANDASET_FRAME, 'type':PandaSetDataset, 'size':404, 'split':'val'},
    'pandaset_mv_image': {'task_type':'bev-detection', 'category':DATASET_CATEGORY_PANDASET_MV_IMAGE, 'type':PandaSetDataset, 'size':404, 'split':'val'},    
 }


dataset_info_dict_experimental = {
    #------------------------semantic segmentation datasets--------------------------#
    'cityscapes': {'task_type':'segmentation', 'category':DATASET_CATEGORY_CITYSCAPES, 'type':CityscapesSegmentation, 'size':500, 'split':'val'},
    #------------------------3D OD datasets--------------------------#
    'kitti_lidar_det_1class': {'task_type':'3d-detection', 'category':DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS, 'type':KittiLidar3D, 'size':3769, 'split':'val'},
    'kitti_lidar_det_3class': {'task_type': '3d-detection', 'category': DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS,'type': KittiLidar3D, 'size': 3769, 'split': 'val'},
    #----------------------- Stereo disparity datasets--------------------------#
    'kitti_2015': {'task_type':'stereo-disparity', 'category':DATASET_CATEGORY_KITTI_2015, 'type':Kitti2015, 'size':159, 'split':'training'},
    #----------------------- BEV Detection datasets for NuScenes --------------------------#
    'nuscenes_frame': {'task_type':'bev-detection', 'category':DATASET_CATEGORY_NUSCENES_FRAME, 'type':NuScenesDataset, 'size':404, 'split':'val'},
    'nuscenes_mv_image': {'task_type':'bev-detection', 'category':DATASET_CATEGORY_NUSCENES_MV_IMAGE, 'type':NuScenesDataset, 'size':404, 'split':'val'},
}


def get_dataset_info_dict(settings):
    dset_info_dict = dataset_info_dict.copy()
    if settings is not None and settings.experimental_models:
        dset_info_dict.update(dataset_info_dict_experimental)
    #
    return dset_info_dict


def get_dataset_categories(settings=None, task_type=None):
    dset_info_dict = get_dataset_info_dict(settings)
    # we are picking category instead of the actual dataset name/variant.
    # the actual dataset to be used cal is selected in get_dataset()
    if task_type is not None:
        dataset_names = [value['category'] for key,value in dset_info_dict.items() if value['task_type'] == task_type]
    else:
        dataset_names = [value['category'] for key,value in dset_info_dict.items()]
    #
    # make it unique - set() is unordered - so use dict.fromkeys()
    dataset_categories = list(dict.fromkeys(dataset_names).keys())
    return dataset_categories


def get_dataset_names(settings, task_type=None):
    print(utils.log_color('WARNING', 'name change', f'please use datasets.get_dataset_categories() '
                                                    f'instead of datasets.get_dataset_names()'))
    dataset_categories = get_dataset_categories(settings, task_type)
    return dataset_categories


def initialize_datasets(settings):
    dataset_categories = get_dataset_categories(settings)
    settings.dataset_cache = {
        ds_category: {'calibration_dataset':ds_category, 'input_dataset':ds_category, 'dataset_init':False} \
        for ds_category in dataset_categories
    }
    return settings.dataset_cache


def _get_additional_datasets(settings, download=False, dataset_list=None):
    dataset_cache = settings.dataset_cache
    dset_info_dict = get_dataset_info_dict(settings)
    dataset_list = dataset_list or get_dataset_categories(settings)
    # this is the max number of calibration images required for device
    #calibration_frames_nx = (settings.calibration_frames * (self.calibration_iterations_factor or 1.0))
    calibration_frames_nx = int(settings.calibration_frames * constants.CALIBRATION_ITERATIONS_FACTOR_NX)

    dataset_variant = 'imagenetv2c'
    if check_dataset_load(settings, dataset_variant) and (dataset_variant in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_IMAGENET} variant:{dataset_variant}"))
        # dataset settings
        imagenet_dataset_dict = dset_info_dict[dataset_variant]
        ImageNetDataSetType = imagenet_dataset_dict['type']
        imagenet_split = imagenet_dataset_dict['split']
        num_imgs = imagenet_dataset_dict['size']
        # the cfg to be used to construct the dataset class
        imagenet_cls_calib_cfg = dict(
            path=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}',
            split=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}.txt',
            shuffle=True,
            num_frames=min(num_imgs,calibration_frames_nx),
            name=dataset_variant)
        imagenet_cls_val_cfg = dict(
            path=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}',
            split=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}.txt',
            shuffle=True,
            num_frames=min(settings.num_frames,num_imgs),
            name=dataset_variant)
        # what is provided is mechanism to select one of the imagenet variants
        # but only one is selected and assigned to the key imagenet
        # all the imagenet models will use this variant.
        print(f'Value of download here: {download}')# TODO: LUKE remove
        try:
            dataset_cache[DATASET_CATEGORY_IMAGENET]['calibration_dataset'] = ImageNetDataSetType(**imagenet_cls_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_IMAGENET]['input_dataset'] = ImageNetDataSetType(**imagenet_cls_val_cfg, download=False)
        except Exception as e:
            dataset_cache[DATASET_CATEGORY_IMAGENET]['calibration_dataset'] = DATASET_CATEGORY_IMAGENET
            dataset_cache[DATASET_CATEGORY_IMAGENET]['input_dataset'] = DATASET_CATEGORY_IMAGENET
            print(f'Exception occurred: {str(e)}')
            print('ImageNet Classification dataset could not be loaded')
        #
    #

def get_datasets(settings, download=False, dataset_list=None):
    dataset_cache = settings.dataset_cache
    dset_info_dict = get_dataset_info_dict(settings)
    dataset_list = dataset_list or get_dataset_categories(settings)
    # this is the max number of calibration images required for device
    #calibration_frames_nx = (settings.calibration_frames * (self.calibration_iterations_factor or 1.0))
    calibration_frames_nx = int(settings.calibration_frames * constants.CALIBRATION_ITERATIONS_FACTOR_NX)

    if check_dataset_load(settings, DATASET_CATEGORY_IMAGENET) and (DATASET_CATEGORY_IMAGENET in dataset_list):
        dataset_variant = settings.dataset_type_dict[DATASET_CATEGORY_IMAGENET] if \
            settings.dataset_type_dict is not None else DATASET_CATEGORY_IMAGENET
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_IMAGENET} variant:{dataset_variant}"))
        # dataset settings
        imagenet_dataset_dict = dset_info_dict[dataset_variant]
        ImageNetDataSetType = imagenet_dataset_dict['type']
        imagenet_split = imagenet_dataset_dict['split']
        num_imgs = imagenet_dataset_dict['size']
        # the cfg to be used to construct the dataset class
        imagenet_cls_calib_cfg = dict(
            path=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}',
            split=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}.txt',
            shuffle=True,
            num_frames=min(num_imgs,calibration_frames_nx),
            name=dataset_variant)
        imagenet_cls_val_cfg = dict(
            path=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}',
            split=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}.txt',
            shuffle=True,
            num_frames=min(settings.num_frames,num_imgs),
            name=dataset_variant)
        # what is provided is mechanism to select one of the imagenet variants
        # but only one is selected and assigned to the key imagenet
        # all the imagenet models will use this variant.
        print(f'Value of download here: {download}')# TODO: LUKE remove
        try:
            dataset_cache[DATASET_CATEGORY_IMAGENET]['calibration_dataset'] = ImageNetDataSetType(**imagenet_cls_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_IMAGENET]['input_dataset'] = ImageNetDataSetType(**imagenet_cls_val_cfg, download=False)
        except Exception as e:
            dataset_cache[DATASET_CATEGORY_IMAGENET]['calibration_dataset'] = DATASET_CATEGORY_IMAGENET
            dataset_cache[DATASET_CATEGORY_IMAGENET]['input_dataset'] = DATASET_CATEGORY_IMAGENET
            print(f'Exception occurred: {str(e)}')
            print('ImageNet Classification dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_COCOKPTS) and (DATASET_CATEGORY_COCOKPTS in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_COCOKPTS} variant:{DATASET_CATEGORY_COCOKPTS}"))
        filter_imgs = True
        coco_kpts_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=min(5000,calibration_frames_nx),
            name=DATASET_CATEGORY_COCOKPTS,
            filter_imgs=filter_imgs)
        coco_kpts_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(settings.num_frames,5000),
            name=DATASET_CATEGORY_COCOKPTS,
            filter_imgs=filter_imgs)
        try:
            dataset_cache[DATASET_CATEGORY_COCOKPTS]['calibration_dataset'] = COCOKeypoints(**coco_kpts_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_COCOKPTS]['input_dataset'] = COCOKeypoints(**coco_kpts_val_cfg, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_COCOKPTS]['calibration_dataset'] = DATASET_CATEGORY_COCOKPTS
            dataset_cache[DATASET_CATEGORY_COCOKPTS]['input_dataset'] = DATASET_CATEGORY_COCOKPTS
            print('COCO Keypoint Detection dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_YCBV) and (DATASET_CATEGORY_YCBV in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_YCBV} variant:{DATASET_CATEGORY_YCBV}"))
        filter_imgs = True
        ycbv_calib_cfg = dict(
            path=f'{settings.datasets_path}/ycbv',
            split='test',
            shuffle=True,
            num_frames=min(900,calibration_frames_nx),
            name=DATASET_CATEGORY_YCBV,
            filter_imgs=filter_imgs)
        ycbv_val_cfg = dict(
            path=f'{settings.datasets_path}/ycbv',
            split='test',
            shuffle=False,
            num_frames=min(settings.num_frames,900),
            name=DATASET_CATEGORY_YCBV,
            filter_imgs=filter_imgs)
        try:
            dataset_cache[DATASET_CATEGORY_YCBV]['calibration_dataset'] = YCBV(**ycbv_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_YCBV]['input_dataset'] = YCBV(**ycbv_val_cfg, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_YCBV]['calibration_dataset'] = DATASET_CATEGORY_YCBV
            dataset_cache[DATASET_CATEGORY_YCBV]['input_dataset'] = DATASET_CATEGORY_YCBV
            print('YCBV 6D-Pose dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_COCO) and (DATASET_CATEGORY_COCO in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_COCO} variant:{DATASET_CATEGORY_COCO}"))
        coco_det_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=min(5000,calibration_frames_nx),
            name=DATASET_CATEGORY_COCO)
        coco_det_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=False, # can be set to True as well, if needed
            num_frames=min(settings.num_frames,5000),
            name=DATASET_CATEGORY_COCO)
        try:
            dataset_cache[DATASET_CATEGORY_COCO]['calibration_dataset'] = COCODetection(**coco_det_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_COCO]['input_dataset'] = COCODetection(**coco_det_val_cfg, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_COCO]['calibration_dataset'] = DATASET_CATEGORY_COCO
            dataset_cache[DATASET_CATEGORY_COCO]['input_dataset'] = DATASET_CATEGORY_COCO
            print('COCO Detection dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_WIDERFACE) and (DATASET_CATEGORY_WIDERFACE in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_WIDERFACE} variant:{DATASET_CATEGORY_WIDERFACE}"))
        widerface_det_calib_cfg = dict(
            path=f'{settings.datasets_path}/widerface',
            split='val',
            shuffle=True,
            num_frames=min(3226,calibration_frames_nx),
            name=DATASET_CATEGORY_WIDERFACE)
        widerface_det_val_cfg = dict(
            path=f'{settings.datasets_path}/widerface',
            split='val',
            shuffle=False, # can be set to True as well, if needed
            num_frames=min(settings.num_frames,3226),
            name=DATASET_CATEGORY_WIDERFACE)
        try:
            dataset_cache[DATASET_CATEGORY_WIDERFACE]['calibration_dataset'] = WiderFaceDetection(**widerface_det_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_WIDERFACE]['input_dataset'] = WiderFaceDetection(**widerface_det_val_cfg, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_WIDERFACE]['calibration_dataset'] = DATASET_CATEGORY_WIDERFACE
            dataset_cache[DATASET_CATEGORY_WIDERFACE]['input_dataset'] = DATASET_CATEGORY_WIDERFACE
            print('WIDERFACE Detection dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_COCOSEG21) and (DATASET_CATEGORY_COCOSEG21 in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_COCOSEG21} variant:{DATASET_CATEGORY_COCOSEG21}"))
        cocoseg21_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=min(5000,calibration_frames_nx),
            name=DATASET_CATEGORY_COCOSEG21)
        cocoseg21_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=min(settings.num_frames,5000),
            name=DATASET_CATEGORY_COCOSEG21)
        try:
            dataset_cache[DATASET_CATEGORY_COCOSEG21]['calibration_dataset'] = COCOSegmentation(**cocoseg21_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_COCOSEG21]['input_dataset'] = COCOSegmentation(**cocoseg21_val_cfg, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_COCOSEG21]['calibration_dataset'] = DATASET_CATEGORY_COCOSEG21
            dataset_cache[DATASET_CATEGORY_COCOSEG21]['input_dataset'] = DATASET_CATEGORY_COCOSEG21
            print('COCO 21 class Segmentation dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_ADE20K) and (DATASET_CATEGORY_ADE20K in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_ADE20K} variant:{DATASET_CATEGORY_ADE20K}"))
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(2000,calibration_frames_nx),
            name=DATASET_CATEGORY_ADE20K)
        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames,2000),
            name=DATASET_CATEGORY_ADE20K)
        try:
            dataset_cache[DATASET_CATEGORY_ADE20K]['calibration_dataset'] = ADE20KSegmentation(**ade20k_seg_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_ADE20K]['input_dataset'] = ADE20KSegmentation(**ade20k_seg_val_cfg, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_ADE20K]['calibration_dataset'] = DATASET_CATEGORY_ADE20K
            dataset_cache[DATASET_CATEGORY_ADE20K]['input_dataset'] = DATASET_CATEGORY_ADE20K
            print('ADE20K Segmentation dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_ADE20K32) and (DATASET_CATEGORY_ADE20K32 in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_ADE20K32} variant:{DATASET_CATEGORY_ADE20K32}"))
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(2000,calibration_frames_nx),
            name=DATASET_CATEGORY_ADE20K32)
        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames,2000),
            name=DATASET_CATEGORY_ADE20K32)
        try:
            dataset_cache[DATASET_CATEGORY_ADE20K32]['calibration_dataset'] = ADE20KSegmentation(**ade20k_seg_calib_cfg, num_classes=32, download=download)
            dataset_cache[DATASET_CATEGORY_ADE20K32]['input_dataset'] = ADE20KSegmentation(**ade20k_seg_val_cfg, num_classes=32, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_ADE20K32]['calibration_dataset'] = DATASET_CATEGORY_ADE20K32
            dataset_cache[DATASET_CATEGORY_ADE20K32]['input_dataset'] = DATASET_CATEGORY_ADE20K32
            print('ADE20K32 Segmentation dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_VOC2012) and (DATASET_CATEGORY_VOC2012 in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_VOC2012} variant:{DATASET_CATEGORY_VOC2012}"))
        voc_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=min(1449,calibration_frames_nx),
            name=DATASET_CATEGORY_VOC2012)
        voc_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=min(settings.num_frames,1449),
            name=DATASET_CATEGORY_VOC2012)
        try:
            dataset_cache[DATASET_CATEGORY_VOC2012]['calibration_dataset'] = VOC2012Segmentation(**voc_seg_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_VOC2012]['input_dataset'] = VOC2012Segmentation(**voc_seg_val_cfg, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_VOC2012]['calibration_dataset'] = DATASET_CATEGORY_VOC2012
            dataset_cache[DATASET_CATEGORY_VOC2012]['input_dataset'] = DATASET_CATEGORY_VOC2012
            print('VOC2012 Segmentation dataset could not be loaded')
        #
    #
    if check_dataset_load(settings, DATASET_CATEGORY_NYUDEPTHV2) and (DATASET_CATEGORY_NYUDEPTHV2 in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_NYUDEPTHV2} variant:{DATASET_CATEGORY_NYUDEPTHV2}"))
        filter_imgs = False
        nyudepthv2_calib_cfg = dict(
            path=f'{settings.datasets_path}/nyudepthv2',
            split='val',
            shuffle=True,
            num_frames=min(654,calibration_frames_nx),
            name=DATASET_CATEGORY_NYUDEPTHV2)
        nyudepthv2_val_cfg = dict(
            path=f'{settings.datasets_path}/nyudepthv2',
            split='val',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(settings.num_frames,654),
            name=DATASET_CATEGORY_NYUDEPTHV2)

        try:
            dataset_cache[DATASET_CATEGORY_NYUDEPTHV2]['calibration_dataset'] = NYUDepthV2(**nyudepthv2_calib_cfg, download=download)
            dataset_cache[DATASET_CATEGORY_NYUDEPTHV2]['input_dataset'] = NYUDepthV2(**nyudepthv2_val_cfg, download=False)
        except:
            dataset_cache[DATASET_CATEGORY_NYUDEPTHV2]['calibration_dataset'] = DATASET_CATEGORY_NYUDEPTHV2
            dataset_cache[DATASET_CATEGORY_NYUDEPTHV2]['input_dataset'] = DATASET_CATEGORY_NYUDEPTHV2
            print('NYUDepthV2 dataset could not be loaded')
        #
    #

    if check_dataset_load(settings, DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD) and (DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD} variant:{DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD}"))
        dataset_calib_cfg = dict(
            path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
            split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/train_img_gt_pair.txt',
            num_classes=19,
            shuffle=True,
            num_frames=min(150,calibration_frames_nx),
            name=DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD
        )

        # dataset parameters for actual inference
        dataset_val_cfg = dict(
            path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
            split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/val_img_gt_pair.txt',
            num_classes=19,
            shuffle=True,
            num_frames=min(settings.num_frames,49),
            name=DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD
        )

        try:
            dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD]['calibration_dataset'] = RobokitSegmentation(**dataset_calib_cfg, download=True)
            dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD]['input_dataset'] = RobokitSegmentation(**dataset_val_cfg, download=True)
        except:
            dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD]['calibration_dataset'] = DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD
            dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD]['input_dataset'] = DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD
            print('Robokit Segmentation dataset could not be loaded')
        #
    #

    if check_dataset_load(settings, DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD) and (DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD} variant:{DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD}"))
        dataset_calib_cfg = dict(
            path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
            split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/train_img_gt_pair.txt',
            num_classes=19,
            shuffle=True,
            num_frames=min(150,calibration_frames_nx),
            name=DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD
        )

        # dataset parameters for actual inference
        dataset_val_cfg = dict(
            path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
            split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/val_img_gt_pair.txt',
            num_classes=19,
            shuffle=True,
            num_frames=min(settings.num_frames,49),
            name=DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD
        )

        try:
            dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD]['calibration_dataset'] = RobokitVisualLocalization(**dataset_calib_cfg, download=True)
            dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD]['input_dataset'] = RobokitVisualLocalization(**dataset_val_cfg, download=True)
        except:
            dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD]['calibration_dataset'] = DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD
            dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD]['input_dataset'] = DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD
            print('Robokit VisLoc dataset could not be loaded')
        #
    #

    if check_dataset_load(settings, DATASET_CATEGORY_PANDASET_FRAME) and (DATASET_CATEGORY_PANDASET_FRAME in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_PANDASET_FRAME} variant:{DATASET_CATEGORY_PANDASET_FRAME}"))
        dataset_calib_cfg = dict(
            path=f'{settings.datasets_path}/pandaset/',
            split='train',
            num_classes=3,
            load_type='frame_based',
            shuffle=False,
            num_frames=min(81, calibration_frames_nx),
            name=DATASET_CATEGORY_PANDASET_FRAME)

        # dataset parameters for actual inference
        dataset_val_cfg = dict(
            path=f'{settings.datasets_path}/pandaset/',
            split='val',
            num_classes=3,
            load_type='frame_based',
            shuffle=False,
            num_frames=min(settings.num_frames, 81),
            name=DATASET_CATEGORY_PANDASET_FRAME)

        if dataset_cache[DATASET_CATEGORY_PANDASET_FRAME]['dataset_init'] is False:
            try:
                # load pandaset dataset first
                ps = load_pandaset(dataset_calib_cfg['path'])
                dataset_cache[DATASET_CATEGORY_PANDASET_FRAME]['calibration_dataset'] = \
                    PandaSetDataset(**dataset_calib_cfg, ps=ps, download=False, read_anno=False)
                dataset_cache[DATASET_CATEGORY_PANDASET_FRAME]['input_dataset'] = \
                    PandaSetDataset(**dataset_val_cfg, ps=ps, download=False, read_anno=True)
                dataset_cache[DATASET_CATEGORY_PANDASET_FRAME]['dataset_init'] =  True
            except Exception as message:
                dataset_cache[DATASET_CATEGORY_PANDASET_FRAME]['calibration_dataset'] = DATASET_CATEGORY_PANDASET_FRAME
                dataset_cache[DATASET_CATEGORY_PANDASET_FRAME]['input_dataset'] = DATASET_CATEGORY_PANDASET_FRAME
                print(f'PandaSet dataset loader could not be created: {message}')
        #
    #

    if check_dataset_load(settings, DATASET_CATEGORY_PANDASET_MV_IMAGE) and (DATASET_CATEGORY_PANDASET_MV_IMAGE in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_PANDASET_MV_IMAGE} variant:{DATASET_CATEGORY_PANDASET_MV_IMAGE}"))
        dataset_calib_cfg = dict(
            path=f'{settings.datasets_path}/pandaset/',
            split='train',
            num_classes=3,
            load_type='mv_image_based',
            shuffle=False,
            num_frames=min(50, calibration_frames_nx),
            name=DATASET_CATEGORY_PANDASET_MV_IMAGE)

        # dataset parameters for actual inference
        dataset_val_cfg = dict(
            path=f'{settings.datasets_path}/pandaset/',
            split='val',
            num_classes=3,
            load_type='mv_image_based',
            shuffle=False,
            num_frames=min(settings.num_frames, 456),
            name=DATASET_CATEGORY_PANDASET_MV_IMAGE)

        # To revisit
        if dataset_cache[DATASET_CATEGORY_PANDASET_MV_IMAGE]['dataset_init'] is False:
            try:
                # load pandaset dataset first
                ps = load_pandaset(dataset_calib_cfg['path'])
                dataset_cache[DATASET_CATEGORY_PANDASET_MV_IMAGE]['calibration_dataset'] = \
                    PandaSetDataset(**dataset_calib_cfg, ps=ps, download=False, read_anno=False)
                dataset_cache[DATASET_CATEGORY_PANDASET_MV_IMAGE]['input_dataset'] = \
                    PandaSetDataset(**dataset_val_cfg, ps=ps, download=False, read_anno=True)
                dataset_cache[DATASET_CATEGORY_PANDASET_MV_IMAGE]['dataset_init'] =  True
            except Exception as message:
                dataset_cache[DATASET_CATEGORY_PANDASET_MV_IMAGE]['calibration_dataset'] = DATASET_CATEGORY_PANDASET_MV_IMAGE
                dataset_cache[DATASET_CATEGORY_PANDASET_MV_IMAGE]['input_dataset'] = DATASET_CATEGORY_PANDASET_MV_IMAGE
                print(f'PandaSet dataset loader could not be created: {message}')
        #
    #

    # the following are datasets cannot be downloaded automatically
    # put it under the condition of experimental_models
    if settings.experimental_models:
        if check_dataset_load(settings, DATASET_CATEGORY_CITYSCAPES) and (DATASET_CATEGORY_CITYSCAPES in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_CITYSCAPES} variant:{DATASET_CATEGORY_CITYSCAPES}"))
            cityscapes_seg_calib_cfg = dict(
                path=f'{settings.datasets_path}/cityscapes',
                split='val',
                shuffle=True,
                num_frames=min(500,calibration_frames_nx),
                name=DATASET_CATEGORY_CITYSCAPES)
            cityscapes_seg_val_cfg = dict(
                path=f'{settings.datasets_path}/cityscapes',
                split='val',
                shuffle=True,
                num_frames=min(settings.num_frames,500),
                name=DATASET_CATEGORY_CITYSCAPES)
            try:
                dataset_cache[DATASET_CATEGORY_CITYSCAPES]['calibration_dataset'] = CityscapesSegmentation(**cityscapes_seg_calib_cfg, download=False)
                dataset_cache[DATASET_CATEGORY_CITYSCAPES]['input_dataset'] = CityscapesSegmentation(**cityscapes_seg_val_cfg, download=False)
            except:
                dataset_cache[DATASET_CATEGORY_CITYSCAPES]['calibration_dataset'] = DATASET_CATEGORY_CITYSCAPES
                dataset_cache[DATASET_CATEGORY_CITYSCAPES]['input_dataset'] = DATASET_CATEGORY_CITYSCAPES
                print('Cityscapes Segmentation dataset could not be loaded')
            #
        #
        if check_dataset_load(settings, DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS) and (DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS} variant:{DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS}"))
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=3,
                shuffle=False,
                num_frames=min(3769,calibration_frames_nx),
                name=DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS)

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=3,
                shuffle=False,
                num_frames=min(settings.num_frames,3769),
                name=DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS)
            try:
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS]['calibration_dataset'] = KittiLidar3D(**dataset_calib_cfg, download=False, read_anno=False)
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS]['input_dataset'] = KittiLidar3D(**dataset_val_cfg, download=False, read_anno=True)
            except Exception as message:
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS]['calibration_dataset'] = DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS]['input_dataset'] = DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS
                print(f'KittiLidar3D 3 class dataset loader could not be created: {message}')
            #
        #
        if check_dataset_load(settings, DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS) and (DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS} variant:{DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS}"))
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=1,
                shuffle=False,
                num_frames=min(3769,calibration_frames_nx),
                name=DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS)

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=1,
                shuffle=False,
                num_frames=min(settings.num_frames,3769),
                name=DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS)
            try:
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS]['calibration_dataset'] = KittiLidar3D(**dataset_calib_cfg, download=False, read_anno=False)
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS]['input_dataset'] = KittiLidar3D(**dataset_val_cfg, download=False, read_anno=True)
            except Exception as message:
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS]['calibration_dataset'] = DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS]['input_dataset'] = DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS
                print(f'KittiLidar3D 1 class dataset loader could not be created: {message}')
            #
        #

        if check_dataset_load(settings, DATASET_CATEGORY_KITTI_2015) and (DATASET_CATEGORY_KITTI_2015 in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_KITTI_2015} variant:{DATASET_CATEGORY_KITTI_2015}"))
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/kitti_2015/',
                split='training',
                shuffle=False,
                max_disp=192,
                num_frames=min(50,calibration_frames_nx))

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/kitti_2015/',
                split='training',
                shuffle=False,
                max_disp=192,
                num_frames=min(settings.num_frames,50))
            try:
                dataset_cache[DATASET_CATEGORY_KITTI_2015]['calibration_dataset'] = Kitti2015(**dataset_calib_cfg, download=False)
                dataset_cache[DATASET_CATEGORY_KITTI_2015]['input_dataset'] = Kitti2015(**dataset_val_cfg, download=False)
            except Exception as message:
                dataset_cache[DATASET_CATEGORY_KITTI_2015]['calibration_dataset'] = DATASET_CATEGORY_KITTI_2015
                dataset_cache[DATASET_CATEGORY_KITTI_2015]['input_dataset'] = DATASET_CATEGORY_KITTI_2015
                print(f'Kitti 2015 dataset loader could not be created: {message}')
            #
        #

        if check_dataset_load(settings, DATASET_CATEGORY_NUSCENES_FRAME) and (DATASET_CATEGORY_NUSCENES_FRAME in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_NUSCENES_FRAME} variant:{DATASET_CATEGORY_NUSCENES_FRAME}"))
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/nuscenes/',
                split='train',
                num_classes=10,
                load_type='frame_based',
                shuffle=False,
                num_frames=min(81, calibration_frames_nx),
                name=DATASET_CATEGORY_NUSCENES_FRAME)

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/nuscenes/',
                split='val',
                num_classes=10,
                load_type='frame_based',
                shuffle=False,
                num_frames=min(settings.num_frames, 81),
                name=DATASET_CATEGORY_NUSCENES_FRAME)

            if dataset_cache[DATASET_CATEGORY_NUSCENES_FRAME]['dataset_init'] is False:
                try:
                    # load nuscnes dataset first
                    nusc, nusc_can_bus = load_nuscenes(dataset_calib_cfg['path'])
                    dataset_cache[DATASET_CATEGORY_NUSCENES_FRAME]['calibration_dataset'] = \
                        NuScenesDataset(**dataset_calib_cfg, nusc=nusc, nusc_can_bus=nusc_can_bus, download=False, read_anno=False)
                    dataset_cache[DATASET_CATEGORY_NUSCENES_FRAME]['input_dataset'] = \
                        NuScenesDataset(**dataset_val_cfg, nusc=nusc, nusc_can_bus=nusc_can_bus, download=False, read_anno=True)
                    dataset_cache[DATASET_CATEGORY_NUSCENES_FRAME]['dataset_init'] =  True
                except Exception as message:
                    dataset_cache[DATASET_CATEGORY_NUSCENES_FRAME]['calibration_dataset'] = DATASET_CATEGORY_NUSCENES_FRAME
                    dataset_cache[DATASET_CATEGORY_NUSCENES_FRAME]['input_dataset'] = DATASET_CATEGORY_NUSCENES_FRAME
                    print(f'Nuscenes dataset loader could not be created: {message}')
            #
        #

        if check_dataset_load(settings, DATASET_CATEGORY_NUSCENES_MV_IMAGE) and (DATASET_CATEGORY_NUSCENES_MV_IMAGE in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_NUSCENES_MV_IMAGE} variant:{DATASET_CATEGORY_NUSCENES_MV_IMAGE}"))
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/nuscenes/',
                split='train',
                num_classes=10,
                load_type='mv_image_based',
                shuffle=False,
                num_frames=min(50, calibration_frames_nx),
                name=DATASET_CATEGORY_NUSCENES_MV_IMAGE)

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/nuscenes/',
                split='val',
                num_classes=10,
                load_type='mv_image_based',
                shuffle=False,
                num_frames=min(settings.num_frames, 456),
                name=DATASET_CATEGORY_NUSCENES_MV_IMAGE)

            # To revisit
            if dataset_cache[DATASET_CATEGORY_NUSCENES_MV_IMAGE]['dataset_init'] is False:
                try:
                    # load nuscnes dataset first
                    nusc, nusc_can_bus = load_nuscenes(dataset_calib_cfg['path'])
                    dataset_cache[DATASET_CATEGORY_NUSCENES_MV_IMAGE]['calibration_dataset'] = \
                        NuScenesDataset(**dataset_calib_cfg, nusc=nusc, nusc_can_bus=None, download=False, read_anno=False)
                    dataset_cache[DATASET_CATEGORY_NUSCENES_MV_IMAGE]['input_dataset'] = \
                        NuScenesDataset(**dataset_val_cfg, nusc=nusc, nusc_can_bus=None, download=False, read_anno=True)
                    dataset_cache[DATASET_CATEGORY_NUSCENES_MV_IMAGE]['dataset_init'] =  True
                except Exception as message:
                    dataset_cache[DATASET_CATEGORY_NUSCENES_MV_IMAGE]['calibration_dataset'] = DATASET_CATEGORY_NUSCENES_MV_IMAGE
                    dataset_cache[DATASET_CATEGORY_NUSCENES_MV_IMAGE]['input_dataset'] = DATASET_CATEGORY_NUSCENES_MV_IMAGE
                    print(f'Nuscenes dataset loader could not be created: {message}')
    #

    _get_additional_datasets(settings, download=download, dataset_list=dataset_list)

    return dataset_cache


def download_datasets(settings, download=True, dataset_list=None):
    # just creating the dataset classes with download=True will check if the dataset folders are present
    # if the dataset folders are missing, it will be downloaded and extracted
    # set download='always' to force re-download the datasets
    settings.dataset_cache = get_datasets(settings, download=download, dataset_list=dataset_list)
    return settings.dataset_cache


def _in_dataset_loading(settings, dataset_names):
    if settings.dataset_loading is None or settings.dataset_loading is True:
        return True
    elif settings.dataset_loading is False:
        return False
    #
    dataset_loading = utils.as_list(settings.dataset_loading)
    dataset_names = utils.as_list(dataset_names)
    for dataset_name in dataset_names:
        if dataset_name in dataset_loading:
            return True
        #
    #
    return False

def _in_dataset_selection(settings, dataset_names):
    if settings.dataset_selection is None or settings.dataset_selection is True:
        return True
    elif settings.dataset_selection is False:
        return False
    #
    dataset_selection = utils.as_list(settings.dataset_selection)
    dataset_names = utils.as_list(dataset_names)
    for dataset_name in dataset_names:
        if dataset_name in dataset_selection:
            return True
        #
    #
    return False

def check_dataset_load(settings, dataset_names):
    return _in_dataset_loading(settings, dataset_names) and _in_dataset_selection(settings, dataset_names)
