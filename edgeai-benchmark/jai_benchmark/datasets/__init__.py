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
from .modelmaker_datasets import *

from .kitti_lidar_det import *
from .coco_kpts import *
from .widerface_det import *

dataset_info_dict = {
    #------------------------image classification datasets--------------------------#
    # Original ImageNet
    'imagenet':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetCls, 'size':50000, 'split':'val'},
    'imagenetv1':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetCls, 'size':50000, 'split':'val'},
    # ImageNetV2 as explained in imagenet_v2.py
    'imagenetv2c':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetV2C, 'size':10000, 'split':'val'},
    'imagenetv2b':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetV2B, 'size':10000, 'split':'val'},
    'imagenetv2a':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetV2A, 'size':10000, 'split':'val'},
    # smaller versions of the original ImageNet
    'tiny-imagenet200':{'task_type':'classification', 'category':'imagenet', 'type':TinyImageNet200Cls, 'size':10000, 'split':'val'},
    'imagenet-dogs120':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetDogs120Cls, 'size':20580, 'split':'train'},
    'imagenet-pseudo120':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetPseudo120Cls, 'size':20580, 'split':'train'},
    'imagenet-resized-64x64':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetResized64x64Cls, 'size':50000, 'split':'val'},
    #------------------------object detection datasets--------------------------#
    'coco': {'task_type':'detection', 'category':'coco', 'type':COCODetection, 'size':5000, 'split':'val2017'},
    'widerface': {'task_type':'detection', 'category':'widerface', 'type':WiderFaceDetection, 'size':3226, 'split':'val'},
    #------------------------semantic segmentation datasets--------------------------#
    'ade20k32': {'task_type':'segmentation', 'category':'ade20k32', 'type':ADE20KSegmentation, 'size':2000, 'split':'validation'},
    'ade20k': {'task_type':'segmentation', 'category':'ade20k', 'type':ADE20KSegmentation, 'size':2000, 'split':'validation'},
    'voc2012': {'task_type':'segmentation', 'category':'voc2012', 'type':VOC2012Segmentation, 'size':1449, 'split':'val'},
    'cocoseg21': {'task_type':'segmentation', 'category':'cocoseg21', 'type':COCOSegmentation, 'size':5000, 'split':'val2017'},
    #------------------------pose estimation datasets--------------------------#
    'cocokpts': {'task_type':'keypoint_detection', 'category':'cocokpts', 'type':COCOKeypoints, 'size':5000, 'split':'val2017'},
    #------------------------depth estimation datasets--------------------------#
    'nyudepthv2': {'task_type':'depth_estimation', 'category':'nyudepthv2', 'type':NYUDepthV2, 'size':654, 'split':'val'},
 }


dataset_info_dict_experimental = {
    #------------------------semantic segmentation datasets--------------------------#
    'cityscapes': {'task_type':'segmentation', 'category':'cityscapes', 'type':CityscapesSegmentation, 'size':500, 'split':'val'},
    'ti-robokit_semseg_zed1hd': {'task_type':'segmentation', 'category':'ti-robokit_semseg_zed1hd', 'type':ImageSegmentation, 'size':49, 'split':'val'},
    #------------------------3D OD datasets--------------------------#
    'kitti_lidar_det': {'task_type':'3d-detection', 'category':'kitti_lidar_det', 'type':KittiLidar3D, 'size':3769, 'split':'val'},
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


def get_datasets(settings, download=False):
    dataset_categories = get_dataset_categories(settings)
    dataset_cache = {ds_category:{'calibration_dataset':None, 'input_dataset':None} for ds_category in dataset_categories}

    dset_info_dict = get_dataset_info_dict(settings)

    if in_dataset_loading(settings, 'imagenet'):
        dataset_variant = settings.dataset_type_dict['imagenet'] if \
            settings.dataset_type_dict is not None else 'imagenet'
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
            num_frames=settings.calibration_frames,
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
        dataset_cache['imagenet']['calibration_dataset'] = ImageNetDataSetType(**imagenet_cls_calib_cfg, download=download)
        dataset_cache['imagenet']['input_dataset'] = ImageNetDataSetType(**imagenet_cls_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'cocokpts'):
        filter_imgs = True
        coco_kpts_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name='cocokpts',
            filter_imgs=filter_imgs)
        coco_kpts_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(settings.num_frames,5000),
            name='cocokpts',
            filter_imgs=filter_imgs)

        dataset_cache['cocokpts']['calibration_dataset'] = COCOKeypoints(**coco_kpts_calib_cfg, download=download)
        dataset_cache['cocokpts']['input_dataset'] = COCOKeypoints(**coco_kpts_val_cfg, download=False)

    if in_dataset_loading(settings, 'coco'):
        coco_det_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name='coco')
        coco_det_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=False, # can be set to True as well, if needed
            num_frames=min(settings.num_frames,5000),
            name='coco')
        dataset_cache['coco']['calibration_dataset'] = COCODetection(**coco_det_calib_cfg, download=download)
        dataset_cache['coco']['input_dataset'] = COCODetection(**coco_det_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'widerface'):
        widerface_det_calib_cfg = dict(
            path=f'{settings.datasets_path}/widerface',
            split='val',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name='widerface')
        widerface_det_val_cfg = dict(
            path=f'{settings.datasets_path}/widerface',
            split='val',
            shuffle=False, # can be set to True as well, if needed
            num_frames=min(settings.num_frames,3226),
            name='widerface')
        dataset_cache['widerface']['calibration_dataset'] = WiderFaceDetection(**widerface_det_calib_cfg, download=download)
        dataset_cache['widerface']['input_dataset'] = WiderFaceDetection(**widerface_det_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'cocoseg21'):
        cocoseg21_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name='cocoseg21')
        cocoseg21_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=min(settings.num_frames,5000),
            name='cocoseg21')
        dataset_cache['cocoseg21']['calibration_dataset'] = COCOSegmentation(**cocoseg21_calib_cfg, download=download)
        dataset_cache['cocoseg21']['input_dataset'] = COCOSegmentation(**cocoseg21_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'ade20k'):
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name='ade20k')
        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames, 2000),
            name='ade20k')
        dataset_cache['ade20k']['calibration_dataset'] = ADE20KSegmentation(**ade20k_seg_calib_cfg, download=download)
        dataset_cache['ade20k']['input_dataset'] = ADE20KSegmentation(**ade20k_seg_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'ade20k32'):
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name='ade20k32')
        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames, 2000),
            name='ade20k32')
        dataset_cache['ade20k32']['calibration_dataset'] = ADE20KSegmentation(**ade20k_seg_calib_cfg, num_classes=32, download=download)
        dataset_cache['ade20k32']['input_dataset'] = ADE20KSegmentation(**ade20k_seg_val_cfg, num_classes=32, download=False)
    #
    if in_dataset_loading(settings, 'voc2012'):
        voc_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name='voc2012')
        voc_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=min(settings.num_frames, 1449),
            name='voc2012')
        dataset_cache['voc2012']['calibration_dataset'] = VOC2012Segmentation(**voc_seg_calib_cfg, download=download)
        dataset_cache['voc2012']['input_dataset'] = VOC2012Segmentation(**voc_seg_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'nyudepthv2'):
        filter_imgs = False
        nyudepthv2_calib_cfg = dict(
            path=f'{settings.datasets_path}/nyudepthv2',
            split='val',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name='nyudepthv2')
        nyudepthv2_val_cfg = dict(
            path=f'{settings.datasets_path}/nyudepthv2',
            split='val',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(settings.num_frames, 654),
            name='nyudepthv2')

        dataset_cache['nyudepthv2']['calibration_dataset'] = NYUDepthV2(**nyudepthv2_calib_cfg, download=download)
        dataset_cache['nyudepthv2']['input_dataset'] = NYUDepthV2(**nyudepthv2_val_cfg, download=False)
    #
    # the following are datasets cannot be downloaded automatically
    # put it under the condition of experimental_models
    if settings.experimental_models:
        if in_dataset_loading(settings, 'cityscapes'):
            cityscapes_seg_calib_cfg = dict(
                path=f'{settings.datasets_path}/cityscapes',
                split='val',
                shuffle=True,
                num_frames=settings.calibration_frames,
                name='cityscapes')
            cityscapes_seg_val_cfg = dict(
                path=f'{settings.datasets_path}/cityscapes',
                split='val',
                shuffle=True,
                num_frames=min(settings.num_frames,500),
                name='cityscapes')
            dataset_cache['cityscapes']['calibration_dataset'] = CityscapesSegmentation(**cityscapes_seg_calib_cfg, download=False)
            dataset_cache['cityscapes']['input_dataset'] = CityscapesSegmentation(**cityscapes_seg_val_cfg, download=False)
        #
        if in_dataset_loading(settings, 'kitti_lidar_det'):
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=1,
                shuffle=False,
                num_frames=min(settings.calibration_frames, 3769))

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=1,
                shuffle=False,
                num_frames=min(settings.num_frames, 3769))

            dataset_cache['kitti_lidar_det']['calibration_dataset'] = KittiLidar3D(**dataset_calib_cfg, download=False, read_anno=False)
            dataset_cache['kitti_lidar_det']['input_dataset'] = KittiLidar3D(**dataset_val_cfg, download=False, read_anno=True)
        #
        if in_dataset_loading(settings, 'ti-robokit_semseg_zed1hd'):
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
                split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/train_img_gt_pair.txt',
                num_classes=19,
                shuffle=True,
                num_frames=min(settings.calibration_frames,150),
                name='ti-robokit_semseg_zed1hd'
            )

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
                split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/val_img_gt_pair.txt',
                num_classes=19,
                shuffle=True,
                num_frames=min(settings.num_frames,49),
                name='ti-robokit_semseg_zed1hd'
            )

            dataset_cache['ti-robokit_semseg_zed1hd']['calibration_dataset'] = ImageSegmentation(**dataset_calib_cfg, download=False)
            dataset_cache['ti-robokit_semseg_zed1hd']['input_dataset'] = ImageSegmentation(**dataset_val_cfg, download=False)
        #
    #
    return dataset_cache


def download_datasets(settings, download=True):
    # just creating the dataset classes with download=True will check of the dataset folders are present
    # if the dataset folders are missing, it will be downloaded and extracted
    # set download='always' to force re-download the datasets
    settings.dataset_cache = get_datasets(settings, download=download)
    return True


def in_dataset_loading(settings, dataset_names):
    if settings.dataset_loading is False:
        return False
    #
    load_all_datasets = (settings.dataset_loading is True or settings.dataset_loading is None)
    dataset_loading = get_dataset_categories(settings) if load_all_datasets else utils.as_list(settings.dataset_loading)
    dataset_names = utils.as_list(dataset_names)
    for dataset_name in dataset_names:
        if dataset_name in dataset_loading:
            return True
        #
    #
    return False

