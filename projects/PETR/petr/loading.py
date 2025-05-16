# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import mmcv
import numpy as np

from einops import rearrange
import torch

from mmengine.fileio import get
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles, LoadAnnotations3D
from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadMapsFromFiles(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(256,3*256)
        results['map_shape']=maps.shape
        results['maps']=maps
        return results



@TRANSFORMS.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                sweeps_num=5,
                to_float32=False,
                optimized_inference=False,
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                sweep_range=[3,27],
                sweeps_id = None,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                prob=1.0,
                ):

        self.sweeps_num = sweeps_num
        self.to_float32 = to_float32
        self.optimized_inference = optimized_inference
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def transform(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)

        # For inferencing, we do not read sweeps information
        # Intead, reuse previous frames
        if self.optimized_inference and self.test_mode:
            results['delta_timestamp'] = timestamp_imgs_list
            results['lidar2ego'] = results['lidar_points']['lidar2ego']
            return results

        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['camera_sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['cam2img'].append(np.copy(results['cam2img'][j]))
                    results['lidar2cam'].append(np.copy(results['lidar2cam'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['camera_sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['camera_sweeps']))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['camera_sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['camera_sweeps']))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
                
            for idx in choices:
                sweep_idx = min(idx, len(results['camera_sweeps']) - 1)
                sweep = results['camera_sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['camera_sweeps'][sweep_idx - 1]
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['cam2img'].append(sweep[sensor]['cam2img'])
                    results['lidar2cam'].append(sweep[sensor]['lidar2cam'])
        results['img'] = sweep_imgs_list
        results['delta_timestamp'] = timestamp_imgs_list  

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@TRANSFORMS.register_module()
class LoadMapsFromFiles_flattenf200f3(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        # maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(3,200*200)
        maps[maps>=0.5]=1
        maps[maps<0.5]=0
        maps=1-maps
        results['map_shape']=maps.shape
        results['maps']=maps
        
        return results


@TRANSFORMS.register_module()
class StreamPETRLoadAnnotations3D(LoadAnnotations3D):

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        The only difference is it remove the proceess for
        `ignore_flag`

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        results['gt_bboxes'] = results['ann_info']['bboxes']
        gt_bboxes_ignore = results['ann_info'].get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_bboxes_labels'] = results['ann_info']['labels']

    def _load_bboxes_depth(self, results: dict) -> dict:
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """

        results['depths'] = results['ann_info']['depths']
        results['centers_2d'] = results['ann_info']['centers_2d']
        return results


