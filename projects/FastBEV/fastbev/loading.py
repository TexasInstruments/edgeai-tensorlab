import numpy as np

from mmcv.transforms import BaseTransform
from mmcv.transforms import Compose, LoadImageFromFile
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile
from mmdet3d.structures.points import get_points_type
from typing import Union, List, Optional
import torch


@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):
    def __init__(self, transforms, n_images, n_times=1, sequential=False):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.n_times = n_times
        self.sequential = sequential

    def __sort_list(self, old_list, order):
        new_list = []
        for i in order:
            new_list.append(old_list[i])
        return new_list

    def transform(self, results):
        imgs = []
        extrinsics = []
        if not self.sequential:
            assert len(results['img_path']) == 6
            ids = np.arange(len(results['img_path']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            ids_list = sorted(ids)  # sort & tolist
        else:
            assert len(results['img_path']) == 6 * self.n_times, f'img path: {len(results["img_path"])}, n_times: {self.n_times}'
            ids_list = np.arange(len(results['img_path'])).tolist()
        for i in ids_list:
            _results = dict()
            for key in ['img_prefix', 'img_path']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_path']:
                results[key] = _results[key]
        results['img'] = imgs
        # resort 2d box by random ids
        if 'gt_bboxes' in results.keys():
            gt_bboxes = self.__sort_list(results['gt_bboxes'], ids_list)
            gt_labels = self.__sort_list(results['gt_labels'], ids_list)
            gt_bboxes_ignore = self.__sort_list(results['gt_bboxes_ignore'], ids_list)
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels
            results['gt_bboxes_ignore'] = gt_bboxes_ignore

        results['lidar2img']['extrinsic'] = extrinsics
        return results


@TRANSFORMS.register_module()
class ResetPointOrigin(BaseTransform):
    """ Reset point cloud origin to Kitti dataset origin
    """
    def __init__(self, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def transform(self, results):
        results['lidar2img']['origin'] = self.origin.copy()
        return results



@TRANSFORMS.register_module()
class CustomLoadPointsFromFile(LoadPointsFromFile):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 dummy: bool = False,
                 backend_args: Optional[dict] = None) -> None:

        super().__init__(coord_type,
                         load_dim = load_dim,
                         use_dim = use_dim,
                         shift_height = shift_height,
                         use_color = use_color,
                         norm_intensity = norm_intensity,
                         norm_elongation = norm_elongation,
                         backend_args = backend_args)

        self.dummy = dummy


    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """

        if self.dummy:
            points = torch.ones([1, self.load_dim], dtype=torch.float32)
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=None)
            results['points'] = points
            return results

        results = super().transform(results)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        repr_str += f'dummy={self.dummy})'
        return repr_str