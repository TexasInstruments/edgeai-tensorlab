import numpy as np

from mmcv.transforms import BaseTransform
from mmcv.transforms import Compose, LoadImageFromFile
from mmdet3d.registry import TRANSFORMS


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


