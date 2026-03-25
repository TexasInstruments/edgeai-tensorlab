# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.datasets.transforms import LoadAnnotations3D
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Sparse4DLoadAnnotations3D(LoadAnnotations3D):
    """Sparse4DLoadAnnotations3D.
        Inherits from LoadAnnotations3D to load instance_inds for Sparse4D.
    """

    def _load_instance_inds(self, results: dict) -> dict:
        results['instance_inds'] = results['ann_info']['instance_inds']
        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        results = super().transform(results)
        results = self._load_instance_inds(results)
        return results


