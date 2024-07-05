import copy

import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
import os

from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.  
                - ann_info (dict): Annotation info.
        """
        info = super().get_data_info(index) 

        #info['sample_idx'] = info['token']
        info['lidar_path'] = info['lidar_points']['lidar_path']


        if self.modality['use_camera']:
            #image_paths = []
            lidar2img_rts = []
            #intrinsics = []
            #lidar2cam_rts = []
            img_timestamp = []
            for cam_type, cam_info in info['images'].items():
                img_timestamp.append(cam_info['timestamp'])
                #image_paths.append(cam_info['img_path'])
                
                # obtain lidar to image transformation matrix
                lidar2cam_rt = cam_info['lidar2cam'] 

                intrinsic = cam_info['cam2img']
                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt) 
                
                ###The extrinsics mean the tranformation from lidar to camera. 
                ### If anyone want to use the extrinsics as sensor to lidar, 
                ### please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and 
                ### LoadMultiViewImageFromMultiSweepsFiles.
                lidar2img_rts.append(lidar2img_rt)

            info.update(
                dict(
                    img_timestamp=img_timestamp,
                    lidar2img=lidar2img_rts,
                ))


        return info

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
        if self._fully_initialized:
            return

        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()

        # sort according to timestamp
        self.data_list = list(sorted(self.data_list, key=lambda e: e['timestamp']))

        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True


    # BEVDet's loss computation is not correct. But when the right GT is used, somehow the model 
    # is not trained properly. So we override parse_ann_info to make it identical to BEVDet.
    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super(NuScenesDataset, self).parse_ann_info(info)

        if ann_info is not None:
            gt_bboxes_3d, gt_labels_3d = info['ann_infos']
            gt_bboxes_3d, gt_labels_3d = np.array(gt_bboxes_3d), np.array(gt_labels_3d)

            if len(gt_bboxes_3d) == 0:
                gt_bboxes_3d = np.zeros((0, 9), dtype=np.float32)

            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
            ann_info['gt_labels_3d'] = gt_labels_3d
        else:
            # empty instance
            ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['attr_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.load_type in ['fov_image_based', 'mv_image_based']:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info

