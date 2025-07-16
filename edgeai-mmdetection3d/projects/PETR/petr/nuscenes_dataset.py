# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion
import os
import random
import math
import torch

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

        #info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        '''
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['camera_sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        '''

        info.update(
            dict(
                #pts_filename=info['lidar_path'],
                #sweeps=info['camera_sweeps'],
                #timestamp=info['timestamp'],
            ))        

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
                '''
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T  
                lidar2cam_rt[3, :3] = -lidar2cam_t
                '''
                lidar2cam_rt = cam_info['lidar2cam'] 
                
                intrinsic = cam_info['cam2img']
                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                #lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                #intrinsics.append(viewpad)
                
                ###The extrinsics mean the tranformation from lidar to camera. 
                ### If anyone want to use the extrinsics as sensor to lidar, 
                ### please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and 
                ### LoadMultiViewImageFromMultiSweepsFiles.
                #lidar2cam_rts.append(lidar2cam_rt)  
                lidar2img_rts.append(lidar2img_rt)

            info.update(
                dict(
                    img_timestamp=img_timestamp,
                    #img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    #intrinsics=intrinsics,
                    #lidar2cam=lidar2cam_rts 
                ))


        #if not self.test_mode:
        #    annos = self.get_ann_info(index)
        #    input_dict['ann_info'] = annos
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


'''
def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix
'''

@DATASETS.register_module()
class StreamNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, *args, **kwargs):
        # Disable "serialize_data" for _set_sequence_group_flag
        # It will slow down data loading for multiple workers
        kwargs["serialize_data"] = False
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_list)):
            #if idx != 0 and len(self.data_list[idx]['lidar_sweeps']) == 0:
            if idx != 0 and self.data_list[idx].get('lidar_sweeps') is None:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_list)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            
            if not self.seq_mode: # for sliding window only
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            """ pre_pipeline:
            results['img_fields'] = []
            results['bbox3d_fields'] = []
            results['pts_mask_fields'] = []
            results['pts_seg_fields'] = []
            results['bbox_fields'] = []
            results['mask_fields'] = []
            results['seg_fields'] = []
            """
            input_dict['box_type_3d'] = self.box_type_3d
            input_dict['box_mode_3d'] = self.box_mode_3d

            #self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            queue.append(example)
            

        for k in range(self.num_frame_losses):
            #if self.filter_empty_gt and \
            #    (queue[-k-1] is None or ~(queue[-k-1]['data_samples'].gt_labels_3d != -1).any()):
            if self.filter_empty_gt and \
                (example is None or example['data_samples'].gt_labels_3d.shape[0] == 0):
                return None
        return self.union2one(queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        """ pre_pipeline:
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        """
        input_dict['box_type_3d'] = self.box_type_3d
        input_dict['box_mode_3d'] = self.box_mode_3d

        #self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def union2one(self, queue):
        imgs_list = [each['inputs']['img'].data for each in queue]
        for key in self.collect_keys:
            if key != 'img':
                queue[-1]['data_samples'].metainfo[key] = [each['data_samples'].metainfo[key] for each in queue]
            """
            if key != 'img_metas':
                #queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
                queue[-1][key] = torch.stack([each[key].data for each in queue])
            else:
                #queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                queue[-1][key] = [each[key].data for each in queue]
            """

        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_bboxes_labels', 'centers_2d', 'depths']:
                queue[-1]['data_samples'].metainfo[key] = [each['data_samples'].metainfo[key] for each in queue]
                """
                if key == 'gt_bboxes_3d':
                    #queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                    queue[-1][key] = [each[key].data for each in queue]
                else:
                    #queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)
                    queue[-1][key] = [each[key].data for each in queue]
                """
        queue[-1]['inputs']['img'] = torch.stack(imgs_list)
        queue = queue[-1]
        return queue


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
        #info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        e2g_matrix = np.array(info['ego2global'])
        l2e_matrix = np.array(info['lidar_points']['lidar2ego'])
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global

        #ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        ego_pose_inv = np.linalg.inv(ego_pose)
        if 'lidar_sweeps' in info:
            lidar_sweeps = info['lidar_sweeps']
        else:
            lidar_sweeps = []
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=info['lidar_points']['lidar_path'],
            sweeps=lidar_sweeps,
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp']
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['images'].items():
                img_timestamp.append(cam_info['timestamp'])
                image_paths.append(cam_info['img_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_rt = cam_info['lidar2cam']

                intrinsic = cam_info['cam2img']
                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)

            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    images=info['images'],
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                ))

        if not self.test_mode:
            annos = info['ann_info']
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers_2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'])
            )
            input_dict['ann_info'] = annos

        return input_dict


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

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
