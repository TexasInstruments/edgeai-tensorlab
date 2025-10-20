import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from pyquaternion import Quaternion

from typing import Callable, List, Union
import os
import random
import math
import torch

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.
    This dataset only add camera intrinsics and extrinsics to the results.
    """
    def __init__(self,
                data_root: str,
                ann_file: str,
                pipeline: List[Union[dict, Callable]] = [],
                box_type_3d: str = 'LiDAR',
                load_type: str = 'frame_based',
                modality: dict = dict(
                    use_camera=False,
                    use_lidar=True,
                ),
                filter_empty_gt: bool = True,
                test_mode: bool = False,
                with_velocity: bool = True,
                use_valid_flag: bool = False,
                batch_size: int = 1,
                seq_frame=0,
                max_tracking_frame_interval=1,
                max_interval=1,
                min_interval=1,
                max_time_interval=5,
                fix_interval=True,
                future_frame=0,
                tracking=False,
                sequences_split_num=1,
                with_seq_flag=False,
                **kwargs
                # data_aug_conf=None,
                # version="v1.0-trainval",
                # load_interval=1,
                # det3d_eval_version="detection_cvpr_2019",
                # track3d_eval_version="tracking_nips_2019",
                # vis_score_threshold=0.35,
                # rot_range=[-0.3925, 0.3925],
                # scale_ratio_range=[1.0, 1.0],
                # translation_std=[0, 0, 0],
                # keep_consistent_seq_aug=True,
            ):
        super().__init__(data_root, ann_file, pipeline=pipeline, 
                         box_type_3d=box_type_3d, load_type=load_type, modality=modality, 
                         filter_empty_gt=filter_empty_gt,test_mode=test_mode, 
                         with_velocity=with_velocity, use_valid_flag=use_valid_flag, **kwargs)

        # self.version = version
        # self.load_interval = load_interval
        # self.det3d_eval_version = det3d_eval_version
        # self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        # self.track3d_eval_version = track3d_eval_version
        # self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        # self.vis_score_threshold = vis_score_threshold
        self.batch_size = batch_size
        self.seq_frame = seq_frame
        self.max_tracking_frame_interval = max_tracking_frame_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_time_interval = max_time_interval
        self.fix_interval = fix_interval
        self.future_frame = future_frame

        # self.rot_range = rot_range
        # self.scale_ratio_range = scale_ratio_range
        # self.translation_std = translation_std
        # self.data_aug_conf = data_aug_conf
        self.tracking = tracking
        self.sequences_split_num = sequences_split_num
        # self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None
        if with_seq_flag:
            self._set_sequence_group_flag()

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_list)):
            if idx != 0 and len(self.data_list[idx]["sweeps"]) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_list)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)
    
    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        def _get_item(idx):
            if self.test_mode:
                return self.prepare_data(idx)
            else:
                while True:
                    data = self.prepare_data(idx)
                    if data is None:
                        idx = self._rand_another()
                        continue
                    return data
                
        data = _get_item(idx)
        if isinstance(data["data_samples"], list):
            cur_timestamp = data["data_samples"][0].metainfo["timestamp"]
        else:
            cur_timestamp = data["data_samples"].metainfo["timestamp"]

        interval = (
            int(random.random() * self.max_interval) + self.min_interval
        ) 
        if self.seq_frame > 0 and not self.test_mode:
            seq_frame_indice = []
            seq_frame_num = self.seq_frame
            if self.tracking:
                seq_frame_num += (int( random.random() * (self.max_tracking_frame_interval + 1)) + 1)

            idx_next = idx
            for i in range(seq_frame_num):
                if idx_next == 0:
                    break
                idx_next -= interval
                if not self.fix_interval:
                    interval = (int(random.random() * self.max_interval) + self.min_interval)
                if (i >= self.seq_frame) and (seq_frame_num - i > self.seq_frame + 1):
                    continue
                idx_next = max(idx_next, 0)
                seq_frame_indice.append(idx_next)

            data_queue = []
            for seq_idx in seq_frame_indice:
                seq_data = _get_item(seq_idx)
                if isinstance(seq_data["data_samples"], list):
                    seq_timestamp = seq_data["data_samples"][0].metainfo["timestamp"]
                else:
                    seq_timestamp = seq_data["data_samples"].metainfo["timestamp"]
                if abs(seq_timestamp - cur_timestamp) > self.max_time_interval:
                    break
                data_queue.append(seq_data)
            data["data_queue"] = data_queue

            if len(data_queue) > 0 and "instance_inds" in data:
                last_frame_id = max(len(data_queue) - self.seq_frame - 1, 0)
                last_instance_inds = data["data_queue"][last_frame_id]["instance_inds"]
                match_flag = (last_instance_inds[None] == data["instance_inds"][:, None])
                dummy_flag = np.logical_not(match_flag.any(axis=-1, keepdims=True))
                match_flag = np.concatenate([dummy_flag, match_flag], axis=-1)
                match_inds = np.where(match_flag)[1] - 1
                data["match_inds"] = match_inds

        if self.future_frame > 0:
            idx_next = idx
            future_data_queue = []
            if self.test_mode:
                interval = 1
            for i in range(self.future_frame):
                if idx_next == self.__len__() - 1:
                    break
                idx_next += interval
                if not self.fix_interval:
                    interval = (int(random.random() * self.max_interval) + self.min_interval)
                idx_next = min(idx_next, self.__len__() - 1)
                future_data = _get_item(idx_next)
                if isinstance(future_data["data_samples"], list):
                    seq_timestamp = future_data["data_samples"][0].metainfo["timestamp"]
                else:
                    seq_timestamp = future_data["data_samples"].metainfo["timestamp"]
                if abs(seq_timestamp - cur_timestamp) > self.max_time_interval:
                    break
                future_data_queue.append(future_data)
            data["future_data_queue"] = future_data_queue
        return data
        # return self.parse_data_info(info)
    
    def parse_data_info(self, info: dict) :
        input_dict = super().parse_data_info(info)
        # standard protocol modified from SECOND.Pytorch
        e2g_matrix = np.array(info['ego2global'])
        l2e_matrix = np.array(info['lidar_points']['lidar2ego'])
        lidar2global =  e2g_matrix @ l2e_matrix # lidar2global

        # #ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        # ego_pose_inv = np.linalg.inv(ego_pose)
        if 'lidar_sweeps' in info:
            lidar_sweeps = info['lidar_sweeps']
        else:
            lidar_sweeps = []
        input_dict.update(dict(
            token=info['token'],
            sample_idx=info['sample_idx'],
            pts_filename=os.path.join(self.data_prefix.get('pts', ''),info['lidar_points']['lidar_path']),
            sweeps=lidar_sweeps,
            timestamp=info['timestamp'],
            scene_token=info['scene_token'],
            lidar2ego=l2e_matrix,
            ego2global=e2g_matrix,
            lidar2global=lidar2global,
            prev_idx=info['prev'],
            next_idx=info['next'],
            frame_idx=info['frame_idx'],
        ))

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['images'].items():
                img_timestamp.append(cam_info['timestamp'])
                if cam_type in self.data_prefix:
                    image_paths.append(os.path.join(self.data_prefix.get(cam_type, ''),cam_info['img_path']))
                else:
                    image_paths.append(os.path.join(self.data_prefix.get('img', ''),cam_info['img_path']))
                # obtain lidar to image transformation matrix
                lidar2cam_rt = cam_info['lidar2cam']

                intrinsic = cam_info['cam2img']
                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)

            # if not self.test_mode: # for seq_mode
            #     prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            # else:
            #     prev_exists = None

            input_dict.update(
                dict(
                    images=info['images'],
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    # prev_exists=prev_exists,
                ))

        if not self.test_mode:
            if 'ann_info' in info:
                annos = info['ann_info']
            else:
                annos = self.parse_ann_info(info)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.get_data_info(index)
        if 'ann_info' not in info:
            ann_info = self.parse_ann_info(info)
        else:
            ann_info = info['ann_info']
        return ann_info
    
    def parse_ann_info(self, info: dict) -> dict:
        #TODO modify for current version
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int32)
            for instance in info['instances']:
                instance['instance_inds'] = instance_inds
        ann_info = super().parse_ann_info(info)
        return ann_info

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


# if __name__ == '__main__':
#     dataset = CustomNuScenesDataset('data/nuscenes', 'nuscenes_infos_val.pkl')
#     data = dataset[0]