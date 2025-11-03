from typing import Callable, List, Union
import math
import numpy as np

#from nuscenes.eval.detection.config import config_factory as det_configs
#from nuscenes.eval.common.config import config_factory as track_configs

from mmengine import logging
from mmengine.logging import print_log

from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset


@DATASETS.register_module()
class Sparse4DNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.
    This dataset only add camera intrinsics and extrinsics to the results.
    """
    def __init__(self,
                 load_interval=1,
                 vis_score_threshold=0.25,
                 data_aug_conf=None,
                 sequences_split_num=1,
                 with_seq_flag=False,
                 batch_size=1,
                 *args,
                 **kwargs
            ):
        # Disable "serialize_data" for _set_sequence_group_flag
        # It will slow down data loading for multiple workers
        kwargs["serialize_data"] = False
        super().__init__(*args, **kwargs)

        self.load_interval = load_interval

        self.vis_score_threshold = vis_score_threshold
        self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        self.current_aug = None
        self.last_id = None
        self.batch_size = batch_size
        if with_seq_flag:
            self._set_sequence_group_flag()


    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_list)):
            if idx != 0 and self.data_list[idx].get('lidar_sweeps') is None:                
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
                        list(range(0,
                                bin_counts[curr_flag],
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
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

        # standard protocal modified from SECOND.Pytorch
        e2g_matrix = np.array(info['ego2global'])
        l2e_matrix = np.array(info['lidar_points']['lidar2ego'])
        lidar2global =  e2g_matrix @ l2e_matrix
        #lidar2global_inv = np.linalg.inv(lidar2global)
        if 'lidar_sweeps' in info:
            lidar_sweeps = info['lidar_sweeps']
        else:
            lidar_sweeps = []
        input_dict = dict(
            sample_idx=info['sample_idx'],
            token=info['token'],
            pts_filename=info['lidar_points']['lidar_path'],
            sweeps=lidar_sweeps,
            timestamp=info['timestamp'],
            lidar2ego=np.array(info['lidar_points']['lidar2ego']),
            ego2global=np.array(info['ego2global']),
            lidar2global=lidar2global,
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
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)

            #if not self.test_mode: # for seq_mode
            #    prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            #else:
            #    prev_exists = None

            input_dict.update(
                dict(
                    images=info['images'],
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    extrinsics=extrinsics,
                ))

        if not self.test_mode:
            if 'ann_info' in info:
                annos = info['ann_info']
            else:
                annos = self.parse_ann_info(info)
            input_dict['ann_info'] = annos

        return input_dict

    def parse_ann_info(self, info: dict) -> dict:
        #TODO modify for current version
        if "instance_inds" in info:
            info["instance_inds"] = np.array(info["instance_inds"], dtype=np.int32)
            for i, instance in enumerate(info['instances']):
                instance['instance_inds'] = info["instance_inds"][i]

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
#     dataset = Sparse4DNuScenesDataset('data/nuscenes', 'nuscenes_infos_val.pkl')
#     data = dataset[0]