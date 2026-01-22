import math
import copy

import torch
import numpy as np
import pyquaternion
from shapely.geometry import LineString
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs

from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
#from .utils import (
#    draw_lidar_bbox3d_on_img,
#    draw_lidar_bbox3d_on_bev,
#)


@DATASETS.register_module()
class SparseDriveNuScenesDataset(NuScenesDataset):

    def __init__(
        self,
        load_interval=1,
        vis_score_threshold=0.25,
        #pipeline=None,
        #data_root=None,
        #classes=None,
        map_classes=None,
        #with_velocity=True,
        #modality=None,
        #test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        #version="v1.0-trainval",
        #use_valid_flag=False,
        ###data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        batch_size=1,
        ###work_dir=None,
        ###eval_config=None,
        *args,
        **kwargs
    ):
        # Disable "serialize_data" for _set_sequence_group_flag
        # It will slow down data loading for multiple workers
        kwargs["serialize_data"] = False
        super().__init__(*args, **kwargs)

        self.load_interval = load_interval
        """
        self.use_valid_flag = use_valid_flag
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0

        if classes is not None:
            self.CLASSES = classes
        """
        if map_classes is not None: 
            self.MAP_CLASSES = map_classes
        """
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.det3d_eval_configs.class_names = list(self.det3d_eval_configs.class_range.keys())
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        self.track3d_eval_configs.class_names = list(self.track3d_eval_configs.class_range.keys())
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        """
        self.vis_score_threshold = vis_score_threshold

        ###self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        ###self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.batch_size = batch_size
        if with_seq_flag:
            self._set_sequence_group_flag()
        
        #self.work_dir = work_dir
        #self.eval_config = eval_config

    def __len__(self):
        return len(self.data_list)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_list))
            return
        
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
                        list(
                            range(0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag] / self.sequences_split_num)))
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

    def anno2geom(self, annos):
        map_geoms = {}
        # Sometimes, particular map elements may not exist
        #for label in range(len(self.MAP_CLASSES)):
        #    map_geoms[label] = []

        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms
    
    def get_data_info(self, index):
        #info = self.data_infos[index]
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
            token=info["token"],
            map_location=info["map_location"],
            pts_filename=info['lidar_points']['lidar_path'],
            sweeps=lidar_sweeps,
            timestamp=info['timestamp'],
            lidar2ego=np.array(info['lidar_points']['lidar2ego']),
            ego2global=np.array(info['ego2global']),
            lidar2global=lidar2global,
            ego_status=info['ego_status'].astype(np.float32),
            #map_infos=info["map_annos"],
        )
        # Some data frame does not have map_infos in pikle file
        # So add empty ones here
        # Revisit pickle generation code later
        if 'map_annos' in info:
            input_dict['map_infos'] = info['map_annos']
        else:
            input_dict['map_infos'] = {1: [], 0: [], 2: []}

        """
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego
        """

        #map_geoms = self.anno2geom(info["map_annos"])
        map_geoms = self.anno2geom(input_dict['map_infos'])
        input_dict["map_geoms"] = map_geoms

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info["images"].items():
                img_timestamp.append(cam_info['timestamp'])
                image_paths.append(cam_info["img_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_rt = cam_info['lidar2cam']
                intrinsic = cam_info['cam2img']
                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    images=info['images'],
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    extrinsics=extrinsics
                )
            )
        #if 'ann_info' in info:
        #    annos = info['ann_info']
        #else:
        #    annos = self.parse_ann_info(info)
        #input_dict['ann_info'] = annos

        # double check if we need it for training as well
        if self.test_mode:
            ann_info = info['eval_ann_info']
        else:
            ann_info = info['ann_info']

        # Add 'fut_boxes in anno
        # It could be moved to evaluation
        if 'gt_ego_fut_trajs' in info:
            ann_info['gt_ego_fut_trajs'] = torch.from_numpy(info['gt_ego_fut_trajs'])
            ann_info['gt_ego_fut_masks'] = torch.from_numpy(info['gt_ego_fut_masks'])
            ann_info['gt_ego_fut_cmd'] = torch.from_numpy(info['gt_ego_fut_cmd'])
        
            ## get future box for planning eval
            fut_ts = int(info['gt_ego_fut_masks'].sum())
            fut_boxes = []
            cur_scene_token = info["scene_token"]
            cur_T_global = get_T_global(info)  # current lidar2global
            for i in range(1, fut_ts + 1):
                fut_info = self.data_list[index + i]
                fut_scene_token = fut_info["scene_token"]
                if cur_scene_token != fut_scene_token:
                    break

                """
                fut_gt_bboxes_3d = []
                if self.use_valid_flag:
                    for i, bbox in enumerate(fut_info['instances']):
                        if bbox["valid_flag"]:
                            fut_gt_bboxes_3d.append(bbox['bbox_3d'])
                else:
                    for i, bbox in enumerate(fut_info['instances']):
                        if bbox["num_lidar_pts"] > 0:
                            fut_gt_bboxes_3d.append(bbox['bbox_3d'])
                fut_gt_bboxes_3d = np.array(fut_gt_bboxes_3d)
                """
                if self.use_valid_flag:
                    mask = np.array([item['bbox_3d_isvalid'] for item in fut_info['instances']])
                else:
                    mask = np.array([item['num_lidar_pts']>0 for item in fut_info['instances']])
                fut_gt_bboxes_3d = np.array([item['bbox_3d'] for item in fut_info['instances']])
                if len(mask) != 0:
                    fut_gt_bboxes_3d = fut_gt_bboxes_3d[mask]
                else:
                    fut_gt_bboxes_3d = np.empty((0, 7))

                fut_T_global = get_T_global(fut_info)
                T_fut2cur = np.linalg.inv(cur_T_global) @ fut_T_global

                center = fut_gt_bboxes_3d[:, :3] @ T_fut2cur[:3, :3].T + T_fut2cur[:3, 3]
                yaw = np.stack([np.cos(fut_gt_bboxes_3d[:, 6]), np.sin(fut_gt_bboxes_3d[:, 6])], axis=-1)
                yaw = yaw @ T_fut2cur[:2, :2].T
                yaw = np.arctan2(yaw[..., 1], yaw[..., 0])

                fut_gt_bboxes_3d[:, :3] = center
                fut_gt_bboxes_3d[:, 6] = yaw
                fut_boxes.append(torch.from_numpy(fut_gt_bboxes_3d).unsqueeze(0))

            ann_info['fut_boxes'] = fut_boxes

        input_dict['ann_info'] = ann_info

        # For eval_pipeline
        input_dict['gt_bboxes_3d'] = ann_info['gt_bboxes_3d']
        input_dict['gt_labels_3d'] = ann_info['gt_labels_3d']
        if 'instance_inds' in ann_info:
            input_dict['instance_inds'] = ann_info['instance_inds']
        input_dict['gt_agent_fut_trajs'] = ann_info['gt_agent_fut_trajs']
        input_dict['gt_agent_fut_masks'] = ann_info['gt_agent_fut_masks']
        input_dict['gt_ego_fut_trajs'] = ann_info['gt_ego_fut_trajs']
        input_dict['gt_ego_fut_masks'] = ann_info['gt_ego_fut_masks']
        input_dict['gt_ego_fut_cmd'] = ann_info['gt_ego_fut_cmd']
        input_dict['fut_boxes'] = ann_info['fut_boxes']
        return input_dict


    def parse_ann_info(self, info: dict) -> dict:

        # Add "instance_inds" into info["instance"][:]
        if "instance_inds" in info:
            info["instance_inds"] = np.array(info["instance_inds"], dtype=np.int32)
            for i, instance in enumerate(info['instances']):
                instance['instance_inds'] = info["instance_inds"][i]

        ann_info = super().parse_ann_info(info)

        # filter out invalid GT
        if 'gt_agent_fut_trajs' in info:
            if self.use_valid_flag:
                mask = np.array([item['bbox_3d_isvalid'] for item in info['instances']])
            else:
                mask = np.array([item['num_lidar_pts']>0 for item in info['instances']])

            if len(mask) != 0:
                info['gt_agent_fut_trajs'] = info['gt_agent_fut_trajs'][mask]
                info['gt_agent_fut_masks'] = info['gt_agent_fut_masks'][mask]

            ann_info['gt_agent_fut_trajs'] = torch.from_numpy(info['gt_agent_fut_trajs'])
            ann_info['gt_agent_fut_masks'] = torch.from_numpy(info['gt_agent_fut_masks'])

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


def get_T_global(info):
    lidar2ego = np.array(info['lidar_points']['lidar2ego'])
    ego2global = np.array(info["ego2global"])
    return ego2global @ lidar2ego
