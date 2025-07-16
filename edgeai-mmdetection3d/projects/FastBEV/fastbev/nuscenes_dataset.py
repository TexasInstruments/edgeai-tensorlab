# -*- coding: utf-8 -*-
import torch
import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmengine.logging import print_log
import logging
import mmcv
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from mmdet3d.structures import LiDARInstance3DBoxes
try:
    from tools.dataset_converters.nuscenes_converter import get_2d_boxes
except:
    print('import error')


def tofloat(x):
    return x.astype(np.float32) if x is not None else None

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    
    def __init__(self,
                 with_box2d=False,
                 sequential=False,
                 n_times=1,
                 speed_mode='relative_dis',
                 prev_only=False,
                 next_only=False,
                 train_adj_ids=None,
                 test_adj='prev',
                 test_adj_ids=None,
                 test_time_id=1,
                 max_interval=3,
                 min_interval=0,
                 fix_direction=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.sequential = sequential
        self.n_times = n_times
        self.prev_only = prev_only
        self.next_only = next_only
        self.train_adj_ids = train_adj_ids
        self.test_adj = test_adj
        self.test_adj_ids = test_adj_ids
        self.test_time_id = test_time_id
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.speed_mode = speed_mode
        self.fix_direction = fix_direction

        """
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)        
        self.scene2map = get_scene2map(self.nusc)
        self.maps = get_nusc_maps()
        """

        # box 2d
        self.with_box2d = with_box2d

        xbound = [-50, 50, 0.5]
        ybound = [-50, 50, 0.5]
        zbound = [-10, 10, 20.0]
        dbound = [4.0, 45.0, 1.0]

        self.nx = np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype='int64')
        self.dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        self.bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])


        """
        self.lane_thickness = 2
        for i in range(5):
            print('lane thickness: {}'.format(self.lane_thickness))

        #self.debug = False
        """

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


    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        while True:
            # TBA
            data = self.prepare_data(idx)

            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def get_adj_data_info(self, data_info, index):
        """ Get adjacent data info. 

        Args:
            data_info (int): Basic data information form .pkl file

        Returns:
            dict: Data information including temporally adjacent data that \
                will be passed to the data preprocessing pipelines.
        """
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=data_info['sample_idx'],
            sample_token=data_info['token'],
            scene_token=data_info['scene_token'],
            pts_filename=data_info['lidar_points']['lidar_path'],
            lidar2ego=data_info['lidar_points']['lidar2ego'],
            ego2global=data_info['ego2global'],
            #sweeps=data_info['sweeps'],
            timestamp=data_info['timestamp'],
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2img_augs = []
            lidar2img_extras = []
            kws = [
                'cam2ego',
                'lidar2cam',
                'cam2img'
            ]

            for cam_type, cam_info in data_info['images'].items():
                image_paths.append(cam_info['img_path'])

                # keep original rts
                lidar2img_extra = {kw: cam_info[kw] for kw in kws}
                lidar2img_extras.append(lidar2img_extra)

                # obtain lidar to image transformation matrix
                intrinsic = cam_info['cam2img']
                lidar2cam_rt = cam_info['lidar2cam']

                # keep aug rts
                lidar2img_aug = {
                    'intrin': np.array(cam_info['cam2img']),
                    'lidar2cam': np.array(cam_info['lidar2cam']),
                    'cam2lidar': np.linalg.inv(cam_info['lidar2cam']),
                    'post_rot': np.eye(3),
                    'post_tran': np.zeros(3),
                }
                lidar2img_augs.append(lidar2img_aug)

                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)


            # extrinsic transfromation between the current and ajacent camera images
            if self.sequential:
                adjacent_type_list = []
                adjacent_id_list = []
                for time_id in range(1, self.n_times):
                    if data_info['prev'] is None or data_info['next'] is None:
                        adjacent = 'prev' if data_info['next'] is None else 'next'
                    else:
                        if self.prev_only or self.next_only:
                            adjacent = 'prev' if self.prev_only else 'next'
                        # stage: test
                        elif self.test_mode:
                            if self.test_adj_ids is not None:
                                assert len(self.test_adj_ids) == self.n_times - 1
                                select_id = self.test_adj_ids[time_id-1]
                                assert self.min_interval <= select_id <= self.max_interval
                                adjacent = {True: 'prev', False: 'next'}[select_id > 0]
                            else:
                                adjacent = self.test_adj
                        # stage: train
                        elif self.train_adj_ids is not None:
                            assert len(self.train_adj_ids) == self.n_times - 1
                            select_id = self.train_adj_ids[time_id-1]
                            assert self.min_interval <= select_id <= self.max_interval
                            adjacent = {True: 'prev', False: 'next'}[select_id > 0]
                        else:
                            adjacent = np.random.choice(['prev', 'next'])

                    if type(data_info[adjacent]) is list:
                        # stage: test
                        if self.test_mode:
                            if len(data_info[adjacent]) <= self.min_interval:
                                select_id = len(data_info[adjacent]) - 1
                            elif self.test_adj_ids is not None:
                                assert len(self.test_adj_ids) == self.n_times - 1
                                select_id = self.test_adj_ids[time_id-1]
                                assert self.min_interval <= select_id <= self.max_interval
                                select_id = min(abs(select_id), len(data_info[adjacent])-1)
                            else:
                                assert self.min_interval >= 0 and self.max_interval >= 0, "single direction only here"
                                select_id_step = (self.max_interval+self.min_interval) // self.n_times
                                select_id = min(self.min_interval + select_id_step * time_id, len(data_info[adjacent])-1)
                        # stage: train
                        else:
                            if len(data_info[adjacent]) <= self.min_interval:
                                select_id = len(data_info[adjacent]) - 1
                            elif self.train_adj_ids is not None:
                                assert len(self.train_adj_ids) == self.n_times - 1
                                select_id = self.train_adj_ids[time_id-1]
                                assert self.min_interval <= select_id <= self.max_interval
                                select_id = min(abs(select_id), len(data_info[adjacent])-1)
                            else:
                                assert self.min_interval >= 0 and self.max_interval >= 0, "single direction only here"
                                select_id = np.random.choice([adj_id for adj_id in range(
                                    min(self.min_interval, len(data_info[adjacent])),
                                    min(self.max_interval, len(data_info[adjacent])))])
                        info_adj = data_info[adjacent][select_id]
                        #if self.verbose:
                        #    print(' get_data_info: ', 'time_id: ', time_id, adjacent, select_id)
                    else:
                        info_adj = data_info[adjacent]

                    adjacent_type_list.append(adjacent)
                    adjacent_id_list.append(select_id)

                    # current and ajacent
                    egocurr2global = np.array(data_info['ego2global'])
                    egoadj2global = info_adj['ego2global']
                    lidar2ego = np.array(data_info['lidar_points']['lidar2ego'])
                    lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                        @ egoadj2global @ lidar2ego

                    kws_adj = [
                        'ego2global',
                    ]
                    for cam_id, (cam_type, cam_info) in enumerate(info_adj['cams'].items()):
                        image_paths.append(cam_info['data_path'])

                        lidar2img_aug = lidar2img_augs[cam_id].copy()
                        # keep aug rts
                        lidar2img_augs.append(lidar2img_aug)

                        mat = lidaradj2lidarcurr @ lidar2img_aug['cam2lidar']
                        lidar2img_aug['cam2lidar'] = mat
                        lidar2img_aug['lidar2cam'] = np.linalg.inv(mat)

                        # obtain lidar to image transformation matrix
                        intrin = lidar2img_aug['intrin']
                        viewpad = np.eye(4)
                        viewpad[:intrin.shape[0], :intrin.shape[1]] = lidar2img_aug['intrin']
                        lidar2img_rt = (viewpad @ lidar2img_aug['lidar2cam'])
                        lidar2img_rts.append(lidar2img_rt)

                        # keep original rts
                        lidar2img_extra = {kw: info_adj[kw] for kw in kws_adj}
                        lidar2img_extras.append(lidar2img_extra)

                #if self.verbose:
                #    time_list = [0.0]
                #    for i in range(self.n_times-1):
                #        time = 1e-6 * (data_info['timestamp'] - data_info[adjacent_type_list[i]][adjacent_id_list[i]]['timestamp'])
                #        time_list.append(time)
                #    print(' get_data_info: ', 'time: ', time_list)

                data_info['adjacent_type'] = adjacent_type_list
                data_info['adjacent_id'] = adjacent_id_list

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    lidar2img_aug=lidar2img_augs,
                    lidar2img_extra=lidar2img_extras
                )
            )
            if self.sequential:
                input_dict.update(dict(info=data_info))

        if not self.test_mode:
            #annos = self.get_ann_info(index)
            #input_dict['ann_info'] = annos
            input_dict['ann_info'] = data_info['ann_info']
            if self.sequential:
                bbox = input_dict['ann_info']['gt_bboxes_3d'].tensor
                if 'abs' in self.speed_mode:
                    bbox[:, 7:7+2] = bbox[:, 7:7+2] + torch.from_numpy(data_info['velo']).view(1, 2)
                if 'dis' in self.speed_mode:
                    assert self.test_time_id is not None
                    adjacent_type = data_info['adjacent_type'][self.test_time_id-1]
                    if adjacent_type == 'next' and not self.fix_direction:
                        bbox[:, 7:7+2] = -bbox[:, 7:7+2]
                    time = abs(input_dict['timestamp'] - 1e-6 * data_info[adjacent_type][self.test_time_id-1]['timestamp'])
                    bbox[:, 7:9] = bbox[:, 7:9] * time
                input_dict['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                    bbox, box_dim=bbox.shape[-1], origin=(0.5, 0.5, 0.0))

        return input_dict

    def refine_data_info(self, data_info):
        """ Choose data info needed for Fast BEV

        Args:
            data_info (dict): Pre-processed data information form .pkl file

        Returns:
            dict: Refined data information including temporally adjacent data that \
                will be passed to the data preprocessing pipelines.
        """
        n_cameras = len(data_info['img_filename'])
        if not self.sequential:
            assert n_cameras == 6

        new_info = dict(
            sample_idx=data_info['sample_idx'],
            sample_token=data_info['sample_token'],
            scene_token=data_info['scene_token'],
            lidar2ego=data_info['lidar2ego'],
            ego2global=data_info['ego2global'],
            img_prefix=[None] * n_cameras,
            img_path=[x for x in data_info['img_filename']],
            lidar2img=dict(
                extrinsic=[tofloat(x) for x in data_info['lidar2img']],
                intrinsic=np.eye(4, dtype=np.float32),
                lidar2img_aug=data_info['lidar2img_aug'],
                lidar2img_extra=data_info['lidar2img_extra']
            )
        )

        if 'ann_info' in data_info:
            gt_bboxes_3d = data_info['ann_info']['gt_bboxes_3d']
            gt_labels_3d = data_info['ann_info']['gt_labels_3d'].copy()
            mask = gt_labels_3d >= 0
            gt_bboxes_3d = gt_bboxes_3d[mask]
            #gt_names = data_info['ann_info']['gt_names'][mask]
            gt_labels_3d = gt_labels_3d[mask]
            new_info['ann_info'] = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                #gt_names=gt_names,
                gt_labels_3d=gt_labels_3d
            )

        return new_info


    def get_data_info(self, index):
        raw_data_info = super().get_data_info(index)

        # Get adjacent data info
        data_info = self.get_adj_data_info(raw_data_info, index)

        # Choose selected data info needed fro Fast BEV
        data_info = self.refine_data_info(data_info)

        sample_token = data_info['sample_token']

        if 'ann_info' in data_info:
            """
            # get bev segm map
            bev_seg_gt = self._get_map_by_sample_token(sample_token)  # 200, 200, 2
            data_info['ann_info']['gt_bev_seg'] = bev_seg_gt
            """

            # get bbox2d for camera
            if self.with_box2d:
                # No need to change change the camera order?
                camera_types = [
                    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                info = raw_data_info
                gt_bboxes_mv, gt_labels_mv, gt_bboxes_ignore_mv = [], [], []
                for cam in camera_types:
                    gt_bboxes, gt_labels, gt_bboxes_ignore = [], [], []
                    coco_infos = get_2d_boxes(self.nusc,
                                              info['cams'][cam]['sample_data_token'],
                                              visibilities=['', '1', '2', '3', '4'],
                                              mono3d=False)
                    for coco_info in coco_infos:
                        if coco_info is None:
                            continue
                        elif coco_info.get('ignore', False):
                            continue
                        x1, y1, w, h = coco_info['bbox']
                        inter_w = max(0, min(x1 + w, 1600) - max(x1, 0))
                        inter_h = max(0, min(y1 + h, 900) - max(y1, 0))
                        if inter_w * inter_h == 0:
                            continue
                        if coco_info['area'] <= 0 or w < 1 or h < 1:
                            continue
                        if coco_info['category_id'] < 0 or coco_info['category_id'] > 9:
                            continue
                        bbox = [x1, y1, x1 + w, y1 + h]
                        if coco_info.get('iscrowd', False):
                            gt_bboxes_ignore.append(bbox)
                        else:
                            gt_bboxes.append(bbox)
                            gt_labels.append(coco_info['category_id'])

                    if gt_bboxes:
                        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                        gt_labels = np.array(gt_labels, dtype=np.int64)
                    else:
                        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                        gt_labels = np.array([], dtype=np.int64)

                    if gt_bboxes_ignore:
                        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
                    else:
                        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

                    gt_bboxes_mv.append(gt_bboxes)
                    gt_labels_mv.append(gt_labels)
                    gt_bboxes_ignore_mv.append(gt_bboxes_ignore)

                data_info['ann_info']['bboxes'] = gt_bboxes_mv 
                data_info['ann_info']['labels'] = gt_labels_mv 
                data_info['ann_info']['bboxes_ignore'] = gt_bboxes_ignore_mv 

        """
        if self.debug:
            print('### debug vis in <get_data_info> ###')
            scale_fac = 10
            bev_seg_road = mmcv.imresize(bev_seg_gt[...,0], (1000,1000)) * 127
            bev_gt_bboxes = data_info['ann_info']['gt_bboxes_3d'].corners.numpy()[:,[0,2,6,4]][..., :2]
          
            # draw nus (my gen GT)
            for idx in range(len(bev_gt_bboxes)):
                box = bev_gt_bboxes[idx]
                box = box[:,::-1] # xy reverse
                bev_gt_img = self.draw_bev_corner(bev_seg_road, box, 200, scale_fac)
            mmcv.imwrite(bev_gt_img, 'figs/bev_ours.png')
            # draw nus (real GT)
            self.nusc.render_sample_data(self.nusc.get('sample', sample_token)['data']['LIDAR_TOP'], axes_limit=50,out_path='trash/bev_nus.png')
            mmcv.imwrite(mmcv.imresize_like(mmcv.imread('figs/bev_nus.png'), bev_seg_road), 'figs/bev_nus.png')
        """

        return data_info

    """
    def coord_transform(self, pts, pose):
        pts = convert_points_to_local(pts, pose)
        pts = np.round(
                (pts - self.bx[:2]) / self.dx[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        return pts

    def _get_map_by_sample_token(self, sample_token):
        egopose = self.nusc.get(
            'ego_pose', 
            self.nusc.get(
                'sample_data', 
                 self.nusc.get('sample', sample_token)['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        pose = [egopose['translation'][0], egopose['translation'][1],
                        np.cos(rot), np.sin(rot)]

        bev_seg_gt_road = np.zeros((self.nx[0], self.nx[1]))
        bev_seg_gt_lane = np.zeros((self.nx[0], self.nx[1]))

        tgt_type = ['road', 'lane']

        scene_name = self.nusc.get('scene', self.nusc.get('sample', sample_token)['scene_token'])['name']
        map_name = self.scene2map[scene_name]
        nmap = self.maps[map_name]

        if 'road' in tgt_type:
            records = getattr(nmap, 'drivable_area')
            for record in records:
                polygons = [nmap.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
                for poly in polygons:
                    # plot exterior
                    ext = self.coord_transform(np.array(poly.exterior.coords), pose)
                    bev_seg_gt_road = cv2.fillPoly(bev_seg_gt_road, [ext], 1)
                    
                    # plot interior
                    intes = [self.coord_transform(np.array(pi.coords), pose) for pi in poly.interiors]
                    bev_seg_gt_road = cv2.fillPoly(bev_seg_gt_road, intes, 0)

        if 'lane' in tgt_type:
            for layer_name in ['road_divider', 'lane_divider']:
                records = getattr(nmap, layer_name)
                for record in records:
                    line = nmap.extract_line(record['line_token'])
                    if line.is_empty:
                        continue
                    line = self.coord_transform(np.array(line.xy).T, pose)
                    bev_seg_gt_lane = cv2.polylines(bev_seg_gt_lane, [line], isClosed=False,
                                                    color=1, thickness=self.lane_thickness)

        # need flip to aligh feature map
        bev_seg_gt_road = np.flip(bev_seg_gt_road, axis=1).copy()
        bev_seg_gt_lane = np.flip(bev_seg_gt_lane, axis=1).copy()
        bev_seg_gt = np.stack([bev_seg_gt_road, bev_seg_gt_lane],axis=-1)

        return bev_seg_gt
    """

    def evaluate(self, results, vis_mode=False, *args, **kwargs):

        eval_seg = 'bev_seg' in results[0]
        eval_det = 'boxes_3d' in results[0]
        assert eval_seg is True or eval_det is True

        new_bevseg_results = None
        new_det_results = None
        if eval_seg:
            new_bevseg_results = []
            new_bevseg_gts_road, new_bevseg_gts_lane = [], []
        if eval_det:
            new_det_results = []

        for i in range(len(results)):
            if eval_det:
                box_type = type(results[i]['boxes_3d'])
                boxes_3d = results[i]['boxes_3d'].tensor
                boxes_3d = box_type(boxes_3d, box_dim=9, origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
                new_det_results.append(dict(
                    boxes_3d=boxes_3d,
                    scores_3d=results[i]['scores_3d'],
                    labels_3d=results[i]['labels_3d']))
            """
            if eval_seg:
                assert results[i]['bev_seg'].shape[0] == 1
                seg_pred = results[i]['bev_seg'][0]
                seg_pred_road, seg_pred_lane = seg_pred[0], seg_pred[1]
                seg_pred_road = (seg_pred_road.sigmoid() > 0.5).int().data.cpu().numpy()
                seg_pred_lane = (seg_pred_lane.sigmoid() > 0.5).int().data.cpu().numpy()

                new_bevseg_results.append(dict(seg_pred_road=seg_pred_road,
                                               seg_pred_lane=seg_pred_lane))

                # bev seg gt path
                seg_gt_path = 'data/nuscenes/maps_bev_seg_gt_2class/'
                if not mmcv.is_filepath(seg_gt_path):
                    # online generate map, too slow
                    if i == 0:
                        print('### first time need generate bev seg map online ###')
                        print('### bev seg map is saved at:{} ###'.format(seg_gt_path))
                    sample_token = self.get_data_info(i)['sample_idx']
                    seg_gt = self._get_map_by_sample_token(sample_token)
                    seg_gt_road, seg_gt_lane = seg_gt[..., 0], seg_gt[..., 1]
                    mmcv.imwrite(seg_gt_road, seg_gt_path+'road/{}.png'.format(i), auto_mkdir=True)
                    mmcv.imwrite(seg_gt_lane, seg_gt_path+'lane/{}.png'.format(i), auto_mkdir=True)

                # load gt from local machine
                seg_gt_road = mmcv.imread(seg_gt_path+'road/{}.png'.format(i), flag='grayscale').astype('float64')
                seg_gt_lane = mmcv.imread(seg_gt_path+'lane/{}.png'.format(i), flag='grayscale').astype('float64')
                new_bevseg_gts_road.append(seg_gt_road)
                new_bevseg_gts_lane.append(seg_gt_lane)
            """

        if vis_mode:
            print('### vis nus test data ###')
            self.show(new_det_results, 'figs/test', bev_seg_results=new_bevseg_results, thr=0.3, fps=3)
            print('### finish vis ###')
            exit()

        result_dict = dict()

        # eval segmentation
        if eval_seg:
            bev_res_dict = self.evaluate_bev(new_bevseg_results,
                                             new_bevseg_gts_road,
                                             new_bevseg_gts_lane)
            for k in bev_res_dict.keys():
                result_dict[k] = bev_res_dict[k]
        # eval detection
        if eval_det:
            result_dict = super().evaluate(new_det_results, *args, **kwargs)
        return result_dict

    def evaluate_bev(self,
                     new_bevseg_results,
                     new_bevseg_gts_road,
                     new_bevseg_gts_lane):
        from mmseg.core import eval_metrics
        # from mmseg.ops import resize
        assert len(new_bevseg_results) == len(new_bevseg_gts_road) == len(new_bevseg_gts_lane)
        print('### evaluate BEV segmentation start ###')
        categories = ['road', 'lane']

        results_road, results_lane = [], []
        for i in range(len(new_bevseg_results)):
            seg_pred_road = new_bevseg_results[i]['seg_pred_road']
            seg_pred_road = cv2.resize(
                seg_pred_road, new_bevseg_gts_road[0].shape, interpolation=cv2.INTER_NEAREST)
            seg_pred_lane = new_bevseg_results[i]['seg_pred_lane']
            seg_pred_lane = cv2.resize(
                seg_pred_lane, new_bevseg_gts_lane[0].shape, interpolation=cv2.INTER_NEAREST)
            results_road.append(seg_pred_road)
            results_lane.append(seg_pred_lane)

        ret_metrics_road = eval_metrics(results_road,
                                        new_bevseg_gts_road,
                                        num_classes=2,
                                        ignore_index=255)

        ret_metrics_lane = eval_metrics(results_lane,
                                        new_bevseg_gts_lane,
                                        num_classes=2,
                                        ignore_index=255)

        IoU_road = ret_metrics_road['IoU'][-1]
        IoU_lane = ret_metrics_lane['IoU'][-1]
        IoU = [IoU_road, IoU_lane]
        res_dict = dict()

        for idx, c in enumerate(categories):
            print("{} IoU: {:.3f}".format(c, IoU[idx]))
            res_dict[c] = IoU[idx]
        print("mIoU: {:.3f}".format(sum(IoU) / len(IoU)))
        print('### evaluate BEV segmentation finish ###')
        return res_dict

"""
def get_scene2map(nusc):
    scene2map = {}
    for rec in nusc.scene:
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    return scene2map


def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def convert_points_to_local(points, pose):
    points -= pose[:2]
    rot = get_rot(np.arctan2(pose[3], pose[2])).T
    points = np.dot(points, rot)
    return points


def get_nusc_maps(map_folder='./data/nuscenes'):
    nusc_maps = {
        map_name: NuScenesMap(dataroot=map_folder, map_name=map_name) for map_name in
        [
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
            "singapore-onenorth",
        ]
    }
    return nusc_maps
"""
