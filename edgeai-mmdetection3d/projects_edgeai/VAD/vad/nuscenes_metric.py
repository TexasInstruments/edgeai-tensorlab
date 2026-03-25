# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import os
import json
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import mmengine
import numpy as np
import pyquaternion
import torch
import copy

#from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.datasets.transforms.formating import to_tensor
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmdet3d.registry import METRICS

from .bbox.nuscenes_box import CustomNuscenesBox

from .datasets.nuscenes_dataset import v1CustomDetectionConfig
from .datasets.nuscenes_dataset import VectorizedLocalMap
from .datasets.nuscenes_dataset import LiDARInstanceLines

from .datasets.vad_custom_nuscenes_eval import NuScenesEval_custom
 


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['bboxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    trajs = detection['trajs_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head

    box_list = []
    nus_box_dims = box_dims[:, [1, 0, 2]]

    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = CustomNuscenesBox(
            center=box_gravity_center[i],
            size=nus_box_dims[i],
            orientation=quat,
            fut_trajs=trajs[i],
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def output_to_vecs(detection):
    box3d = detection['map_boxes_3d'].numpy()
    scores = detection['map_scores_3d'].numpy()
    labels = detection['map_labels_3d'].numpy()
    pts = detection['map_pts_3d'].numpy()

    vec_list = []
    # import pdb;pdb.set_trace()
    for i in range(box3d.shape[0]):
        vec = dict(
            bbox = box3d[i], # xyxy
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list

def lidar_nusc_box_to_global(
        info: dict, boxes: List[NuScenesBox], classes: List[str],
        eval_configs: DetectionConfig) -> List[NuScenesBox]:
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

    Returns:
        List[:obj:`DetectionConfig`]: List of standard NuScenesBoxes in the
        global coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        box.rotate(
            pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
        box.translate(lidar2ego[:3, 3])
        # filter det in ego.
        cls_range_x_map = eval_configs.class_range_x
        cls_range_y_map = eval_configs.class_range_y
        #cls_range_map = eval_configs.class_range

        x_distance, y_distance = box.center[0], box.center[1]
        det_range_x = cls_range_x_map[classes[box.label]]
        det_range_y = cls_range_y_map[classes[box.label]]
        #[det_range_x, det_range_y] = cls_range_map[classes[box.label]]
        if abs(x_distance) > det_range_x or abs(y_distance) > det_range_y:
            continue

        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list



@METRICS.register_module()
class VADNuScenesMetric(NuScenesMetric):
    """Nuscenes evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        jsonfile_prefix (str, optional): The prefix of json files including the
            file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        eval_version (str): Configuration version of evaluation.
            Defaults to 'detection_cvpr_2019'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 map_metric: Union[str, List[str]] = 'chamfer',
                 map_classes: Union[str, List[str]] = ['divider', 'ped_crossing', 'boundary'],
                 map_ann_file: Optional[str] = None,
                 map_fixed_ptsnum_per_line=-1,
                 map_eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 use_pkl_result=False,
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 eval_version: str = 'detection_cvpr_2019',
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'NuScenes metric'
        super(NuScenesMetric, self).__init__(prefix, collect_device)

        # From NuScenesMetric.init()
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.format_only = format_only
        if self.format_only:
            assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'

        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args

        self.metrics = metric if isinstance(metric, list) else [metric]

        self.eval_version = eval_version
        # End of NuScenesMetric.init()

        # Check if config exists.
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(this_dir, '%s.json' % self.eval_version)
        assert os.path.exists(cfg_path), \
            'Requested unknown configuration {}'.format(self.eval_version)

        # Load config file and deserialize it.
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        self.custom_eval_detection_configs = v1CustomDetectionConfig.deserialize(data)

        # Set self.map_metric
        self.map_metric = map_metric
        self.map_classes = map_classes
        self.map_ann_file = map_ann_file
        self.num_map_classes = len(self.map_classes)
        self.use_pkl_result = use_pkl_result

        self.pc_range = pc_range
        patch_h = pc_range[4]-pc_range[1]
        patch_w = pc_range[3]-pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = map_fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = map_eval_use_same_gt_sample_num_flag
        self.vector_map = VectorizedLocalMap(self.data_root, 
                            patch_size=self.patch_size, map_classes=self.map_classes, 
                            fixed_ptsnum_per_line=map_fixed_ptsnum_per_line,
                            padding_value=self.padding_value)
        
        self.overlap_test = False


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            result['metric_results'] = data_sample['metric_results']
            result['pts_bbox'] = data_sample['pts_bbox']
            self.results.append(result)

    def evaluate(self, size: int) -> dict:
        result_metric_names = ['EPA', 'ADE', 'FDE', 'MR']
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']

        all_metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                all_metric_dict[met+'_'+cls] = 0.0
        result_dict = {}
        for met in result_metric_names:
            for cls in motion_cls_names:
                result_dict[met+'_'+cls] = 0.0

        alpha = 0.5

        for i in range(len(self.results)):
            for key in all_metric_dict.keys():
                all_metric_dict[key] += self.results[i]['metric_results'][key]

        for cls in motion_cls_names:
            result_dict['EPA_'+cls] = (all_metric_dict['hit_'+cls] - \
                 alpha * all_metric_dict['fp_'+cls]) / max(all_metric_dict['gt_'+cls], 1.0)
            result_dict['ADE_'+cls] = all_metric_dict['ADE_'+cls] / max(all_metric_dict['cnt_ade_'+cls], 1.0)
            result_dict['FDE_'+cls] = all_metric_dict['FDE_'+cls] / max(all_metric_dict['cnt_fde_'+cls], 1.0)
            result_dict['MR_'+cls] = all_metric_dict['MR_'+cls] / max(all_metric_dict['cnt_fde_'+cls], 1.0)

        print('\n')
        print('-------------- Motion Prediction --------------')
        for k, v in result_dict.items():
            print(f'{k}: {v}')

        # NOTE: print planning metric
        print('\n')
        print('-------------- Planning --------------')
        metric_dict = None
        num_valid = 0
        for res in self.results:
            if res['metric_results']['fut_valid_flag']:
                num_valid += 1
            else:
                continue
            if metric_dict is None:
                metric_dict = copy.deepcopy(res['metric_results'])
            else:
                for k in res['metric_results'].keys():
                    metric_dict[k] += res['metric_results'][k]
        
        for k in metric_dict:
            metric_dict[k] = metric_dict[k] / num_valid
            print("{}:{}".format(k, metric_dict[k]))

        # Read data_infos
        self.data_infos = mmengine.load(
            self.ann_file, backend_args=self.backend_args)['data_list']

        # sort data_infos
        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))

        self.version = self.dataset_meta['version']
        self.classes = self.dataset_meta['classes']

        result_files, tmp_dir = self.format_results(self.results, self.jsonfile_prefix)

        result_names = ['pts_bbox']
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=self.metrics, map_metric=self.map_metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=self.metrics, map_metric=self.map_metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        #if show:
        #    self.show(results, out_dir, pipeline=pipeline)
        return results_dict


    def vectormap_pipeline(self, example, input_dict):
        '''
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        '''
        e2g_matrix = np.array(input_dict['ego2global'])
        l2e_matrix = np.array(input_dict['lidar_points']['lidar2ego'])
        lidar2global = e2g_matrix @ l2e_matrix

        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global[:3, :3], rtol=1e-05, atol=1e-05).q)

        location = input_dict['map_location']
        anns_results = self.vector_map.gen_vectorized_samples(
            location, lidar2global_translation, lidar2global_rotation
        )
        
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                # empty tensor, will be passed in train, 
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc

        example['map_gt_labels_3d'] = gt_vecs_label
        example['map_gt_bboxes_3d'] = gt_vecs_pts_loc

        return example


    def _format_gt(self, dataset_length):
        gt_annos = []
        print('Start to convert gt map format...')
        # assert self.map_ann_file is not None
        if (not os.path.exists(self.map_ann_file)) :
            #dataset_length = len(self)
            #dataset_length = len(self.data_infos)
            prog_bar = mmengine.utils.ProgressBar(dataset_length)
            mapped_class_names = self.map_classes
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['token']
                gt_anno = {}
                gt_anno['sample_token'] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, self.data_infos[sample_id])
                gt_labels = gt_sample_dict['map_gt_labels_3d'].numpy()
                gt_vecs = gt_sample_dict['map_gt_bboxes_3d'].instance_list
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords)),
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors']=gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'GTs': gt_annos
            }
            print('\n GT anns writes to', self.map_ann_file)
            mmengine.dump(nusc_submissions, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')


    def _format_bbox(self, results, jsonfile_prefix=None, score_thresh=0.2):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        det_mapped_class_names = self.classes #self.dataset_meta['classes']

        # assert self.map_ann_file is not None
        map_pred_annos = {}
        map_mapped_class_names = self.map_classes

        plan_annos = {}

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']

            plan_annos[sample_token] = [det['ego_fut_preds'], det['ego_fut_cmd']]

            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             det_mapped_class_names,
                                             self.custom_eval_detection_configs)
            for i, box in enumerate(boxes):
                if box.score < score_thresh:
                    continue
                name = det_mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                    fut_traj=box.fut_trajs.tolist())
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos


            map_pred_anno = {}
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]['token']
            map_pred_anno['sample_token'] = sample_token
            pred_vec_list=[]
            for i, vec in enumerate(vecs):
                name = map_mapped_class_names[vec['label']]
                anno = dict(
                    # sample_token=sample_token,
                    pts=vec['pts'],
                    pts_num=len(vec['pts']),
                    cls_name=name,
                    type=vec['label'],
                    confidence_level=vec['score'])
                pred_vec_list.append(anno)
                # annos.append(nusc_anno)
            # nusc_annos[sample_token] = annos
            map_pred_anno['vectors'] = pred_vec_list
            map_pred_annos[sample_token] = map_pred_anno

        if not os.path.exists(self.map_ann_file):
            self._format_gt(len(results))
        else:
            print(f'{self.map_ann_file} exist, not update')
        # with open(self.map_ann_file,'r') as f:
        #     GT_anns = json.load(f)
        # gt_annos = GT_anns['GTs']

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
            'map_results': map_pred_annos,
            'plan_results': plan_annos
            # 'GTs': gt_annos
        }

        mmengine.mkdir_or_exist(jsonfile_prefix)
        if self.use_pkl_result:
            res_path = osp.join(jsonfile_prefix, 'results_nusc.pkl')
        else:
            res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmengine.dump(nusc_submissions, res_path)
        return res_path


    def format_results(self, results, classes=None, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        if isinstance(results, dict):
            # print(f'results must be a list, but get dict, keys={results.keys()}')
            # assert isinstance(results, list)
            results = results['bbox_results']
        assert isinstance(results, list)
        #assert len(results) == len(self), (
        #    'The length of results is not equal to the dataset len: {} != {}'.
        #    format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name == 'metric_results':
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir


    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         map_metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        detail = dict()
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=False)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.custom_eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
        metric_prefix = f'{result_name}_NuScenes'
        classes = self.dataset_meta['classes']
        for name in classes:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']


        from .datasets.map_utils.mean_ap import eval_map
        from .datasets.map_utils.mean_ap import format_res_gt_by_classes
        result_path = osp.abspath(result_path)
        
        print('Formating results & gts by classes')
        pred_results = mmengine.load(result_path)
        map_results = pred_results['map_results']
        gt_anns = mmengine.load(self.map_ann_file)
        map_annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     map_results,
                                                     map_annotations,
                                                     cls_names=self.map_classes,
                                                     num_pred_pts_per_instance=self.fixed_num,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range)
        map_metrics = map_metric if isinstance(map_metric, list) else [map_metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in map_metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        for metric in map_metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)
            if metric == 'chamfer':
                thresholds = [0.5,1.0,1.5]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds),self.num_map_classes))
            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                map_results,
                                map_annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.map_classes,
                                logger=logger,
                                num_pred_pts_per_instance=self.fixed_num,
                                pc_range=self.pc_range,
                                metric=metric)
                for j in range(self.num_map_classes):
                    cls_aps[i, j] = cls_ap[j]['ap']
            for i, name in enumerate(self.map_classes):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()
            for i, name in enumerate(self.map_classes):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail

