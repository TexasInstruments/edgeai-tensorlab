# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Dict, List, Optional, Sequence, Union
import os
import prettytable
import torch
import mmengine
import numpy as np
from mmengine import load
from mmengine.logging import print_log, MMLogger
from nuscenes.eval.common.config import config_factory

from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics.nuscenes_metric import (
    NuScenesMetric, output_to_nusc_box, lidar_nusc_box_to_global)

@METRICS.register_module()
class SparseDriveNuScenesMetric(NuScenesMetric):
    """Nuscenes evaluation metric with data sorted.

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
    MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
    )

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 eval_version: str = 'detection_cvpr_2019',
                 track3d_eval_version: str = "tracking_nips_2019",
                 collect_device: str = 'cpu',
                 with_det: Optional[bool] = True,
                 with_tracking: Optional[bool] = False,
                 with_map: Optional[bool] = False,
                 with_motion: Optional[bool] = False,
                 with_planning: Optional[bool] = False,
                 tracking_threshold: Optional[float] = None,
                 motion_threshold: Optional[float] = None,
                 eval_config: Optional[dict] = None,
                 backend_args: Optional[dict] = None) -> None:
        super(SparseDriveNuScenesMetric, self).__init__(
            data_root, ann_file, metric, modality, prefix, format_only,
            jsonfile_prefix, eval_version, collect_device, backend_args)

        # Always enable detection evaluation
        self.with_det = True #with_det
        self.with_tracking = with_tracking
        self.with_map = with_map
        self.with_motion = with_motion
        self.with_planning = with_planning
        self.tracking_threshold = tracking_threshold
        self.motion_threshold = motion_threshold
        self.eval_config = eval_config
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = config_factory(self.track3d_eval_version)

        self.CLASSES = self.eval_config.dataset.metainfo.classes
        self.MAP_CLASSES = self.eval_config.dataset.map_classes


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
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                if torch.is_tensor(pred_3d[attr_name]):
                    pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']

        # load annotations
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)['data_list']
        # sort data_infos
        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))

        metric_dict = {}
        if True: #self.with_det: # Always True
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.with_tracking:
                    continue

                result_dict, tmp_dir = self.format_results(results,
                                                           classes,
                                                           self.jsonfile_prefix,
                                                           tracking=tracking,
                                                           tracking_threshold=self.tracking_threshold)

                if self.format_only:
                    logger.info(
                        f'results are saved in {osp.basename(self.jsonfile_prefix)}')
                else:
                    for metric in self.metrics:
                        ap_dict = self.nus_evaluate(
                            result_dict, classes=classes, metric=metric, tracking=tracking, logger=logger)
                        for result in ap_dict:
                            metric_dict[result] = ap_dict[result]

                if tmp_dir is not None:
                    tmp_dir.cleanup()

        output_dir = osp.join(*osp.split(result_dict['pred_instances_3d'])[:-1])
        if self.with_map:
            from ..map.vector_eval import VectorEvaluate
            map_evaluator = VectorEvaluate(self.eval_config)
            result_path = self.format_map_results(results, prefix=output_dir)
            map_results_dict = map_evaluator.evaluate(result_path, logger=logger)
            metric_dict.update(map_results_dict)

        if self.with_motion:
            thresh = self.motion_threshold
            result_files = self.format_motion_results(results, jsonfile_prefix=self.jsonfile_prefix, thresh=thresh)
            motion_results_dict = self._evaluate_single_motion(result_files, output_dir, logger=logger)
            metric_dict.update(motion_results_dict)

        if self.with_planning:
            from ..planning.planning_eval import planning_eval
            planning_results_dict = planning_eval(results, self.eval_config, logger=logger)
            metric_dict.update(planning_results_dict)
        
        return metric_dict

    """
    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        from nuscenes import NuScenes

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False
        )
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval

            nusc_eval = NuScenesEval(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                    ] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval

            nusc_eval = TrackingEval(
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics = nusc_eval.main()

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail
    """

    def format_map_results(self, results, prefix=None):
        submissions = {'results': {},}
        
        for j, pred in enumerate(results):
            '''
            For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
            'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                contain all vectors predicted in this sample.
            'scores: List of score(float), 
                contain scores of all instances in this sample.
            'labels': List of label(int), 
                contain labels of all instances in this sample.
            '''
            if pred is None: # empty prediction
                continue
            pred = pred['pred_instances_3d']

            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = self.data_infos[j]['token']
            for i in range(len(pred['scores'])):
                score = pred['scores'][i].item()
                label = pred['labels'][i].item()
                vector = np.array(pred['vectors'][i])

                # A line should have >=2 points
                if len(vector) < 2:
                    continue
                
                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)
            
            submissions['results'][token] = single_case
        
        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmengine.dump(submissions, out_path)
        return out_path

    def format_motion_results(self, results, jsonfile_prefix=None, tracking=False, thresh=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, _ = output_to_nusc_box(
                det['pred_instances_3d'], threshold=None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.eval_detection_configs,
                filter_with_cls_range=False,
            )
            for i, box in enumerate(boxes):
                if thresh is not None and box.score < thresh:
                    continue
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenesMetric.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenesMetric.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )
                nusc_anno.update(
                    dict(
                        trajs=det['pred_instances_3d']['trajs_3d'][i].numpy(),
                    )
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        return nusc_submissions 

    def _evaluate_single_motion(self,
                         results,
                         result_path,
                         logger=None,
                         metric='bbox',
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
        from nuscenes import NuScenes
        from ..motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion

        output_dir = result_path
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEvalMotion(
            nusc,
            config=self.eval_detection_configs,
            results=results,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            seconds=6)
        metrics = nusc_eval.main(render_curves=False)
        
        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n'+str(table), logger=logger)
        return metrics


    '''
    def evaluate(
        self,
        results,
        eval_mode,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        res_path = "results.pkl" if "trainval" in self.version else "results_mini.pkl"
        res_path = osp.join(self.work_dir, res_path)
        print('All Results write to', res_path)
        mmcv.dump(results, res_path)

        results_dict = dict()
        if eval_mode['with_det']:
            self.tracking = eval_mode["with_tracking"]
            self.tracking_threshold = eval_mode["tracking_threshold"]
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.tracking:
                    continue
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix=self.work_dir, tracking=tracking
                )

                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(
                            result_files[name], tracking=tracking
                        )
                    results_dict.update(ret_dict)
                elif isinstance(result_files, str):
                    ret_dict = self._evaluate_single(
                        result_files, tracking=tracking
                    )
                    results_dict.update(ret_dict)

                if tmp_dir is not None:
                    tmp_dir.cleanup()

        if eval_mode['with_map']:
            from .evaluation.map.vector_eval import VectorEvaluate
            self.map_evaluator = VectorEvaluate(self.eval_config)
            result_path = self.format_map_results(results, prefix=self.work_dir)
            map_results_dict = self.map_evaluator.evaluate(result_path, logger=logger)
            results_dict.update(map_results_dict)

        if eval_mode['with_motion']:
            thresh = eval_mode["motion_threshhold"]
            result_files = self.format_motion_results(results, jsonfile_prefix=self.work_dir, thresh=thresh)
            motion_results_dict = self._evaluate_single_motion(result_files, self.work_dir, logger=logger)
            results_dict.update(motion_results_dict)
        
        if eval_mode['with_planning']:
            from .evaluation.planning.planning_eval import planning_eval
            planning_results_dict = planning_eval(results, self.eval_config, logger=logger)
            results_dict.update(planning_results_dict)

        """
        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        """

        # print main metrics for recording
        metric_str = '\n'
        if "img_bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("img_bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("img_bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("img_bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("img_bbox_NuScenes/mAOE"):.4f}\n' 
            metric_str += f'mAVE: {results_dict.get("img_bbox_NuScenes/mAVE"):.4f}\n' 
            metric_str += f'mAAE: {results_dict.get("img_bbox_NuScenes/mAAE"):.4f}\n' 
            metric_str += f'NDS: {results_dict.get("img_bbox_NuScenes/NDS"):.4f}\n\n'
        
        if "img_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["img_bbox_NuScenes/amota"]:.4f}\n' 
            metric_str += f'AMOTP: {results_dict["img_bbox_NuScenes/amotp"]:.4f}\n' 
            metric_str += f'RECALL: {results_dict["img_bbox_NuScenes/recall"]:.4f}\n' 
            metric_str += f'MOTAR: {results_dict["img_bbox_NuScenes/motar"]:.4f}\n' 
            metric_str += f'MOTA: {results_dict["img_bbox_NuScenes/mota"]:.4f}\n' 
            metric_str += f'MOTP: {results_dict["img_bbox_NuScenes/motp"]:.4f}\n' 
            metric_str += f'IDS: {results_dict["img_bbox_NuScenes/ids"]}\n\n' 
        
        if "mAP_normal" in results_dict:
            metric_str += f'ped_crossing= {results_dict["ped_crossing"]:.4f}\n' 
            metric_str += f'divider= {results_dict["divider"]:.4f}\n' 
            metric_str += f'boundary= {results_dict["boundary"]:.4f}\n' 
            metric_str += f'mAP_normal= {results_dict["mAP_normal"]:.4f}\n\n' 

        if "car_EPA" in results_dict:
            metric_str += f'Car / Ped\n' 
            metric_str += f'epa= {results_dict["car_EPA"]:.4f} / {results_dict["pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["car_min_ade_err"]:.4f} / {results_dict["pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["car_min_fde_err"]:.4f} / {results_dict["pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["car_miss_rate_err"]:.4f} / {results_dict["pedestrian_miss_rate_err"]:.4f}\n\n' 

        if "L2" in results_dict:
            metric_str += f'obj_box_col: {(results_dict["obj_box_col"]*100):.3f}%\n'
            metric_str += f'L2: {results_dict["L2"]:.4f}\n\n'
        
        print_log(metric_str, logger=logger)
        return results_dict
    '''
