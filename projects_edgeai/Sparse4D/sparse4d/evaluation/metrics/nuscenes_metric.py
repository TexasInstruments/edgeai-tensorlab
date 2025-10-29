# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Dict, List, Optional, Union
from mmengine import load
from mmengine.logging import MMLogger
from nuscenes.eval.common.config import config_factory as track_configs

from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric


@METRICS.register_module()
class Sparse4DNuScenesMetric(NuScenesMetric):
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
                 tracking: Optional[bool] = False,
                 tracking_threshold: Optional[float] = None,
                 backend_args: Optional[dict] = None) -> None:
        super(Sparse4DNuScenesMetric, self).__init__(
            data_root, ann_file, metric, modality, prefix, format_only,
            jsonfile_prefix, eval_version, collect_device, backend_args)

        self.tracking = tracking
        self.tracking_threshold = tracking_threshold
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)


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
        for metric in ["detection", "tracking"]:
            tracking = metric == "tracking"
            if tracking and not self.tracking:
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

        return metric_dict