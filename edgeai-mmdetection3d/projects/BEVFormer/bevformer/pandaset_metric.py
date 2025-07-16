from typing import Dict, List
from os import path as osp

from mmengine import load
from mmengine.logging import MMLogger
from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics.pandaset_metric import PandaSetMetric

@METRICS.register_module()
class CustomPandaSetMetric(PandaSetMetric):

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        self.bbox_type_3d = type(results[0]['pred_instances_3d']['bboxes_3d'])
        self.NameMapping = dict(map(lambda x:(x,x),self.dataset_meta['classes']))

        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']

        # load annotations
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)['data_list']

        # sort data_infos
        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))

        result_dict, tmp_dir = self.format_results(results, classes,
                                                   self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            return metric_dict

        for metric in self.metrics:
            ap_dict = self.nus_evaluate(
                result_dict, classes=classes, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict


