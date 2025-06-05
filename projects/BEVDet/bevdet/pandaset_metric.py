from typing import Dict, List
from os import path as osp

from mmengine import load
from mmengine.logging import MMLogger
from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics.pandaset_metric import PandaSetMetric
from mmdet3d.structures import LiDARInstance3DBoxes

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


    def load_gt_bboxes(self, classes, class_mapping_func):
        def create_instance_list(instances, gt_ego_boxes, sample_token, info=None, cam_type=None):
            res_instances = []
            index = 0
            for instance in instances:
                if not instance['bbox_3d_isvalid']:
                    continue
                box = instance['bbox_3d']
                name = classes[class_mapping_func(instance['bbox_label_3d'])]
                label = class_mapping_func(instance['bbox_label_3d'])
                attr = self.get_attr_name(instance['attr_label'], name)
                name = classes[label]
                #velocity = instance['velocity']

                #if self.bbox_type_3d == CameraInstance3DBoxes:
                #    assert info is not None and cam_type is not None, 'info and cam_type should be provided for CameraInstance3DBoxes'
                #    velocity = np.array(velocity)
                #    corners = convert_bbox_to_corners_for_camera(box)
                #    cam2global = np.array(info['images'][cam_type]['cam2global'])
                #    global_corners = corners @ cam2global[:3,:3].T + cam2global[:3,3]
                #    velocity = velocity @ cam2global[:3,:3].T
                #    box = convert_corners_to_bbox_for_lidar_box(global_corners)
                #if isinstance(velocity, list):
                #    velocity = velocity[:2]
                #else:
                #    velocity = velocity.tolist()[:2]
                res_instances.append(dict(
                    sample_token=sample_token,
                    translation=gt_ego_boxes[index][:3],
                    size=gt_ego_boxes[index][3:6],
                    yaw=gt_ego_boxes[index][6],
                    velocity=gt_ego_boxes[index][7:9],
                    detection_name=name,
                    attribute_name=attr
                ))
                index += 1
            return res_instances

        gt_boxes = {}
        if self.bbox_type_3d == LiDARInstance3DBoxes:
            for info in self.data_infos:
                sample_token = info['token']
                if sample_token in gt_boxes:
                    gt_boxes[sample_token].extend(create_instance_list(info['instances'], info['ann_infos'][0], sample_token, info))
                else:
                    gt_boxes[sample_token] = create_instance_list(info['instances'], info['ann_infos'][0], sample_token, info)
        else:
            raise NotImplementedError

        return gt_boxes
