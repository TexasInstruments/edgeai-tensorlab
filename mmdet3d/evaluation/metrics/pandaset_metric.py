import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
import time

import mmengine
import numpy as np
import pyquaternion
import torch
from mmengine import Config, load
from .nuscenes_metric import NuScenesMetric
from mmengine.logging import MMLogger

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)
from mmdet3d.datasets.convert_utils import (convert_bbox_to_corners_for_lidar, 
                                            convert_bbox_to_corners_for_camera, 
                                            convert_corners_to_bbox_for_cam_box, 
                                            convert_corners_to_bbox_for_lidar_box)

from mmdet3d.datasets.pandaset_dataset import UNIQUE_ATTRIBUTE_LABELS,CLASSES

@METRICS.register_module()
class PandaSetMetric(NuScenesMetric):
    default_attrinutes = {
        'Car':'vehicle.Parked', 
        'Pickup Truck':'vehicle.Parked', 
        'Medium-sized Truck':'vehicle.Parked', 
        'Semi-truck':'vehicle.Parked',     
        'Towed Object':'vehicle.Parked', 
        'Other Vehicle - Construction Vehicle':'vehicle.Parked',
        'Other Vehicle - Uncommon':'vehicle.Parked', 
        'Other Vehicle - Pedicab':'vehicle.Parked',
        'Bus':'vehicle.Parked', 
        'Train':'vehicle.Parked', 
        'Trolley':'vehicle.Parked', 
        'Tram / Subway':'vehicle.Parked',
        'Emergency Vehicle':'emergency_vehicle.Parked.Lights not Flashing',
        'Pedestrian':'Adult.Standing', 
        'Pedestrian with Object':'Adult.Standing',
        'Motorcycle':'vehicle.Parked', 
        'Personal Mobility Device':'vehicle.Parked', 
        'Motorized Scooter':'vehicle.Parked',
        'Bicycle':'vehicle.Parked', 
        'Animals - Other':'vehicle.Parked',
    }
    

    
    def __init__(self, data_root, ann_file, metric = 'bbox', modality = None, prefix = None, format_only = False, jsonfile_prefix = None, eval_version = 'detection_cvpr_2019', collect_device = 'cpu', backend_args = None, max_dists=None):
        super().__init__(data_root, ann_file, metric, modality, prefix, format_only, jsonfile_prefix, eval_version, collect_device, backend_args)
        if max_dists is None:
            self.max_dist_func = lambda cls: 50
        elif isinstance(max_dists, dict):
            self.max_dist_func = lambda cls: max_dists.get(cls, 50)
        elif isinstance (max_dists, (int, float)):
            self.max_dist_func = lambda cls: max_dists
        elif isinstance(max_dists, (tuple, list)):
            max_dists = dict((self.dataset_meta['classes'],max_dists))
    
    def get_attr_name(self, attr_idx, label_name):
        attr_mapping = UNIQUE_ATTRIBUTE_LABELS
        if attr_idx == -1:
            return 'None'
        
        if label_name in ('Car', 'Pickup Truck', 'Medium-sized Truck', 'Semi-truck',     
                'Towed Object', 'Other Vehicle - Construction Vehicle',
                'Other Vehicle - Uncommon', 'Other Vehicle - Pedicab',
                'Bus', 'Train', 'Trolley', 'Tram / Subway',):
            if attr_idx in list(range(13,22)):
                return attr_mapping[attr_idx]
            return self.default_attrinutes[label_name]
        
        if label_name == 'Emergency Vehicle':
            if attr_idx in list(range(9,13)):
                return attr_mapping[attr_idx]
            return self.default_attrinutes[label_name]
        
        if label_name in ('Pedestrian', 'Pedestrian with Object'):
            if attr_idx in list(range(8)):
                return attr_mapping[attr_idx]
            return self.default_attrinutes[label_name]
        
        if label_name in ('Motorcycle', 'Personal Mobility Device', 'Motorized Scooter',
                        'Bicycle', 'Animals - Other',):
            if attr_idx in list(range(13,22)):
                return attr_mapping[attr_idx]
            return self.default_attrinutes[label_name]
        
        return 'None'
    
    def _format_lidar_bbox(self, results, sample_idx_list, classes = None, jsonfile_prefix = None):
        # return
        nusc_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes = det['bboxes_3d']
            if 'attr_labels' in det:
                attrs = det['attr_labels'].to(self.collect_device).numpy().tolist()
            else:
                attrs = None
            scores = det['scores_3d'].to(self.collect_device).numpy().tolist()
            labels = det['labels_3d'].to(self.collect_device).numpy().tolist()
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]['token']
            for i in range(boxes.tensor.shape[0]):
                box = boxes.tensor[i]
                name = classes[labels[i]]
                if attrs:
                    attr = self.get_attr_name(attrs[i], name)
                elif np.linalg.norm(box[7:9]) > 0.2:
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
                    translation=box[0:3].tolist(),
                    size=box[3:6].tolist(),
                    yaw=box[6:7].tolist(),
                    velocity=box[7:9].tolist(),
                    detection_name=name,
                    detection_score=scores[i],
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_pandaset.json')
        print(f'Results writes to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path

    def _format_camera_bbox(self, results, sample_idx_list, classes = None, jsonfile_prefix = None):
        """Convert the results to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        # return
        nusc_annos = {}

        print('Start to convert detection format...')

        # Camera types in Nuscenes datasets
        camera_types = [
            'back_camera',
            'front_camera',
            'front_left_camera',
            'front_right_camera',
            'left_camera',
            'right_camera',
        ]

        CAM_NUM = 6
        if classes is None:
            NUM_CLASSES = 0
        else:
            NUM_CLASSES = len(classes)
        # l = [ det['bboxes_3d'].tensor.shape[0] for det in results]
        # print(l)
        for i, det in enumerate(mmengine.track_iter_progress(results)):

            sample_idx = sample_idx_list[i]

            frame_sample_idx = sample_idx // CAM_NUM
            camera_type_id = sample_idx % CAM_NUM
            camera_type = camera_types[camera_type_id]

            if camera_type_id == 0:
                corners_per_frame = []
                velocities_per_frame = []
                attrs_per_frame = []
                labels_per_frame = []
                scores_per_frame = []

            # need to merge results from images of the same sample
            info = self.data_infos[frame_sample_idx]
            sample_token = info['token']
            annos = []
            boxes = det['bboxes_3d']
            ego_corners, velocities = cam_bbox_to_ego_corners3d(boxes, info, camera_type)
            attrs = det['attr_labels'].to(self.collect_device).numpy().tolist()
            scores = det['scores_3d'].to(self.collect_device).numpy().tolist()
            labels = det['labels_3d'].to(self.collect_device).numpy().tolist()
            corners_per_frame.extend(ego_corners.tolist())
            velocities_per_frame.extend(velocities.tolist())
            attrs_per_frame.extend(attrs)
            labels_per_frame.extend(labels)
            scores_per_frame.extend(scores)
            # Remove redundant predictions caused by overlap of images
            if (sample_idx + 1) % CAM_NUM != 0:
                continue
            assert len(corners_per_frame) == len(velocities_per_frame) == len(attrs_per_frame) == len(labels_per_frame) == len(scores_per_frame)
            # cam_boxes3d = get_cam3d_boxes_per_frame(boxes_per_frame)
            scores= torch.Tensor(scores_per_frame).to(self.collect_device)
            nms_scores = scores.new_zeros(scores.shape[0],NUM_CLASSES+1)
            labels = torch.LongTensor(labels_per_frame).to(self.collect_device)
            indices = labels.new_tensor(list(range(scores.shape[0])))
            nms_scores[indices,labels] = scores
            scores = nms_scores
            ego_corners= np.array(corners_per_frame)
            velocities = velocities_per_frame
            cam_boxes3d = ego_corners3d_to_cam_bbox(ego_corners, velocities, info)
            
            # box nms 3d over 6 images in a frame
            # TODO: move this global setting into config
            nms_cfg = dict(
                use_rotate_nms=True,
                nms_across_levels=False,
                nms_pre=4096,
                nms_thr=0.05,
                score_thr=0.05,
                min_bbox_size=0,
                max_per_frame=500)
            nms_cfg = Config(nms_cfg)
            cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
            boxes3d = cam_boxes3d.tensor
            # generate attr scores from attr labels
            attrs = labels.new_tensor([attr for attr in attrs_per_frame])
            boxes3d, scores, labels, attrs = box3d_multiclass_nms(
                boxes3d,
                cam_boxes3d_for_nms,
                scores,
                nms_cfg.score_thr,
                nms_cfg.max_per_frame,
                nms_cfg,
                mlvl_attr_scores=attrs)
            cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
            det = bbox3d2result(cam_boxes3d, scores, labels, attrs)
            
            boxes = det['bboxes_3d']
            attrs = det['attr_labels'].to(self.collect_device).tolist()
            scores = det['scores_3d'].to(self.collect_device).tolist()
            labels = det['labels_3d'].to(self.collect_device).tolist()
            ego_corners , velocities = cam_bbox_to_ego_corners3d(boxes, info, camera_type)
            boxes = ego_corners3d_to_ego_bbox(ego_corners, velocities, )
            
            for i in range(boxes.tensor.shape[0]):
                box = boxes.tensor[i]

                name = classes[labels[i]]
                attr = self.get_attr_name(attrs[i], name)
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box[0:3].tolist(),
                    size=box[3:6].tolist(),
                    yaw=box[6].tolist(),
                    velocity=box[7:9].tolist(),
                    detection_name=name,
                    detection_score=scores[i],
                    attribute_name=attr)
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_pandaset.json')
        print(f'Results writes to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path
    
    def compute_metrics(self, results):
        self.bbox_type_3d = type(results[0]['pred_instances_3d']['bboxes_3d'])
        self.NameMapping = dict(map(lambda x:(x,x),self.dataset_meta['classes']))
        return super().compute_metrics(results)
    
    def _evaluate_single(self, result_path, classes = None, result_name = 'pred_instances_3d'):
        output_dir = osp.join(*osp.split(result_path)[:-1])
        detail = dict()
        
        metrics = self.pandaset_evaluate(result_path)
        metric_prefix = f'{result_name}_PandaSet'
        for name in classes:
            for k, v in metrics['mean_dist_aps'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail
    
    def pandaset_evaluate(self, result_path):
        # load result json file
        data = mmengine.load(result_path)
        pred_boxes = data['results']
        meta = data['meta']
        classes =self.dataset_meta['classes']
        class_mapping = self.dataset_meta.get('class_mapping', list(range(len(self.dataset_meta['classes']))))
        gt_boxes = self.load_gt_bboxes(classes, class_mapping)
        gt_boxes = filter_eval_boxes(gt_boxes, self.max_dist_func)
        pred_boxes = filter_eval_boxes(pred_boxes, self.max_dist_func)
        dist_ths = [0.5, 1.0, 2.0, 4.0]
        metrics = pandaset_evaluate_matrics(pred_boxes, gt_boxes, classes, dist_ths, 2.0)
        return metrics

    def load_gt_bboxes(self, classes, class_mapping):
        def create_instance_list(instances, sample_token, info=None, cam_type=None):
            res_instances = []
            for instance in instances:
                if not instance['bbox_3d_isvalid']:
                    continue
                box = instance['bbox_3d']
                name = classes[class_mapping[instance['bbox_label_3d']]]
                label = class_mapping[instance['bbox_label_3d']]
                attr = self.get_attr_name(instance['attr_label'], name)
                name = classes[label]
                velocity = instance['velocity'] 
                if self.bbox_type_3d == CameraInstance3DBoxes:
                    assert info is not None and cam_type is not None, 'info and cam_type should be provided for CameraInstance3DBoxes'
                    velocity = np.array(velocity)
                    corners = convert_bbox_to_corners_for_camera(box)
                    cam2ego = np.array(info['images'][cam_type]['cam2ego'])
                    ego_corners = corners @ cam2ego[:3,:3].T + cam2ego[:3,3]
                    velocity = velocity @ cam2ego[:3,:3].T
                    box = convert_corners_to_bbox_for_lidar_box(ego_corners)
                velocity = velocity.tolist()[:2]
                res_instances.append(dict(
                    sample_token=sample_token,
                    translation=box[:3],
                    size=box[3:6],
                    yaw=box[6],
                    velocity=velocity,
                    detection_name=name,

                    attribute_name=attr
                ))
            return res_instances
        
        gt_boxes = {}
        if self.bbox_type_3d == LiDARInstance3DBoxes:
            for info in self.data_infos:
                sample_token = info['token']
                if sample_token in gt_boxes:
                    gt_boxes[sample_token].extend(create_instance_list(info['instances'], sample_token, info,))
                else:
                    gt_boxes[sample_token] = create_instance_list(info['instances'], sample_token, info)
                
        elif self.bbox_type_3d == CameraInstance3DBoxes:
            for info in self.data_infos:
                sample_token = info['token']
                
                for name, instacnes in info['cam_instances'].items():
                    if sample_token in gt_boxes:
                        gt_boxes[sample_token].extend(create_instance_list(instacnes, sample_token, info, name))
                    else:
                        gt_boxes[sample_token] = create_instance_list(instacnes, sample_token, info, name)
                    
        else:
            raise NotImplementedError
        
        return gt_boxes

def cam_bbox_to_ego_corners3d(boxes, info, camera_type):
    if isinstance(boxes, CameraInstance3DBoxes):
        curr_corners = convert_bbox_to_corners_for_camera(boxes)
    else:
        raise NotImplementedError(f'Not implemented for box type {type(boxes)}, only for CameraInstance3DBoxes')
    velocities = (boxes.tensor[:,7:] if boxes.tensor.shape[1] > 7 else boxes.tensor.new_tensor(torch.zeros([boxes.tensor.shape[0],2]))).numpy()
    availabe_camera_types = list(info['images'].keys())
    assert camera_type in availabe_camera_types, f'camera type {camera_type} not in available camera types \n\t{availabe_camera_types}' 
    cam2ego = np.array(info['images'][camera_type]['cam2ego'])
    ego_corners = curr_corners @ cam2ego[:3,:3].T + cam2ego[:3,3]
    
    velocities3d = np.zeros((velocities.shape[0],3))
    velocities3d[:,[0,2]] = velocities
    ego_velocities = velocities3d @ cam2ego[:3,:3].T
    
    return ego_corners, ego_velocities[:,:2]

def ego_corners3d_to_cam_bbox(corners, velocities, info):
    cam2ego =np.array( info['images']['front_camera']['cam2ego'])
    ego2cam = np.linalg.inv(cam2ego)
    corners = np.array(corners)
    velocities = np.array(velocities)
    velocities3d = np.zeros((velocities.shape[0],3))
    velocities3d[:,[0,1]] = velocities
    velocities = (velocities3d @ ego2cam[:3,:3].T)[:,::2].tolist()
    corners = corners @ ego2cam[:3,:3].T + ego2cam[:3,3]
    boxes = [convert_corners_to_bbox_for_cam_box(corner) for corner in corners]
    bboxes = [bbox+velocities[i] for i,bbox in enumerate(boxes)]
    return CameraInstance3DBoxes(tensor=bboxes, box_dim=9, origin=(0.5,0.5,0.5))

def ego_corners3d_to_ego_bbox(corners, velocities,):
    # corners = corners @ ego2cam[:3,:3].T + ego2cam[:3,3]
    boxes = [convert_corners_to_bbox_for_lidar_box(corner) for corner in corners]
    bboxes = [bbox+velocities[i].tolist() for i,bbox in enumerate(boxes)]
    return LiDARInstance3DBoxes(tensor=torch.Tensor(bboxes), box_dim=9, origin=(0.5,0.5,0.5))

def filter_eval_boxes(boxes:dict[str,list], max_dist_func):
    for sample_token, instances in boxes.items():
        boxes[sample_token] = [instance for instance in instances if np.linalg.norm((instance['translation'][:2])) < max_dist_func(instance['detection_name'])]
    return boxes

def pandaset_evaluate_matrics(pred_boxes, gt_boxes, classes, dist_thrs, dist_thr_tp):
    from mmdet3d.evaluation.metrics import pandaset_metric_utils
    start_time = time.time()
    MEAN_AP_WEIGHT = 5
    MIN_PRECISION, MIN_RECALL = 0.1, 0.1
    metric_data_lists = {}
    TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
    mean_dist_aps = {}
    for name in classes:
        metric_data_list = metric_data_lists[name] = {}
        for dist_th in dist_thrs:
            metric_data_list[dist_th] = metric_data = pandaset_metric_utils.get_metrics(pred_boxes, gt_boxes, name, dist_th)
            # if len(metric_data) == 0:
            #     continue
            ap = pandaset_metric_utils.calc_ap(metric_data, MIN_RECALL, MIN_PRECISION)
            metric_data['ap'] = ap
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[dist_thr_tp]
            # if len(metric_data) == 0:
            #     continue
            metric_data[metric_name] = pandaset_metric_utils.calc_tp(metric_data, MIN_RECALL, metric_name)

        mean_dist_aps[name] ={'ap': np.mean(np.array([metric_data['ap'] for metric_data in metric_data_list.values() if metric_data]))}
    
    mean_ap = np.mean([ap_dict['ap'] for ap_dict in mean_dist_aps.values()])
    tp_errors = {}
    label_tp_errors = {}
    for metric_name in TP_METRICS:
        class_errors = []
        for detection_name in classes:
            metric_data = metric_data_lists[detection_name][dist_thr_tp]
            # if len(metric_data) ==0:
            #     continue
            class_errors.append(metric_data.get(metric_name, float('nan')))
            if detection_name not in label_tp_errors:
                label_tp_errors[detection_name] = {metric_name:metric_data.get(metric_name, float('nan'))}
            else:
                label_tp_errors[detection_name][metric_name] = metric_data.get(metric_name, float('nan'))

        tp_errors[metric_name] = float(np.nanmean(class_errors))
    
    tp_scores = {}
    for metric_name in TP_METRICS:
        # We convert the true positive errors to "scores" by 1-error.
        score = 1.0 - tp_errors.get(metric_name, float('nan'))

        # Some of the true positive errors are unbounded, so we bound the scores to min 0.
        score = max(0.0, score)

        tp_scores[metric_name] = score

    total = float(MEAN_AP_WEIGHT * mean_ap + np.sum(list(tp_scores.values())))

    # Normalize.
    nd_score = total / float(MEAN_AP_WEIGHT + len(tp_scores.keys()))
    print('mean AP: %.4f' % (mean_ap))
    err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
    for tp_name, tp_val in tp_errors.items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
    print('NDS: %.4f' % (nd_score))
    eval_time = time.time()-start_time
    print('Eval Time %.1fs' % (eval_time))
    print()
    print('Per-class results:')
    print('%-40s\t%-10s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'))
    class_aps = mean_dist_aps
    class_tps = label_tp_errors
    for class_name in class_aps.keys():
        class_ap = class_aps.get(class_name, {'ap':float('nan')})['ap']
        class_tp = class_tps.get(class_name, {name: float('nan') for name in err_name_mapping})
        print('%-40s\t%-6.6f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
            % (class_name, class_ap,
                class_tp['trans_err'],
                class_tp['scale_err'],
                class_tp['orient_err'],
                class_tp['vel_err'],
                class_tp['attr_err']))

    return dict(mean_ap=mean_ap, mean_dist_aps=mean_dist_aps, label_tp_errors=label_tp_errors, tp_errors=tp_errors, tp_scores=tp_scores, nd_score=nd_score, metric_data_lists=metric_data_lists, eval_time = eval_time)
