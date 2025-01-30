import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    

    
    def __init__(self, data_root, ann_file, metric = 'bbox', modality = ..., prefix = None, format_only = False, jsonfile_prefix = None, eval_version = 'detection_cvpr_2019', collect_device = 'cpu', backend_args = None):
        super().__init__(data_root, ann_file, metric, modality, prefix, format_only, jsonfile_prefix, eval_version, collect_device, backend_args)
        self.NameMapping = dict(map(lambda x:(x,x),self.dataset_meta['classes']))
    
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
                attrs = det['attr_labels'].to('cpu').numpy().tolist()
            else:
                attrs = None
            scores = det['scores_3d'].to('cpu').numpy().tolist()
            labels = det['labels_3d'].to('cpu').numpy().tolist()
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]['token']
            for i in range(boxes.tensor.shape[0]):
                box = boxes.tensor[i]
                name = classes[labels[i]]
                if attrs:
                    attr = self.get_attr_name(attrs[i], name)
                elif np.sqrt(np.sum(np.square(box[7:9]))) > 0.2:
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
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
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
            attrs = det['attr_labels'].to('cpu').numpy().tolist()
            scores = det['scores_3d'].to('cpu').numpy().tolist()
            labels = det['labels_3d'].to('cpu').numpy().tolist()
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
            scores= torch.Tensor(scores_per_frame).to('cuda')
            nms_scores = scores.new_zeros(scores.shape[0],NUM_CLASSES+1)
            labels = torch.LongTensor(labels_per_frame).to('cuda')
            indices = labels.new_tensor(list(range(scores.shape[0])))
            nms_scores[indices,labels] = scores
            scores = nms_scores
            cam_boxes3d = ego_corners3d_to_cam_bbox(ego_corners, velocities, info)
            
            # box nms 3d over 6 images in a frame
            # TODO: move this global setting into config
            nms_cfg = dict(
                use_rotate_nms=True,
                nms_across_levels=False,
                nms_pre=4096,
                nms_thr=0.05,
                score_thr=0.01,
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
            attrs = det['attr_labels'].to('cpu').tolist()
            scores = det['scores_3d'].to('cpu').tolist()
            labels = det['labels_3d'].to('cpu').tolist()
            ego_corners , velocities = cam_bbox_to_ego_corners3d(boxes)
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
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print(f'Results writes to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path
    
    def compute_metrics(self, results):
        self.bbox_type_3d = type(results[0]['bboxes_3d'])
        return super().compute_metrics(results)
    
    def _evaluate_single(self, result_path, classes = None, result_name = 'pred_instances_3d'):
        output_dir = osp.join(*osp.split(result_path)[:-1])
        detail = dict()
        
        metrics = self.pandaset_evaluate(result_path)
        metric_prefix = f'{result_name}_NuScenes'
        for name in classes:
            for k, v in metrics['label_aps'][name].items():
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
        results = data['results']
        meta = data['meta']
        # load annotation json file
        self.data_infos
        # load classes
        classes =self.dataset_meta['classes']
        class_mapping = self.dataset_meta.get('class_mapping', list(range(len(self.dataset_meta['classes']))))
        
        pred_boxes = {}
        for sample_token, boxes in results.items():
            if sample_token not in pred_boxes:
                pred_boxes[sample_token] = boxes.copy()
            else:
                pred_boxes[sample_token].extend(boxes)
        
        gt_boxes = {}
        if self.bbox_type_3d == LiDARInstance3DBoxes:
            for info in self.data_infos:
                sample_token = info['token']
                if sample_token in gt_boxes:
                    gt_boxes[sample_token].extend(info['instances'])
                else:
                    gt_boxes[sample_token] = info['instances'].copy()
        elif self.bbox_type_3d == CameraInstance3DBoxes:
            for info in self.data_infos:
                sample_token = info['token']
                for name, instacnes in info['cam_instances'].items():
                    if sample_token in gt_boxes:
                        gt_boxes[sample_token].extend(instacnes)
                    else:
                        gt_boxes[sample_token] = instacnes.copy()
        
        
        metrics = dict()
        return metrics


def cam_bbox_to_ego_corners3d(boxes, info, camera_type):
    if isinstance(boxes, CameraInstance3DBoxes):
        curr_corners = convert_bbox_to_corners_for_camera(boxes)
    else:
        raise NotImplementedError(f'Not implemented for box type {type(boxes)} only for CameraInstance3DBoxes and LiDARInstance3DBoxes')
    velocities = (boxes.tensor[:,-2:] if boxes.tensor.shape[1] > 7 else boxes.tensor.new_tensor(torch.zeros([boxes.tensor.shape[0],2]))).numpy()
    availabe_camera_types = list(info['images'].keys())
    assert camera_type not in availabe_camera_types, f'camera type {camera_type} not in available camera types \n\t{availabe_camera_types}' 
    cam2ego = np.array(info['images'][camera_type]['cam2ego'])
    ego_corners = curr_corners @ cam2ego[:3,:3].T + cam2ego[:3,3]
    
    velocities3d = np.zeros(velocities.shape[0],3)
    velocities3d[:,[0,2]] = velocities
    ego_velocities = velocities3d @ cam2ego[:3,:3].T
    
    return ego_corners, ego_velocities[:,:2]

def ego_corners3d_to_cam_bbox(corners, velocities, info):
    cam2ego =np.array( info['images']['front_camera']['cam2ego'])
    ego2cam = np.linalg.inv(cam2ego)
    corners = np.array(corners)
    velocities = np.array(velocities)
    velocities3d = np.zeros(velocities.shape[0],3)
    velocities3d[:,[0,1]] = velocities
    velocities = (velocities3d @ ego2cam[:3,:3].T)[:,::2].tolist()
    corners = corners @ ego2cam[:3,:3].T + ego2cam[:3,3]
    boxes = [convert_corners_to_bbox_for_cam_box(corner) for corner in corners]
    bboxes = [bbox+velocities[i] for i,bbox in enumerate(boxes)]
    return CameraInstance3DBoxes(tensor=bboxes, box_dim=9, origin=(0.5,0.5,0.5))

def ego_corners3d_to_ego_bbox(corners, velocities,):
    # corners = corners @ ego2cam[:3,:3].T + ego2cam[:3,3]
    boxes = [convert_corners_to_bbox_for_lidar_box(corner) for corner in corners]
    bboxes = [bbox+velocities[i] for i,bbox in enumerate(boxes)]
    return LiDARInstance3DBoxes(tensor=torch.Tensor(bboxes), box_dim=9, origin=(0.5,0.5,0.5))
