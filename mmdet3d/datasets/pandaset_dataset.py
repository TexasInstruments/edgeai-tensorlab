from .nuscenes_dataset import NuScenesDataset
from os import path as osp
import copy
from typing import Union, List
import torch
import numpy as np

from mmdet3d.registry import DATASETS
from mmengine.utils import is_abs
from mmengine.fileio import join_path
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes



CAMERA_NAMES = [
    'back_camera',
    'front_camera',
    'front_left_camera',
    'front_right_camera',
    'left_camera',
    'right_camera',
]

# BEV networks do not detect emergency vehicle light, age and rider status
ALL_ATTRIBUTES =  [
    'object_motion', 'pedestrian_behavior'
]

CLASSES = [
    'Car', 
    'Semi-truck', 
    'Other Vehicle - Construction Vehicle', 
    'Pedestrian with Object', 
    'Train', 
    'Animals - Bird', 
    'Bicycle', 
    'Rolling Containers', 
    'Pylons', 
    'Signs', 
    'Emergency Vehicle', 
    'Towed Object', 
    'Personal Mobility Device', 
    'Motorcycle', 
    'Tram / Subway', 
    'Other Vehicle - Uncommon', 
    'Other Vehicle - Pedicab', 
    'Temporary Construction Barriers', 
    'Animals - Other', 
    'Bus', 
    'Motorized Scooter', 
    'Pickup Truck', 
    'Road Barriers', 
    'Pedestrian', 
    'Construction Signs', 
    'Cones', 
    'Medium-sized Truck'
]


'''CLASSES = [
    'Car',
    'Pedestrian with Object',
    'Rolling Containers',
    'Pylons',
    'Signs',
    'Temporary Construction Barriers',
    'Pickup Truck',
    'Pedestrian',
    'Cones',
    'Medium-sized Truck'
]
'''
get_original_label = lambda x: (CLASSES.index(x) if x in CLASSES else -1)


UNIQUE_ATTRIBUTE_LABELS = [
    'pedestrian.Lying',
    'pedestrian.Sitting',
    'pedestrian.Standing',
    'pedestrian.Walking',
    'None',
    'emergency_vehicle.Moving.Lights not Flashing',
    'emergency_vehicle.Parked.Lights Flashing',
    'emergency_vehicle.Parked.Lights not Flashing',
    'emergency_vehicle.Stopped.Lights not Flashing',
    'vehicle.Moving',
    'vehicle.Moving.With Rider',
    'vehicle.Parked',
    'vehicle.Parked.With Rider',
    'vehicle.Parked.Without Rider',
    'vehicle.Stopped',
    'vehicle.Stopped.With Rider',
    'vehicle.With Rider',
    'vehicle.Without Rider',
]


def get_attribute_labels(cls_label, attributes):
    assert all(attr in ALL_ATTRIBUTES for attr in attributes)
    label = ''
    result = None
    if cls_label in ('Car', 'Pickup Truck', 'Medium-sized Truck', 'Semi-truck',     
                    'Towed Object', 'Other Vehicle - Construction Vehicle',
                    'Other Vehicle - Uncommon', 'Other Vehicle - Pedicab',
                    'Bus', 'Train', 'Trolley', 'Tram / Subway',):
        if 'object_motion' in attributes:
            if attributes['object_motion']:
                label += f'.{attributes["object_motion"]}'
        result = f'vehicle{label}' if label != '' else None
    
    if cls_label == 'Emergency Vehicle':
        if 'object_motion' in attributes:
            if attributes['object_motion']:
                label += f'.{attributes["object_motion"]}'
        if 'emergency_vehicle_lights' in attributes:
            if attributes['emergency_vehicle_lights']:
                label += f'.{attributes["emergency_vehicle_lights"]}'
        result = f'emergency_vehicle{label}' if label != '' else None
            
    if cls_label in ('Pedestrian', 'Pedestrian with Object'):
        label = 'pedestrian'
        if 'pedestrian_behavior' in attributes:
            if attributes['pedestrian_behavior']:
                label += f'.{attributes["pedestrian_behavior"]}'
        result = label if label != '' else None
    if cls_label in ('Motorcycle', 'Personal Mobility Device', 'Motorized Scooter',
                    'Bicycle', 'Animals - Other',):
        if 'object_motion' in attributes:
            if attributes['object_motion']:
                label += f'.{attributes["object_motion"]}'
        if 'rider_status' in attributes:
            if attributes['rider_status']:
                label += f'.{attributes["rider_status"]}'
        result: str | None = f'vehicle{label}' if label != '' else None
    return UNIQUE_ATTRIBUTE_LABELS.index('None') if (result is None or result not in UNIQUE_ATTRIBUTE_LABELS )else UNIQUE_ATTRIBUTE_LABELS.index(result)


@DATASETS.register_module()
class PandaSetDataset(NuScenesDataset):
    
    METAINFO = {
        'classes':CLASSES,
        'version': 'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
            (255, 255, 0), # yellow
            (0, 255, 255), # cyan
            (255, 0, 255), # magenta       
            (80, 200, 120), # emerald green
            (64, 204, 228), # Torquoise
            (34, 139, 34), # forest green
            (0, 128, 128), # teal
            (250, 128, 114), # salmon
            (255, 20, 147), # deep pink
            (128, 0, 0), # maroon
            (0, 128, 0), # green
            (0, 0, 128), # navy
            (128, 128, 0), # olive
            (128, 0, 128), # purple
            (128, 128, 128), # grey
            (192, 192, 192), # silver
            (184, 115, 51), # copper
        ]
    }
    
    def __init__(self, 
                 data_root,
                 ann_file,
                 pipeline = ...,
                 box_type_3d = 'LiDAR',
                 load_type = 'frame_based',
                 modality = ...,
                 filter_empty_gt = True,
                 test_mode = False,
                 with_velocity = True,
                 use_valid_flag = False,
                 max_dist_thr = None,
                 **kwargs):
        if 'metainfo' in kwargs and (orig_class_mapping := kwargs['metainfo'].get('class_mapping', None)) is not None:
            if not isinstance(orig_class_mapping,(dict, list, tuple)):
                print(f"Wrong class mapping of type {type(orig_class_mapping).__name__} provided! No mapping is used!")
                print("Please provide a mapping of type dict, list or tuple")
                orig_class_mapping = None
        else:
            orig_class_mapping = None
        if orig_class_mapping :
            if len(orig_class_mapping) != len(self.METAINFO['classes']):
                print(f"Wrong class mapping of Length {len(orig_class_mapping)} provided! No mapping is used!")
                print("Please provide a mapping of length", len(self.METAINFO['classes']))
                orig_class_mapping = None
            else:
                if isinstance(orig_class_mapping, dict):
                    temp = [0]*len(orig_class_mapping)
                    for k,v in orig_class_mapping.items():
                        if isinstance(k, str):
                            if k not in self.METAINFO['classes']:
                                print(f"Wrong key {k} provided for class mapping! No mapping is used!")
                                print(f"Please provide a key avilable in \n\t{self.METAINFO['classes']}\n")
                                temp = None
                                break
                            temp[self.METAINFO['classes'].index(k)] = v
                        elif isinstance(k, int):
                            if (k<0) or (k>=len(self.METAINFO['classes'])):
                                print(f"Wrong key {k} provided for class mapping! No mapping is used!")
                                print(f"Please provide a key avilable between 0 and {len(self.METAINFO['classes'])-1}\n")
                                temp = None
                                break
                            temp[k]=v
                        else:
                            print(f"Wrong key {k} of type {type(k).__name__} provided for class mapping! No mapping is used!")
                            print("Please provide a key of type int or string")
                            temp = None
                            break
                    orig_class_mapping = temp
                temp = [0]*len(orig_class_mapping)
                for i, k in enumerate(orig_class_mapping):
                    if isinstance(k, str):
                        if k not in kwargs['metainfo']['classes']:
                            print(f"Wrong key {k} provided for class mapping! No mapping is used!")
                            print(f"Please provide a key avilable in \n\t{kwargs['metainfo']['classes']}\n")
                            temp = None
                            break
                        temp[i] = kwargs['metainfo']['classes'].index(k)
                    elif isinstance(k, int):
                        if (k<0) or (k>=len(kwargs['metainfo']['classes'])):
                            print(f"Wrong key {k} provided for class mapping! No mapping is used!")
                            print(f"Please provide a key avilable between 0 and {len(kwargs['metainfo']['classes'])-1}\n")
                            temp = None
                            break
                        temp[i]=k
                    else:
                        print(f"Wrong key {k} of type {type(k).__name__} provided for class mapping! No mapping is used!")
                        print("Please provide a key of type int or string")
                        temp = None
                        break
                    orig_class_mapping = temp
                orig_class_mapping = [max(min(x,len(kwargs['metainfo']['classes'])),0) for x in orig_class_mapping ]
            kwargs['metainfo']['class_mapping'] = orig_class_mapping
        self._orig_data_prefix = copy.deepcopy(kwargs.get('data_prefix',{}))
        self.get_label_func = (lambda x : orig_class_mapping[x]) if orig_class_mapping else (lambda x: x)
        self.max_dist_thr = max_dist_thr
        self.label_mapping_changed = False
        self.new_num_ins_per_cat = [0]*len(kwargs['metainfo']['classes'] if 'metainfo' in kwargs and 'classes' in kwargs['metainfo'] else self.METAINFO['classes'])
        super().__init__(data_root, ann_file, pipeline, box_type_3d, load_type, modality, filter_empty_gt, test_mode, with_velocity, use_valid_flag, **kwargs)

    def full_init(self):
        if not self.label_mapping_changed:
            for k in self.label_mapping:
                self.label_mapping[k] = self.get_label_func(k)
            self.label_mapping_changed = True
        result = super().full_init()
        self.num_ins_per_cat = self.new_num_ins_per_cat
        return result

    def _filter_with_mask(self, ann_info):
        if self.max_dist_thr:
            filtered_ann_info = {}
            gt_bboxes_3d = ann_info['gt_bboxes_3d']
            labels = ann_info['gt_labels_3d']
            if isinstance(self.max_dist_thr,(int, float)):
                max_dist_thr = self.max_dist_thr
            elif isinstance(self.max_dist_thr, (list, tuple)):
                max_dist_thr = np.array([self.max_dist_thr[i] for i in labels])
            elif isinstance(self.max_dist_thr, dict):
                max_dist_thr = [0]*len(self.metainfo['classes'])
                for key, val in self.max_dist_thr.items():
                    if isinstance(key, int):
                        max_dist_thr[key] = val
                    elif isinstance(key,str) and key in self.metainfo['classes']:
                        max_dist_thr[self.metainfo['classes'].index(key)] = val
                    else:
                        max_dist_thr = [50]*len(max_dist_thr)
                        break
                max_dist_thr = np.array([max_dist_thr[i] for i in labels])
            else:
                max_dist_thr = 50
            if self.load_type == 'mv_image_based':
                translations = gt_bboxes_3d[:,[0,2]]
            else:
                translations = gt_bboxes_3d[:,:2]
            filtered_indices = np.where(np.linalg.norm(np.array(translations),axis=-1)<max_dist_thr)[0].tolist()
            for key, value in ann_info.items():
                if isinstance(value,np.ndarray):
                    value = value[filtered_indices]
                elif isinstance(value, (list,tuple)):
                    value = [v for i,v in enumerate(value) if i in filtered_indices]
                filtered_ann_info[key] = value
            ann_info = filtered_ann_info
        return super()._filter_with_mask(ann_info)
    
    def parse_ann_info(self, info):
        instances = info['instances']
        for instance in instances:
            instance['bbox_label'] = self.get_label_func(instance['bbox_label'])
            instance['bbox_label_3d'] = self.get_label_func(instance['bbox_label_3d'])
            instance['velocity'] = instance['velocity'] [::2] if self.load_type == 'mv_image_based' else instance['velocity'][:2]
        cam_instances = info.get('cam_instances',{})
        for name, instances in cam_instances.items():
            for instance in instances:
                instance['bbox_label'] = self.get_label_func(instance['bbox_label'])
                instance['bbox_label_3d'] = self.get_label_func(instance['bbox_label_3d'])
                instance['velocity'] = instance['velocity'] [::2]
        ann_info =  super().parse_ann_info(info)
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.new_num_ins_per_cat[label] += 1
        return ann_info
    
    def _join_prefix(self, scene_id=None):
        if scene_id is None:
            if self.ann_file and not is_abs(self.ann_file) and self.data_root and (self.data_root not in self.ann_file):
                self.ann_file = join_path(self.data_root, self.ann_file)
            return
        self.data_prefix = copy.deepcopy(self._orig_data_prefix)
        for key, prefix in self.data_prefix.items():
            self.data_prefix[key] = osp.join(scene_id, prefix)
        return super()._join_prefix()

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        scene_id = info.get('scene_token', '')
        self._join_prefix(scene_id)
        return super().parse_data_info(info)
