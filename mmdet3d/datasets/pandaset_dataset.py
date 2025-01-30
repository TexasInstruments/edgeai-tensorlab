from .nuscenes_dataset import NuScenesDataset
from os import path as osp
import copy
from typing import Union, List

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

ALL_ATTRIBUTES =  [
    'object_motion', 'pedestrian_behavior', 'pedestrian_age', 'rider_status', 'emergency_vehicle_lights'
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
get_original_label = lambda x: (CLASSES.index(x) if x in CLASSES else -1)


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
        if 'pedestrian_age' in attributes:
            if attributes['pedestrian_age']:
                label += f'{attributes["pedestrian_age"]}'
        if label == '':
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
        result = f'vehicle{label}' if label != '' else None
    return -1 if (result is None or result not in UNIQUE_ATTRIBUTE_LABELS )else UNIQUE_ATTRIBUTE_LABELS.index(result)


UNIQUE_ATTRIBUTE_LABELS = [
    'Adult.Lying',
    'Adult.Sitting',
    'Adult.Standing',
    'Adult.Walking',
    'Child.Sitting',
    'Child.Standing',
    'Child.Walking',
    'Child.Lying'
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
    
    def __init__(self, data_root, ann_file, pipeline = ..., box_type_3d = 'LiDAR', load_type = 'frame_based', modality = ..., filter_empty_gt = True, test_mode = False, with_velocity = True, use_valid_flag = False, **kwargs):
        if 'metainfo' in kwargs:
            orig_class_mapping = kwargs['metainfo'].get('class_mapping', None)
        else:
            orig_class_mapping = None
        self._orig_data_prefix = copy.deepcopy(kwargs.get('data_prefix',{}))
        self.get_label_func = (lambda x : orig_class_mapping[x]) if orig_class_mapping else (lambda x: x)
        super().__init__(data_root, ann_file, pipeline, box_type_3d, load_type, modality, filter_empty_gt, test_mode, with_velocity, use_valid_flag, **kwargs)
        
    def filter_data(self):
        return super().filter_data()
    
    def parse_ann_info(self, info):
        instances = info['instances']
        for instance in instances:
            instance['bbox_label'] = self.get_label_func(instance['bbox_label'])
            instance['bbox_label_3d'] = self.get_label_func(instance['bbox_label_3d'])
            instance['velocities'] = instance['velocities'] [::2] if self.load_type == 'mv_image_based' else instance['velocities'][:2]
        cam_instances = info.get('cam_instances',{})
        for name, instances in cam_instances.items():
            for instance in instances:
                instance['bbox_label'] = self.get_label_func(instance['bbox_label'])
                instance['bbox_label_3d'] = self.get_label_func(instance['bbox_label_3d'])
                instance['velocities'] = instance['velocities'] [::2]
        ann_info =  super().parse_ann_info(info)
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
