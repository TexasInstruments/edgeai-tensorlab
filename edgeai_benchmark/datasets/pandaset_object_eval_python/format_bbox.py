import pandaset as PS
import pandas as pd
import numpy as np


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

ALL_ATTRIBUTES =  [
    'object_motion', 'pedestrian_behavior'
]

ORIG_CLASSES = [
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
# get_original_label = lambda x: (ORIG_CLASSES.index(x) if x in CLASSES else -1)

## functions in this files are based edgeai-mmdetection3d

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


def filter_siblings(cuboids, velocities, sibling_idx):
    if isinstance(cuboids, pd.DataFrame):
        cuboids = cuboids.values
    elif isinstance(cuboids, (list, tuple)):
        cuboids = np.array(cuboids,dtype=object)
# cuboids =  seq.cuboids.data[0].to_numpy()
    tokens = cuboids[:,0]
    siblings = cuboids[:, sibling_idx]
    has_sibling = siblings != '-'
    # siblings = siblings[has_sibling]
    siblings_map = {}
    for token, sibling in zip(tokens[has_sibling], siblings[has_sibling]):
        if sibling in siblings_map:
            if token not in  siblings_map :
                siblings_map[sibling].add( token)
        elif not all(t in siblings_map for t in (token, siblings)):
            siblings_map[token] = {sibling}
    removable_siblings = []
    [removable_siblings.extend(s) for s in siblings_map.values()]
    non_removed_bboxes = cuboids[[token not in removable_siblings  for token in tokens]]
    non_removed_velocities = velocities[[token not in removable_siblings  for token in tokens]]
    return non_removed_bboxes, non_removed_velocities

def calculate_velocities(all_cuboids, timestamps):
    all_velocities = []
    for index, cuboids in enumerate(all_cuboids):
        cuboids = cuboids.values.tolist()
        prev = (index-1) if index > 0 else None
        next = (index+1) if index < len(all_cuboids.data) - 1 else None
        velocities = []
        prev_cuboids= all_cuboids[prev].values.tolist() if prev is not None else None
        next_cuboids = all_cuboids[next].values.tolist() if next is not None else None
        prev_timestamp = timestamps[prev] if prev is not None else None
        next_timestamp = timestamps[next] if next is not None else None
        curr_timestamp = timestamps[index]
        for token, label, yaw, stationary, camera_used, x, y, z, width, length, height, *others in cuboids:
            # For the objects with static attribute
            if others[0] != 'Moving' and (len(others) < 5 or others[4] != 'Walking'):
                velocities += [[0,0,0]]
                continue

            prev_cuboid = next_cuboid = None
            velocity = None
            if prev is not None:
                prev_cuboids1 = [value for value in prev_cuboids if value[0] == token]
                # print(prev, len(prev_cuboids1))
                if len(prev_cuboids1):
                    prev_cuboid = prev_cuboids1[0]
            if next is not None:
                next_cuboids1 = [value for value in next_cuboids if value[0] == token]
                # print(next, len(next_cuboids1))
                if len(next_cuboids1):
                    next_cuboid = next_cuboids1[0]
            
            if prev_cuboid is None and next_cuboid is None:
                velocities += [[0,0,0]]
                continue
            velocity = [[],[],[]]
            if prev_cuboid is not None:
                prev_x, prev_y, prev_z = prev_cuboid[5:8]
                prev_timestamp = prev_timestamp
                prev_time_diff = curr_timestamp - prev_timestamp
                prev_vel = (np.array([x,y,z]) - np.array([prev_x, prev_y, prev_z]))/prev_time_diff
                for i, vel in enumerate(velocity):
                    vel.append (prev_vel[i])
            if next_cuboid is not None:
                next_x, next_y, next_z = next_cuboid[5:8]
                next_timestamp = next_timestamp
                next_time_diff = next_timestamp - curr_timestamp
                next_vel = (np.array([next_x, next_y, next_z]) - np.array([x,y,z]))/next_time_diff
                for i, vel in enumerate(velocity):
                    vel.append (next_vel[i])
            
            velocities += [np.mean(velocity, axis=-1).tolist()]
        all_velocities.append(velocities)
    return all_velocities


def convert_ps_bbox_to_proper_bbox_for_globals(bbox):
    x, y, z, w, l, h, yaw = bbox
    new_yaw = yaw+np.pi/2
    if new_yaw > np.pi:
        new_yaw -= 2*np.pi
    if new_yaw < np.pi:
        new_yaw += 2*np.pi
    return [ x, y, z, l, w, h, new_yaw]


def get_gt_for_seq(seq:PS.dataset.Sequence):
    all_velocities = calculate_velocities(seq.cuboids, seq.timestamps)
    all_instances = []
    for frame_idx in range(len(seq.cuboids._data_structure)):
        cuboids = seq.cuboids[frame_idx]
        colns = list(cuboids.columns)
        cuboids = cuboids.values
        velocities = np.array(all_velocities[frame_idx])
        cuboids, velocities = filter_siblings(cuboids, velocities, colns.index('cuboids.sibling_id'))
        available_attrs = [attr for attr in ALL_ATTRIBUTES if f'attributes.{attr}' in colns]
        attribute_dicts = {}
        for value in cuboids.tolist():
            token = value[0]
            if token not in cuboids[:,0]:
                continue
            attributes = {}
            for attr in ALL_ATTRIBUTES:
                if attr in available_attrs:
                    attr_val = value[colns.index(f'attributes.{attr}')]
                    attributes[attr] = attr_val if str(attr_val) != 'nan' else None
                else:
                    attributes[attr] = None
            attribute_dicts[token] = attributes
        instances = []
        for i, (token, label, yaw, stationary, camera_used, x, y, z, width, length, height, *others) in enumerate(cuboids.tolist()):
            attr_label = get_attribute_labels(label, attribute_dicts[token])
            label = (label)
            instances.append(dict(
                token=token,
                bbox_label=label,
                # bbox_3d = lidar_bboxes[i],
                # bbox_3d_isvalid = valid_flags[token],
                # bbox_3d_isstationary = stationary,
                # num_lidar_pts = num_lidar_points[i], # change to num_lidar_pts
                # velocity = lidar_velocities[i],
                bbox3d = convert_ps_bbox_to_proper_bbox_for_globals([x, y, z, width, length, height, yaw]),
                velocity = velocities[i],
                attr_label=attr_label,
                ))
        all_instances.append(instances)
    return all_instances


def filter_eval_boxes(boxes, max_dist_func):
    for sample_token, instances in boxes.items():
        boxes[sample_token] = [instance for instance in instances if np.linalg.norm((instance['translation'][:2])) < max_dist_func(instance['detection_name'])]
    return boxes


def xywhr2xyxyr(
        boxes_xywhr: np.ndarray) -> np.ndarray:
    boxes = np.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]

    return boxes


def bbox3d2result(bboxes, scores, labels, attrs=None):
    result_dict = dict(
        bboxes_3d=bboxes,
        scores_3d=scores,
        labels_3d=labels)

    if attrs is not None:
        result_dict['attr_labels'] = attrs

    return result_dict


def rotation_3d_in_axis(
    points: np.ndarray,
    angles: np.ndarray,
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False
) ->tuple[np.ndarray, np.ndarray]|np.ndarray :
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray or Tensor): Points with shape (N, M, 3).
        angles (np.ndarray or Tensor or float): Vector of angles with shape
            (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it
            will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[Tensor, Tensor] or np.ndarray or
        Tensor: Rotated points with shape (N, M, 3) and rotation matrix with
        shape (N, 3, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = np.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and \
        points.shape[0] == angles.shape[0], 'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = np.stack([
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = np.stack([
                np.stack([rot_cos, rot_sin, zeros]),
                np.stack([-rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = np.stack([
                np.stack([ones, zeros, zeros]),
                np.stack([zeros, rot_cos, rot_sin]),
                np.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(
                f'axis should in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = np.stack([
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = np.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = np.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


def convert_bbox_to_cornrers3d(bboxes,yaw_axis):
    dims = bboxes[:, 3:6]
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin (0.5, 1, 0.5)
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = dims.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    corners = rotation_3d_in_axis(
        corners, bboxes[:, 6], axis=yaw_axis)
    corners += bboxes[:, :3].reshape(-1, 1, 3)
    return corners


def cam_bbox_to_global_corners3d(boxes, info, camera_type):
    box_dims= boxes[:, 3:6]
    box_yaw = boxes[:, 6:7]
    center = boxes[:, :3]
    velocity = np.zeros_like(center)
    velocity[:,[0,2]] = boxes[:,7:9] if boxes.shape[1]>7 else velocity[:,[0,2]]
    bottom_center = boxes[:, :3]
    gravity_center = np.zeros_like(bottom_center)
    gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]]
    gravity_center[:, 1] = bottom_center[:, 1] - boxes[:, 4] * 0.5
    corners = convert_bbox_to_cornrers3d(np.concatenate([gravity_center,box_dims,box_yaw], axis=1), yaw_axis=1)
    cam2global = np.array(info['images'][camera_type]['cam2global'])
    global_corners = corners @ cam2global[:3,:3].T + cam2global[:3,3]
    lobal_velocities = velocity @ cam2global[:3,:3].T
    return global_corners, lobal_velocities


def convert_corners_to_bbox_for_lidar_box(corners):
    corners = np.array(corners)
    x,y,z = np.mean(corners, axis=0)
    vector = corners[4] - corners[0]
    yaw = np.arctan2(vector[1], vector[0])
    corners = corners-np.array([[x,y,z]])
    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = (rot_matrix @ corners.T).T
    width = np.linalg.norm(corners[0] - corners[4])
    length = np.linalg.norm(corners[0] - corners[3])
    height = np.linalg.norm(corners[0] - corners[1])
    return [x, y, z, width, length, height, yaw]


def convert_corners_to_bbox_for_cam_box(corners):
    corners = np.array(corners)
    x,y,z = np.mean(corners, axis=0)
    vector = corners[4] - corners[0]
    yaw = -np.arctan2(vector[2], vector[0])
    corners = corners-np.array([[x,y,z]])
    corners = corners[:,[2,0,1]]
    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = (rot_matrix @ corners.T).T
    corners = corners[:,[1,2,0]]
    width = np.linalg.norm(corners[0] - corners[4])
    length = np.linalg.norm(corners[0] - corners[3])
    height = np.linalg.norm(corners[0] - corners[1])
    return [x, y, z, width, length, height, yaw]


def global_corners3d_to_cam_bbox(global_corners, velocities, info):
    cam2global =np.array( info['images'][list(info['images'].keys())[0]]['front_cam2global'])
    global2cam = np.linalg.inv(cam2global)
    corners = np.array(global_corners)
    velocities = np.array(velocities)
    velocities3d = np.zeros((velocities.shape[0],3))
    velocities3d = velocities
    velocities = (velocities3d @ global2cam[:3,:3].T)[:,::2].tolist()
    corners = corners @ global2cam[:3,:3].T + global2cam[:3,3]
    boxes = [convert_corners_to_bbox_for_cam_box(corner) for corner in corners]
    bboxes = [bbox+velocities[i] for i,bbox in enumerate(boxes)]
    return np.array(bboxes)


def front_cam_bbox_to_global_corners3d(boxes, info):
    box_dims= boxes[:, 3:6]
    box_yaw = boxes[:, 6:7]
    center = boxes[:, :3]
    velocity = np.zeros_like(center)
    velocity[:,[0,2]] = boxes[:,7:9]
    bottom_center = boxes[:, :3]
    gravity_center = np.zeros_like(bottom_center)
    gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]]
    gravity_center[:, 1] = bottom_center[:, 1] - boxes[:, 4] * 0.5
    corners = convert_bbox_to_cornrers3d(np.concatenate([gravity_center,box_dims,box_yaw], axis=1), yaw_axis=1)
    cam2global = np.array(info['images'][list(info['images'].keys())[0]]['front_cam2global'])
    global_corners = corners @ cam2global[:3,:3].T + cam2global[:3,3]
    lobal_velocities = velocity @ cam2global[:3,:3].T
    return global_corners, lobal_velocities


def global_corners3d_to_global_bbox(global_corners, velocities):
    boxes = [convert_corners_to_bbox_for_lidar_box(corner) for corner in global_corners]
    bboxes = [bbox+velocities[i].tolist() for i,bbox in enumerate(boxes)]
    return np.array(bboxes)


def convert_lidar_box_to_global_box(boxes, info, task_name):
    box_dims= boxes[:, 3:6]
    box_yaw = boxes[:, 6:7]
    center = boxes[:, :3]
    velocity = np.zeros_like(center)
    velocity[:,[0,2]] = boxes[:,7:9] if boxes.shape[1]>7 else velocity[:,[0,2]]
    bottom_center = boxes[:, :3]
    gravity_center = np.zeros_like(bottom_center)
    gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]]
    gravity_center[:, 1] = bottom_center[:, 1] - boxes[:, 4] * 0.5
    corners = convert_bbox_to_cornrers3d(np.concatenate([gravity_center,box_dims,box_yaw], axis=1), yaw_axis=2)
    # BEVDet predicted bboxes are in ego coordinate system
    if task_name == 'BEVDet':
        lidar2global = np.array(info['ego2global'])
    else:
        lidar2global = np.array(info['lidar2global'])
    global_corners = corners @ lidar2global[:3,:3].T + lidar2global[:3,3]
    lobal_velocities = velocity @ lidar2global[:3,:3].T
    return global_corners3d_to_global_bbox(global_corners, lobal_velocities)