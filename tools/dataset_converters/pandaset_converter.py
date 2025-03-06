
import pandaset as ps
import os
import pandas as pd
import tqdm
import random
import numpy as np
import mmengine
import argparse
import copy

from mmdet3d.datasets.pandaset_dataset import CLASSES, ALL_ATTRIBUTES, CAMERA_NAMES, get_attribute_labels, get_original_label
from mmdet3d.datasets.convert_utils import convert_corners_to_bbox_for_lidar_box, convert_corners_to_bbox_for_cam_box, convert_bbox_to_corners_for_lidar, \
                                           convert_bbox_to_lidar

test_scenes_const  = ['014', '101', '069', '091', '120', '011', '115', '059', '117', '068', '086',
                      '112', '019', '013', '052', '039', '113', '044', '079', '024', '099']
train_scenes_const = ['041', '064', '067', '012', '073', '093', '106', '004', '057', '090', '051',
                      '063', '021', '035', '008', '119', '056', '102', '095', '065', '066', '005',
                      '017', '109', '023', '105', '043', '047', '034', '003', '104', '042', '033',
                      '058', '006', '103', '158', '139', '037', '050', '029', '110', '018', '098',
                      '016', '084', '100', '122', '048', '001', '040', '070', '046', '149', '085',
                      '074', '002', '078', '097', '062', '054', '116', '077', '124', '053', '027',
                      '071', '015', '045', '080', '072', '030', '123', '094', '092', '089', '028',
                      '055', '088', '038', '020', '032']
    
def delete_obj(obj):
    if isinstance(obj, (list, tuple)):
        for o in obj:
            delete_obj(o)
    
    elif isinstance(obj, dict):
        for o in obj.values():
            delete_obj(o)
    
    del obj

def intrinsic_to_mat(instrinsic:ps.sensors.Intrinsics):
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = instrinsic.fx
    K[1, 1] = instrinsic.fy
    K[0, 2] = instrinsic.cx
    K[1, 2] = instrinsic.cy
    return K


def get_bbox_2d(bbox_corners_image, image_size,):
    w,h = image_size
    temp = []
    if isinstance(bbox_corners_image,np.ndarray):
        bbox_corners_image = bbox_corners_image.tolist()
    for a,b in (bbox_corners_image):
        if a>=0 and a<=w and b>=0 and b<=h:
            temp.append([a,b]) 
            continue
        if a < 0:
            a = 0
        elif a > w:
            a = w
        if b < 0:
            b = 0
        elif b > h:
            b = h
        temp.append([a,b])
    bbox_corners_image = np.array(temp)
    min_coords = np.min(bbox_corners_image, axis=0)
    max_coords = np.max(bbox_corners_image, axis=0)
    bbox_2d = [min_coords[0], min_coords[1], max_coords[0], max_coords[1]]  # [xmin, ymin, xmax, ymax]
    center = np.mean([bbox_2d[:2],bbox_2d[2:]], axis=-1).tolist()
    return bbox_2d, center

def compute_valid_flag_for_bboxes(cuboids, lidar_data):
    if isinstance(cuboids, pd.DataFrame):
        bboxes = cuboids.values.tolist()
    elif isinstance(cuboids, np.ndarray):
        bboxes  = cuboids.tolist()
    else:
        bboxes = cuboids
    lidar_points = lidar_data.to_numpy()[:,:3]

    valid_flags = []
    lidar_points_within_bboxes = []
    for token, label, yaw, stationary, camera_used, x, y, z, width, length, height, *others in bboxes:
        rot_mat_for_bbox = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw),  np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0,0,0,1]
        ])
        rot_mat_for_lidar =np.linalg.inv(rot_mat_for_bbox)
        lidar_points_from_center = (rot_mat_for_lidar @ np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))]).T).T[:,:3]
        condition = np.abs(lidar_points_from_center) <= np.array([[width/2, length/2, height/2]])
        points_within_center = np.all(condition, axis=-1)
        points_within_bboxes = lidar_points_from_center[points_within_center]
        points_within_center = np.any(points_within_center, axis=-1)
        valid_flags.append(points_within_center)
        lidar_points_within_bboxes.append(points_within_bboxes)
    
    return valid_flags, lidar_points_within_bboxes

def compute_camera_bboxes(cuboids, camera, frame_index):
    camera_pose = camera.poses[frame_index]
    cam_intrinsics = camera.intrinsics
    data = camera[frame_index]
    frame_idx = frame_index
    
    if isinstance(cuboids, pd.DataFrame):
        cuboids = cuboids.values.tolist()
    elif isinstance(cuboids, np.ndarray):
        cuboids = cuboids.tolist()
    filtered_cuboids = []
    for value in cuboids:
        bbox = value[5:11] + [value[2]]
        corners =convert_bbox_to_corners_for_lidar(bbox)
        corners = np.array(corners)
        projected_points2d, camera_points_3d, inner_indices = ps.projection(corners, data, camera_pose, cam_intrinsics,filter_outliers=False)
        # center = np.mean(camera_points_3d,axis=1)
        condition1 = camera_points_3d[ 2,:] > 0.0
        if np.all(condition1 == False):
            continue
        image_w, image_h = camera[frame_idx].size
        
        condition2 = np.logical_and(
            (projected_points2d[:, 1] < image_h) & (projected_points2d[:, 1] > 0),
            (projected_points2d[:, 0] < image_w) & (projected_points2d[:, 0] > 0))
        if np.all(condition2 == False):
            continue
        filtered_cuboids.append((value[0], projected_points2d, camera_points_3d))
    return filtered_cuboids

def calculate_velocities(all_cuboids, timestamps):
    all_velocities = []
    for index, cuboids in enumerate(all_cuboids):
        cuboids = cuboids.values.tolist()
        prev = (index-1) if index > 0 else None
        next = (index+1) if index < len(all_cuboids) - 1 else None
        velocities = []
        prev_cuboids= all_cuboids[prev].values.tolist() if prev is not None else None
        next_cuboids = all_cuboids[next].values.tolist() if next is not None else None
        prev_timestamp = timestamps[prev] if prev is not None else None
        next_timestamp = timestamps[next] if next is not None else None
        curr_timestamp = timestamps[index]
        for token, label, yaw, stationary, camera_used, x, y, z, width, length, height, *others in cuboids:
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

def filter_siblings(cuboids,sibling_idx):
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
    return non_removed_bboxes

def filter_unused_labels(cuboids):
    if isinstance(cuboids, pd.DataFrame):
        cuboids = cuboids.values
    elif isinstance(cuboids, (list, tuple)):
        cuboids = np.array(cuboids,dtype=object)

    valid_index =[]
    for i in range(cuboids.shape[0]):
        if cuboids[i, 1] in CLASSES:
            valid_index.append(i)

    cuboids=cuboids[valid_index]
    return pd.DataFrame(cuboids)



def create_frame_dict(seq, scene_id, frame_idx, all_velocities, cam2img ):
    scene_token = scene_id
    frame_token = f'{scene_id}_{frame_idx:02}'
    prev = (frame_idx-1) if frame_idx > 0 else None
    next = (frame_idx+1) if frame_idx < len(seq.lidar._data_structure) - 1 else None
    timestamp = seq.timestamps[frame_idx]
    gps = seq.gps[frame_idx]

    # lidar = ego 
    # all coordinates are in global based coordinate system
    lidar_file = seq.lidar._data_structure[frame_idx]
    lidar2global = (ps.geometry._heading_position_to_mat(**seq.lidar.poses[frame_idx]))
    lidar_timestamp = seq.lidar.timestamps[frame_idx]
    lidar2lidar90 = np.array(
        [[ 0,-1, 0, 0],
         [ 1, 0, 0, 0],
         [ 0, 0, 1, 0],
         [ 0, 0, 0, 1]]
    )
    lidar902ego = np.eye(4) # lidar90 = ego 
    
    # Current frame Lidar
    
    #                 up z    x left 
    #                 ^   ^
    #                 |  /
    #                 | /
    #  back y <------ 0
    # required lidar orientation(lidar90)

    #                             up z    x front (yaw=0)
    #                             ^   ^
    #                             |  /
    #                             | /
    # (yaw=0.5*pi) left y <------ 0
    
    #lidar2ego = lidar902ego @ lidar2lidar90
    lidar2ego = np.eye(4)
    ego2global = lidar2global @ np.linalg.inv(lidar2ego)
    
    cam2global = { name : (ps.geometry._heading_position_to_mat(**camera.poses[frame_idx])) for name, camera  in seq.camera.items()}
    image_files = { name : camera._data_structure[frame_idx].split(os.path.sep)[-1] for name, camera in seq.camera.items()}
    cam2lidar = {name: (np.linalg.inv(lidar2global) @ cam2global[name]) for name in CAMERA_NAMES}
    lidar2cam = {name: np.linalg.inv(cam2lidar[name]) for name in CAMERA_NAMES}
    cam2ego = {name: (lidar2ego @ cam2lidar[name]) for name in CAMERA_NAMES}
    cam_timestamp = {name: camera.timestamps[frame_idx] for name, camera in seq.camera.items()}
    cam_token = {name: f'{scene_id}_{i}_{frame_idx:02}' for i, name in enumerate(CAMERA_NAMES)}

    velocities = all_velocities[frame_idx]
    cuboids = seq.cuboids[frame_idx]

    sibling_column_index = list(cuboids.columns).index('cuboids.sibling_id')

    cuboids = filter_unused_labels(cuboids)

    filtered_cuboids = filter_siblings(cuboids,sibling_column_index)
    world_velocities = [velocities[i] for i in range(len(cuboids)) if cuboids.to_numpy()[i][0] in filtered_cuboids[:,0]]
    valid_flags, lidar_points_within_bboxes = compute_valid_flag_for_bboxes(filtered_cuboids, seq.lidar.data[frame_idx])
    num_lidar_points = [len(points) for i,points in enumerate(lidar_points_within_bboxes) ]
    # filtered_cuboids = filtered_cuboids[valid_flags]
    # world_velocities = [world_velocities[i] for i in range(len(valid_flags)) if valid_flags[i]]
    world_velocities = np.array(world_velocities)

    lidar_velocities = (np.linalg.inv(lidar2global[:3,:3]) @ world_velocities.T).T.tolist()
    cam_velocities = {name: dict(zip(filtered_cuboids[:,0].tolist(),(np.linalg.inv(cam2global[name][:3,:3]) @ world_velocities.T).T.tolist())) for name in CAMERA_NAMES}
    cuboids_data = {value[0]: value[1:] for value in filtered_cuboids}
    
    lidar_bboxes = []
    for cuboid in filtered_cuboids.tolist():
        bbox = cuboid[5:11] + [cuboid[2]]
        # print((cuboid),(bbox),sep='\n')
        corners = convert_bbox_to_corners_for_lidar(bbox)
        # we have to store in lidar90 reference not in lidar refrence, of current frame
        lidar_corners = ( np.linalg.inv(lidar2global) @ np.hstack([corners, np.ones((corners.shape[0], 1))]).T).T[:,:3]
        lidar_bboxes.append(convert_corners_to_bbox_for_lidar_box(lidar_corners))
        
    available_attrs = [attr for attr in ALL_ATTRIBUTES if f'attributes.{attr}' in cuboids.columns]
    columns_names = list(cuboids.columns)
    attribute_dicts = {}
    for value in cuboids.values.tolist():
        token = value[0]
        if token not in filtered_cuboids[:,0]:
            continue
        attributes = {}
        for attr in ALL_ATTRIBUTES:
            if attr in available_attrs:
                attr_val = value[columns_names.index(f'attributes.{attr}')]
                attributes[attr] = attr_val if str(attr_val) != 'nan' else None
            else:
                attributes[attr] = None
        attribute_dicts[token] = attributes
    
    instances = []
    for i, (token, label, yaw, stationary, camera_used, x, y, z, width, length, height, *others) in enumerate(filtered_cuboids):
        attr_label = get_attribute_labels(label, attribute_dicts[token])
        label = get_original_label(label)
        instances.append(dict(
            token=token,
            bbox_label=label,
            bbox_label_3d =label,
            bbox3d = lidar_bboxes[i],
            bbox_3d_isvalid = valid_flags[i],
            bbox_3d_isstationary = stationary,
            num_lidar_pts = num_lidar_points[i], # change to num_lidar_pts
            velocity = lidar_velocities[i],
            world_bbox3d = [x, y, z, width, length, height, yaw],
            world_velocity = world_velocities[i],
            attr_label=attr_label,
            ))
    
    camera_bboxes = {}
    for name, camera in seq.camera.items():
        camera_cuboids = compute_camera_bboxes(filtered_cuboids, camera, frame_idx)
        bboxes = []
        image_size = camera[frame_idx].size
        for token, projected_points2d, camera_points_3d in camera_cuboids:
            bbox_2d, center = get_bbox_2d(projected_points2d, image_size,)
            bbox3d = convert_corners_to_bbox_for_cam_box(camera_points_3d.T)
            bboxes.append((token, bbox_2d, center, bbox3d))
        camera_bboxes[name] = bboxes
    
    cam_instances:dict[str,list] = {}
    for name, bboxes in camera_bboxes.items():
        cam_instances[name] = []
        for token, bbox2d, center, bbox3d in bboxes:
            label= cuboids_data[token][0]
            attr_label = get_attribute_labels(label, attribute_dicts[token])
            label = get_original_label(label)
            cam_instances[name].append(dict(
                token=token,
                bbox_label=label,
                bbox_label_3d=label,
                bbox=bbox2d,
                bbox_3d=bbox3d,
                depth=bbox3d[2],
                bbox_3d_isvalid=True,
                center_2d=center,
                velocity=cam_velocities[name][token],
                attr_label=attr_label,
            ))
    
    frame_dict = dict(
        token=frame_token,
        timestamp=timestamp,
        ego2global=ego2global.tolist(),
        images={
            name: {
                'sample_data_token':cam_token[name],
                'img_path': image_files[name],
                'cam2img': cam2img[name].tolist(),
                'timestamp': cam_timestamp[name],
                'lidar2cam' : lidar2cam[name].tolist(),
                'cam2ego': cam2ego[name].tolist(),
                'cam2global': cam2global[name].tolist(),
            } for name in CAMERA_NAMES
        },
        lidar_points = dict(
            num_pts_feats=4,
            lidar_path=lidar_file.split(os.path.sep)[-1],
            lidar2ego=lidar2ego.tolist(),
            timestamp=lidar_timestamp,
            lidar2global=lidar2global.tolist(),
        ),
        frame_idx=frame_idx,
        prev= f'{scene_id}_{prev:02}' if prev is not None else None,
        next= f'{scene_id}_{next:02}' if next is not None else None,
        scene_token = scene_token,
        gps = gps,
        instances= instances,
        cam_instances=cam_instances,
    )
    return frame_dict


def create_datalist_per_scene(scene_id, dataset, ):
    data_list = []
    seq = dataset[scene_id]
    seq.load()
    cam2img = {}
    for name, camera in seq.camera.items():
        cam2img[name] = intrinsic_to_mat(camera.intrinsics)
    all_velocities = calculate_velocities(seq.cuboids.data,seq.timestamps,) # global coordinates
    print("Creating pickle data for scene:", scene_id)
    with tqdm.tqdm(total=len(seq.lidar._data_structure)) as pbar:

        for frame_idx in range(len(seq.lidar._data_structure)):
            frame_dict = create_frame_dict(seq, scene_id, frame_idx, all_velocities, cam2img)
            data_list.append(copy.deepcopy(frame_dict))
            delete_obj(frame_dict)
            pbar.update()
    
    for frame_idx in range(len(seq.lidar._data_structure)):
        del seq.cuboids.data[0]
        del seq.lidar.data[0]
        del seq.lidar.poses[0]
        for camera in seq.camera.values():
            del camera.data[0]
            del camera.poses[0]
        if seq.semseg:
            del seq.semseg.data[0]
    
    for var in (seq, cam2img, all_velocities):
        delete_obj(var)
    
    print(f'Created {len(seq.lidar._data_structure)} pickle data for scene {scene_id}')
    
    return data_list


def create_pickle_file( dataset, scenes, output_dir=None, info_prefix=None, version=None, dataset_name=None,train_split=True):
    output_dir = output_dir or os.getcwd()
    version = version or 'v1.0'
    dataset_name = dataset_name or 'PandaSetDataset'
    info_prefix = info_prefix or ''
    metainfo = dict(
        dataset= dataset_name,
        version=version,
        info_version='1.0',
        categories={name:get_original_label(name) for name in CLASSES}
    )

    prefix = info_prefix + '_' if len(info_prefix) else ''
    file_name = prefix + f'pandaset_infos_{"train" if train_split else "val"}.pkl'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name)
    data_list = []

    for scene_id in scenes:
        data_list.extend(create_datalist_per_scene(scene_id, dataset,))
    
    for i, frame in enumerate(data_list):
        frame['sample_idx'] = i

    pkl_data =  dict(metainfo=metainfo, data_list=data_list)
    mmengine.dump(pkl_data, output_file)
    print('Pickle data saved to', output_file)



def create_pickle_files(dataset_path, output_dir, info_prefix, version, dataset_name, with_semseg=False, fixed_split=True, train_split=0.80):
    dataset = ps.DataSet(dataset_path)
    semseg_scenes = dataset.sequences(with_semseg=with_semseg)
    scenes = dataset.sequences()

    has_semseg = [scene in semseg_scenes for scene in scenes]
    if with_semseg:
        scenes = semseg_scenes

    if fixed_split is True:
        train_scenes = train_scenes_const
        test_scenes  = test_scenes_const
    else:
        train_scenes = random.sample(scenes, k=int(len(scenes)*train_split))
        test_scenes  = [scene for scene in scenes if scene not in train_scenes]

    if with_semseg:
        pass
    else:
        create_pickle_file(dataset, test_scenes, output_dir, info_prefix, version, dataset_name, train_split=False)
        create_pickle_file(dataset, train_scenes,  output_dir, info_prefix, version, dataset_name,)

def create_pandaset_infos(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir):
    create_pickle_files(root_path, out_dir, info_prefix, version, dataset_name)
    

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--info-prefix', type=str, default='')
    parser.add_argument('--version', type=str, default='v1.0')
    parser.add_argument('--dataset-name', type=str, default='PandaSetDataset')
    parser.add_argument('--with-semseg',  action='store_true')
    parser.add_argument('--train-split', type=float, default=0.75)
    args = parser.parse_args() if args is None else parser.parse_args(args)
    create_pickle_files(args.dataset_path, args.output_dir, args.info_prefix, args.version, args.dataset_name, args.with_semseg, args.train_split)

if __name__ == '__main__':
    main(['--dataset-path', '/data/ssd/files/a0507161/edgeai/edgeai-mmdetection3d/data/pandaset/data/', '--output-dir', '/data/ssd/files/a0507161/edgeai/edgeai-mmdetection3d/data/pandaset/data/'])
