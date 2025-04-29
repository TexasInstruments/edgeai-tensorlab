# Copyright (c) Phigent Robotics. All rights reserved.

import pickle
from nuscenes import NuScenes
import pandaset as ps
import numpy as np
from pyquaternion import Quaternion
import argparse
import os

from mmdet3d.datasets.utils import convert_quaternion_to_matrix

def add_adj_info_nuscenes(root_path, version):
    interval = 3
    max_adj = 60

    #camera_types = [
    #    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    #    'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT',
    #]
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT',
    ]

    #sample_num = None
    # this script is needed only for training
    for set in ['train']:
        if version == 'v1.0-mini':
            nuscenes_version = version
            dataset = pickle.load(open('%s/nuscenes_mini_infos_%s.pkl' % (root_path, set), 'rb'))
            if set == 'test':
                continue
        else:
            dataset = pickle.load(open('%s/nuscenes_infos_%s.pkl' % (root_path, set), 'rb'))
            if set in ['train', 'val']:
                nuscenes_version = 'v1.0-trainval'
            else:
                nuscenes_version = 'v1.0-test'
        dataroot = root_path
        nuscenes = NuScenes(nuscenes_version, dataroot)

        map_token_to_id = dict()
        for id in range(len(dataset['data_list'])):
            map_token_to_id[dataset['data_list'][id]['token']] = id
            #if sample_num is not None and id > sample_num:
            #    break

        for id in range(len(dataset['data_list'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['data_list'])))
            #if sample_num is not None and id > sample_num:
            #    break
            info = dataset['data_list'][id]
            sample = nuscenes.get('sample', info['token'])
            for adj in ['next', 'prev']:
                sweeps = []
                adj_list = dict()
                for cam in camera_types:
                    adj_list[cam] = []
                    sample_data = nuscenes.get('sample_data', sample['data'][cam])

                    count = 0
                    while count < max_adj:
                        if sample_data[adj] == '':
                            break

                        sd_adj = nuscenes.get('sample_data', sample_data[adj])
                        sample_data = sd_adj

                        # add only key frames
                        #if sd_adj['is_key_frame'] is False:
                        #    continue
                        # check if scene token are identical - add only frames with the scame scene token
                        adj_sample = nuscenes.get('sample', sd_adj['sample_token'])
                        if sample['scene_token'] != adj_sample['scene_token']:
                            break

                        # Use ego pos from LiDAR
                        #lidar_rec = nuscenes.get('sample_data', adj_sample['data']['LIDAR_TOP'])
                        adj_list[cam].append(dict(data_path=os.path.join(dataroot, sd_adj['filename']),
                                                  timestamp=sd_adj['timestamp'],
                                                  ego_pose_token=sd_adj['ego_pose_token']))
                                                  #ego_pose_token=lidar_rec['ego_pose_token']))
                        count += 1

                for count in range(interval - 1, min(max_adj, len(adj_list['CAM_FRONT'])), interval):
                    timestamp_front = adj_list['CAM_FRONT'][count]['timestamp']
                    # get ego pose
                    pose_record = nuscenes.get('ego_pose', adj_list['CAM_FRONT'][count]['ego_pose_token'])

                    # get cam infos (use timestamp to find cameras sampled at the same time with front cam)
                    cam_infos = dict(CAM_FRONT=dict(data_path=adj_list['CAM_FRONT'][count]['data_path']))
                    for cam in camera_types:
                        timestamp_curr_list = np.array([t['timestamp'] for t in adj_list[cam]], dtype=np.int64)
                        diff = np.abs(timestamp_curr_list - timestamp_front)
                        selected_idx = np.argmin(diff)
                        cam_infos[cam] = dict(data_path=adj_list[cam][int(selected_idx)]['data_path'])

                    sweeps.append(dict(timestamp=timestamp_front, cams=cam_infos,
                                       ego2global=convert_quaternion_to_matrix(
                                                  pose_record['rotation'],
                                                  pose_record['translation'])))

                dataset['data_list'][id][adj] = sweeps if len(sweeps) > 0 else None

            # get ego speed and transfrom the targets velocity from global frame into ego-relative mode
            previous_id = id
            if not sample['prev'] == '':
                sample_tmp = nuscenes.get('sample', sample['prev'])
                previous_id = map_token_to_id[sample_tmp['token']]
            next_id = id
            if not sample['next'] == '':
                sample_tmp = nuscenes.get('sample', sample['next'])
                next_id = map_token_to_id[sample_tmp['token']]
            time_pre = dataset['data_list'][previous_id]['timestamp']
            time_next = dataset['data_list'][next_id]['timestamp']
            time_diff = time_next - time_pre
            posi_pre = np.array(dataset['data_list'][previous_id]['ego2global'], dtype=np.float32)[0:3, 3]
            posi_next = np.array(dataset['data_list'][next_id]['ego2global'], dtype=np.float32)[0:3, 3]
            velocity_global = (posi_next - posi_pre) / time_diff

            l2e_r_mat = np.array(info['lidar_points']['lidar2ego'])[0:3, 0:3]
            e2g_r_mat = np.array(info['ego2global'])[0:3, 0:3]

            velocity_global = np.array([*velocity_global[:2], 0.0])
            #velocity_lidar = velocity_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
            #    l2e_r_mat).T
            velocity_lidar = velocity_global @ e2g_r_mat @ l2e_r_mat
            velocity_lidar = velocity_lidar[:2]

            dataset['data_list'][id]['velo'] = velocity_lidar
            if set in ['train', 'val']:
                #gt_velocity = np.array([instance['velocity'] for instance in dataset['data_list'][id]['instances']])
                #if gt_velocity.size == 0:
                #    dataset['data_list'][id]['gt_velocity'] = gt_velocity
                #else:
                #    dataset['data_list'][id]['gt_velocity'] = gt_velocity - velocity_lidar.reshape(1, 2)
                for instance in dataset['data_list'][id]['instances']:
                    instance['velocity'] = instance['velocity'] - velocity_lidar.reshape(2)

        #filename = './data/nuscenes/nuscenes_infos_%s_4d_interval%d_max%d.pkl' % (set, interval, max_adj)
        if version == 'v1.0-mini':
            filename = './data/nuscenes/nuscenes_mini_infos_%s_fastbev.pkl' % (set)
        else:
            filename = './data/nuscenes/nuscenes_infos_%s_fastbev.pkl' % (set)

        with open(filename, 'wb') as fid:
            pickle.dump(dataset, fid)


def add_adj_info_pandaset(root_path, version):
    interval = 1
    max_adj = 4

    camera_types = [
        'back_camera',
        'front_camera',
        'front_left_camera',
        'front_right_camera',
        'left_camera',
        'right_camera',
    ]
 
    # this script is needed only for training
    for set in ['train']:
        dataset = pickle.load(open('%s/pandaset_infos_%s.pkl' % (root_path, set), 'rb'))

        #dataroot = root_path
        #pandaset = ps.DataSet(dataroot)
        #scenes = pandaset.sequences()

        map_token_to_id = dict()
        for id in range(len(dataset['data_list'])):
            map_token_to_id[dataset['data_list'][id]['token']] = id

        for id in range(len(dataset['data_list'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['data_list'])))

            cur_info = dataset['data_list'][id]
            scene_token = cur_info['scene_token']
            for adj in ['next', 'prev']:
                adj_id = id

                count = 0
                sweeps = []
                adj_list = dict()
                while count < max_adj:
                    # init adj_list[cam]
                    for cam in camera_types:
                        adj_list[cam] = []

                    info = dataset['data_list'][adj_id]
                    if info[adj] is None:
                        break

                    if adj == 'next':
                        adj_id += 1
                    else:
                        adj_id -= 1

                    #print("adj: {}, daj_id:{},len:{}, adj:{}".format(adj, adj_id, len(dataset['data_list']), info[adj]))
                    adj_info = dataset['data_list'][adj_id]
                    if adj_info['scene_token'] != scene_token:
                        break

                    for cam in camera_types:
                        data_path = '%s/' % root_path +adj_info['scene_token']+'/camera/%s/%s' % \
                            (cam, adj_info['images'][cam]['img_path'])
                        #adj_list[cam].append(dict(data_path=data_path))
                        adj_list[cam] = dict(data_path=data_path)

                    sweeps.append(dict(timestamp=np.int64(adj_info['timestamp']*1e+6), cams=adj_list,
                                       ego2global=adj_info['ego2global']))
                    count += 1

                dataset['data_list'][id][adj+'_samples'] = sweeps if len(sweeps) > 0 else None

            # get ego speed and transfrom the targets velocity from global frame into ego-relative mode
            previous_id = id
            if cur_info['prev'] is not None:
                #previous_id = id - 1
                previous_id = map_token_to_id[cur_info['prev']]
            next_id = id
            if cur_info['next'] is not None:
                #next_id = id + 1
                next_id = map_token_to_id[cur_info['next']]
            time_pre = dataset['data_list'][previous_id]['timestamp']
            time_next = dataset['data_list'][next_id]['timestamp']
            time_diff = time_next - time_pre
            posi_pre = np.array(dataset['data_list'][previous_id]['ego2global'], dtype=np.float32)[0:3, 3]
            posi_next = np.array(dataset['data_list'][next_id]['ego2global'], dtype=np.float32)[0:3, 3]
            velocity_global = (posi_next - posi_pre) / time_diff

            l2e_r_mat = np.array(cur_info['lidar_points']['lidar2ego'])[0:3, 0:3]
            e2g_r_mat = np.array(cur_info['ego2global'])[0:3, 0:3]

            velocity_global = np.array([*velocity_global[:2], 0.0])
            velocity_lidar = velocity_global @ e2g_r_mat @ l2e_r_mat
            velocity_lidar = velocity_lidar[:2]

            dataset['data_list'][id]['velo'] = velocity_lidar
            for instance in dataset['data_list'][id]['instances']:
                instance['velocity'] = instance['velocity'][:2] - velocity_lidar.reshape(2)

        # prev_samples -> prev, next_samples -> next
        for id in range(len(dataset['data_list'])):
            dataset['data_list'][id]['prev'] = dataset['data_list'][id]['prev_samples']
            dataset['data_list'][id]['next'] = dataset['data_list'][id]['next_samples']
            del dataset['data_list'][id]['prev_samples']
            del dataset['data_list'][id]['next_samples']

        filename = '%s/pandaset_infos_%s_fastbev.pkl' % (root_path, set)
        with open(filename, 'wb') as fid:
            pickle.dump(dataset, fid)


parser = argparse.ArgumentParser(description='fasetbev sweep generator arg parser')
parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0-trainval',
    required=False,
    help='specify the dataset version for NuScenes')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes':
        add_adj_info_nuscenes(args.root_path, args.version)
    elif args.dataset == 'pandaset':
        add_adj_info_pandaset(args.root_path, args.version)


