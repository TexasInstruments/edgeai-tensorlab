# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
import os
import numpy as np
import cv2
from mmengine import print_log

from tools.dataset_converters import indoor_converter as indoor
from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters import lyft_converter as lyft_converter
from tools.dataset_converters import nuscenes_converter as nuscenes_converter
from tools.dataset_converters import semantickitti_converter
from tools.dataset_converters import pandaset_converter
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

export_2d_anno = False

def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False,
                    use_color=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    if use_color == True:
        #kitti_data_prep_point_painting(root_path)
        pts_prefix = "velodyne_painted"
        num_features = 8
        file_tail='.bin'
    else:
        pts_prefix = "velodyne"
        num_features = 4
        file_tail='.bin'

    kitti.create_kitti_info_file(root_path, info_prefix, with_plane, pts_prefix=pts_prefix, num_features=num_features,file_tail=file_tail)
    kitti.create_reduced_point_cloud(root_path, info_prefix, num_features=num_features)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_test_path)
    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'),
        use_color=use_color)

def load_kitti_calib_data(calib_dir_lines):
    # P2 * R0_rect * Tr_velo_to_cam * y
    #lines = open(calib_dir).readlines()
    lines = [line.split()[1:] for line in calib_dir_lines][:-1]
    CAM = 2
    #
    P = np.array(lines[CAM]).reshape(3, 4)
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

def kitti_data_prep_point_painting(root_path,
                    semantic_folder_name='Task0'):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    root_path = osp.join(root_path,"training")
    voldyne_reduced_path = osp.join(root_path,'velodyne')
    semantic_score_path = osp.join(root_path,semantic_folder_name)
    calib_path = osp.join(root_path,'calib')
    voldyne_reduced_painted_path = osp.join(root_path,'velodyne_painted')

    if (not osp.isdir(voldyne_reduced_path)) or (not osp.isdir(semantic_score_path)) or (not osp.isdir(calib_path)):
        print('Velodyne_reduced or semantic or calib folder is not available at {}'.format(root_path))
        print('Hence exiting kitti_data_prep_point_painting without doing anything')
    else:

        os.makedirs(voldyne_reduced_painted_path, exist_ok=True)

        directory = os.fsencode(voldyne_reduced_path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            file_base_name, ext = os.path.splitext(filename)
            min_x = 2
            min_y = 2
            if ext == ".bin":
                point_cloud_data = np.fromfile(osp.join(voldyne_reduced_path,filename),dtype=np.float32)
                point_cloud_data = np.reshape(point_cloud_data,(-1,4))
                reflection_data = np.expand_dims(point_cloud_data[:,3].copy(),1)
                point_cloud_data[:,3] = 1.0
                calib_file = osp.join(calib_path, file_base_name) + ".txt"
                semantic_file = osp.join(semantic_score_path, file_base_name+".png") + ".npy"
                semantic_data = np.load(semantic_file)
                fp = open(calib_file,"r")
                calib_data = fp.readlines()

                P, Tr_velo_to_cam, R_cam_to_rect = load_kitti_calib_data(calib_data)

                lidr_2_img = P @ R_cam_to_rect @ Tr_velo_to_cam

                pts_2d = point_cloud_data @ lidr_2_img.T

                pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
                pts_2d[:, 0] /= pts_2d[:, 2]
                pts_2d[:, 1] /= pts_2d[:, 2]

                pts_2d[:, 0] = np.clip(pts_2d[:, 0], a_min=min_x, a_max=semantic_data.shape[2]-1 - min_x)
                pts_2d[:, 1] = np.clip(pts_2d[:, 1], a_min=min_y, a_max=semantic_data.shape[1]-1 - min_y)

                selected_semantic_data = np.transpose(semantic_data[:,pts_2d[:,1].squeeze().astype('int32'),pts_2d[:,0].squeeze().astype('int32')]) # Nx5

                point_cloud_data = np.concatenate((point_cloud_data[:,:3], reflection_data, selected_semantic_data), axis=1)

                output_file = open(osp.join(voldyne_reduced_painted_path,filename), 'wb')
                point_cloud_data.tofile(output_file)
                output_file.close()

                pt_cls = np.argmax(selected_semantic_data, axis=1)

                pt_img = np.zeros((semantic_data.shape[1],semantic_data.shape[2],3))

                color_map = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0]])

                out_img_name = osp.join(voldyne_reduced_painted_path,file_base_name) + ".png"

                for cls,y,x in zip(pt_cls,pts_2d[:, 1],pts_2d[:, 0]):
                    if cls != (selected_semantic_data.shape[-1]-1):
                        if cls < color_map.shape[0]:
                            color = color_map[cls]
                        else:
                            color = color_map[cls]

                        pt_img[int(y)-min_y:int(y)+min_y,int(x)-min_x:int(x)+min_x,:] = color #np.expand_dims(np.expand_dims(color,axis=1),axis=2)

                cv2.imwrite(out_img_name, pt_img)
            else:
                continue

def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10,
                       enable_bevdet=False,
                       enable_petrv2=False,
                       enable_strpetr=False):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps,
        enable_bevdet=enable_bevdet, enable_strpetr=enable_strpetr)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path,
                         enable_bevdet=enable_bevdet, enable_petrv2=enable_petrv2,
                         enable_strpetr=enable_strpetr)

        if export_2d_anno is True:
            nuscenes_converter.export_2d_annotation(
                root_path, info_test_path, version=version)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path,
                     enable_bevdet=enable_bevdet, enable_petrv2=enable_petrv2, 
                     enable_strpetr=enable_strpetr)
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path,
                     enable_bevdet=enable_bevdet, enable_petrv2=enable_petrv2,
                     enable_strpetr=enable_strpetr)

    if export_2d_anno is True:
        nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    if version == 'v1.01-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_test_path)
    elif version == 'v1.01-train':
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_train_path)
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_val_path)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_test_path)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
    for split in splits:
        filename = osp.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
        update_pkl_infos('s3dis', out_dir=out_dir, pkl_path=filename)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_val_path)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=10,
                    only_gt_database=False,
                    save_senor_data=False,
                    skip_cam_instances_infos=False):
    """Prepare waymo dataset. There are 3 steps as follows:

    Step 1. Extract camera images and lidar point clouds from waymo raw
        data in '*.tfreord' and save as kitti format.
    Step 2. Generate waymo train/val/test infos and save as pickle file.
    Step 3. Generate waymo ground truth database (point clouds within
        each 3D bounding box) for data augmentation in training.
    Steps 1 and 2 will be done in Waymo2KITTI, and step 3 will be done in
    GTDatabaseCreater.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default to 10. Here we store ego2global information of these
            frames for later use.
        only_gt_database (bool, optional): Whether to only generate ground
            truth database. Default to False.
        save_senor_data (bool, optional): Whether to skip saving
            image and lidar. Default to False.
        skip_cam_instances_infos (bool, optional): Whether to skip
            gathering cam_instances infos in Step 2. Default to False.
    """
    from tools.dataset_converters import waymo_converter as waymo

    if version == 'v1.4':
        splits = [
            'training', 'validation', 'testing',
            'testing_3d_camera_only_detection'
        ]
    elif version == 'v1.4-mini':
        splits = ['training', 'validation']
    else:
        raise NotImplementedError(f'Unsupported Waymo version {version}!')
    out_dir = osp.join(out_dir, 'kitti_format')

    if not only_gt_database:
        for i, split in enumerate(splits):
            load_dir = osp.join(root_path, 'waymo_format', split)
            if split == 'validation':
                save_dir = osp.join(out_dir, 'training')
            else:
                save_dir = osp.join(out_dir, split)
            converter = waymo.Waymo2KITTI(
                load_dir,
                save_dir,
                prefix=str(i),
                workers=workers,
                test_mode=(split
                           in ['testing', 'testing_3d_camera_only_detection']),
                info_prefix=info_prefix,
                max_sweeps=max_sweeps,
                split=split,
                save_senor_data=save_senor_data,
                save_cam_instances=not skip_cam_instances_infos)
            converter.convert()
            if split == 'validation':
                converter.merge_trainval_infos()

        from tools.dataset_converters.waymo_converter import \
            create_ImageSets_img_ids
        create_ImageSets_img_ids(out_dir, splits)

    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()

    print_log('Successfully preparing Waymo Open Dataset')


def semantickitti_data_prep(info_prefix, out_dir):
    """Prepare the info file for SemanticKITTI dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
    """
    semantickitti_converter.create_semantickitti_info_file(
        info_prefix, out_dir)


def pandaset_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       enable_bevdet=False):
    pandaset_converter.create_pandaset_infos(root_path, info_prefix, version, dataset_name, out_dir, enable_bevdet)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='''Whether to only generate ground truth database.
        Only used when dataset is NuScenes or Waymo!''')
parser.add_argument(
    '--skip-cam_instances-infos',
    action='store_true',
    help='''Whether to skip gathering cam_instances infos.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--skip-saving-sensor-data',
    action='store_true',
    help='''Whether to skip saving image and lidar.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--bevdet',
    action='store_true',
    help='''Whether to add info needed for BEVDet in a pickle file''')
parser.add_argument(
    '--petrv2',
    action='store_true',
    help='''Whether to add info needed for PETRv2 in a pickle file''')
parser.add_argument(
    '--strpetr',
    action='store_true',
    help='''Whether to add info needed for StreamPETR in a pickle file''')

args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    if args.dataset == 'kitti':
        if args.only_gt_database:
            create_groundtruth_database(
                'KittiDataset',
                args.root_path,
                args.extra_tag,
                f'{args.extra_tag}_infos_train.pkl',
                relative_path=False,
                mask_anno_path='instances_train.json',
                with_mask=(args.version == 'mask'))
        else:
            if args.extra_tag == 'kitti_point_painting':
                kitti_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=args.version,
                    out_dir=args.out_dir,
                    with_plane=args.with_plane,
                    use_color=True)
            else:
                kitti_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=args.version,
                    out_dir=args.out_dir,
                    with_plane=args.with_plane)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            train_version = f'{args.version}-trainval'
            nuscenes_data_prep(
                root_path=args.root_path,
                can_bus_root_path=args.canbus,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                enable_bevdet=args.bevdet,
                enable_petrv2=args.petrv2,
                enable_strpetr=args.strpetr)
            test_version = f'{args.version}-test'
            nuscenes_data_prep(
                root_path=args.root_path,
                can_bus_root_path=args.canbus,
                info_prefix=args.extra_tag,
                version=test_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                enable_bevdet=args.bevdet,
                enable_petrv2=args.petrv2,
                enable_strpetr=args.strpetr)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            train_version = f'{args.version}'
            nuscenes_data_prep(
                root_path=args.root_path,
                can_bus_root_path=args.canbus,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                enable_bevdet=args.bevdet,
                enable_petrv2=args.petrv2,
                enable_strpetr=args.strpetr)
    elif args.dataset == 'pandaset':
        pandaset_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            dataset_name='PandasetDataset',
            out_dir=args.out_dir,
            enable_bevdet=args.bevdet)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps,
            only_gt_database=args.only_gt_database,
            save_senor_data=not args.skip_saving_sensor_data,
            skip_cam_instances_infos=args.skip_cam_instances_infos)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'semantickitti':
        semantickitti_data_prep(
            info_prefix=args.extra_tag, out_dir=args.out_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
