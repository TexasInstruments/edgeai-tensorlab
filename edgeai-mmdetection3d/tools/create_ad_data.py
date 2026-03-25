# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
import os
import numpy as np
import cv2
from mmengine import print_log

from tools.dataset_converters import nuscenes_ad_converter as nuscenes_ad_converter
from tools.dataset_converters.update_ad_infos_to_v2 import update_pkl_ad_infos


def nuscenes_ad_data_prep(root_path,
                          can_bus_root_path,
                          info_prefix,
                          version,
                          dataset_name,
                          out_dir,
                          max_sweeps=10,
                          enable_sparsedrive=False):
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
    nuscenes_ad_converter.create_nuscenes_ad_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps,
        enable_sparsedrive=enable_sparsedrive)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_ad_infos_test.pkl')
        update_pkl_ad_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path,
                            enable_sparsedrive=enable_sparsedrive)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_ad_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_ad_infos_val.pkl')
    update_pkl_ad_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path,
                        enable_sparsedrive=enable_sparsedrive)
    update_pkl_ad_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path,
                        enable_sparsedrive=enable_sparsedrive)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data/nuscenes',
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
    '--out-dir',
    type=str,
    default='./data/nuscenes',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--sparsedrive',
    action='store_true',
    help='''Whether to add info needed for SparseDrive in a pickle file''')

args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_ad_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            enable_sparsedrive=args.sparsedrive)
        test_version = f'{args.version}-test'
        nuscenes_ad_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            enable_sparsedrive=args.sparsedrive)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_ad_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            enable_sparsedrive=args.sparsedrive)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
