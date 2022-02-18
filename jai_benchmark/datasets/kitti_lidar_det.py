# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import random
from colorama import Fore
from .. import utils
from .dataset_base import *
import numpy as np

class KittiLidar3D(DatasetBase):
    def __init__(self, download=False, read_anno=True, dest_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided in kwargs'
        assert 'num_classes' in self.kwargs, f'num_classes must be provided while creating {self.__class__.__name__}'

        path = self.kwargs['path']
        split_folder = self.kwargs['split']

        from tools.data_converter import kitti_converter as kitti

        # download the data if needed
        if download:
            assert False , 'Download is not supported for this dataset'

        assert os.path.exists(path) and os.path.isdir(path), \
            utils.log_color('\nERROR', 'dataset path is empty', path)

        # create list of images and classes
        self.imgs = utils.get_data_list(input= path + "/ImageSets/val.txt", dest_dir=dest_dir)
        self.num_frames = self.kwargs.get('num_frames',len(self.imgs))
        shuffle = self.kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #
        if read_anno == True:
            import pickle

            if os.path.exists(os.path.join(self.kwargs['path'],'kitti_infos_val.pkl')) == False:
                self.kitti_data_prep(root_path = self.kwargs['path'],
                                info_prefix='kitti',
                                version='v1.0',
                                out_dir=self.kwargs['path'] + '/kitti_temp')

            pipeline = [{'type':'LoadPointsFromFile', 'coord_type':'LIDAR', 'load_dim':4, 'use_dim':4},
                        {'type': 'MultiScaleFlipAug3D', 'img_scale': (1333, 800), 'pts_scale_ratio': 1, 'flip': False, 'transforms': [{'type': 'GlobalRotScaleTrans', 'rot_range': [0, 0], 'scale_ratio_range': [1.0, 1.0], 'translation_std': [0, 0, 0]}, {'type': 'RandomFlip3D'}, {'type': 'PointsRangeFilter', 'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1]}, {'type': 'DefaultFormatBundle3D', 'class_names': ['Car'], 'with_label': False}, {'type': 'Collect3D', 'keys': ['points']}]}
                        ]

            from mmdet3d.datasets import KittiDataset

            self.kitti_dataset = KittiDataset(self.kwargs['path'],os.path.join(self.kwargs['path'],'kitti_infos_val.pkl'),
                         self.kwargs['path'],
                         pipeline=pipeline,
                         pts_prefix='velodyne_reduced',
                         modality = {'use_lidar':True, 'use_camera':False},
                         classes=['Car'],
                         test_mode=True)
            assert self.num_frames <= len(self.kitti_dataset.data_infos), \
                'Number of frames is higher than length of dataset \n'

            self.kitti_dataset.data_infos = self.kitti_dataset.data_infos[:self.num_frames]

        self.num_classes = kwargs['num_classes']
        self.class_to_name = {
                                0: 'Car',
                                1: 'Pedestrian',
                                2: 'Cyclist',
                                3: 'Van',
                                4: 'Person_sitting',
                        }

    def kitti_data_prep(self, root_path, info_prefix, version, out_dir):
        """Prepare data related to Kitti dataset.

        Related data consists of '.pkl' files recording basic infos,
        2D annotations and groundtruth database.

        Args:
            root_path (str): Path of dataset root.
            info_prefix (str): The prefix of info filenames.
            version (str): Dataset version.
            out_dir (str): Output directory of the groundtruth database info.
        """
        kitti.create_kitti_info_file(root_path, info_prefix)
        kitti.create_reduced_point_cloud(root_path, info_prefix)

        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        info_trainval_path = osp.join(root_path,
                                    f'{info_prefix}_infos_trainval.pkl')
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        kitti.export_2d_annotation(root_path, info_train_path)
        kitti.export_2d_annotation(root_path, info_val_path)
        kitti.export_2d_annotation(root_path, info_trainval_path)
        kitti.export_2d_annotation(root_path, info_test_path)

        create_groundtruth_database(
            'KittiDataset',
            root_path,
            info_prefix,
            f'{out_dir}/{info_prefix}_infos_train.pkl',
            relative_path=False,
            mask_anno_path='instances_train.json',
            with_mask=(version == 'mask'))

    def download(self, path, split_file):
        return None

    def __getitem__(self, idx, **kwargs):

        with_label = kwargs.get('with_label', False)
        words = self.imgs[idx].split(' ')
        image_name = os.path.join(self.kwargs['path'],self.kwargs['split'],self.kwargs['pts_prefix']) + '/' + words[0] + '.bin'
        if with_label:
            assert len(words)>0, f'ground truth requested, but missing at the dataset entry for {words}'
            label = int(words[1])
            return image_name, label
        else:
            return image_name
        #

    def __len__(self):
        return self.num_frames

    def __call__(self, predictions, **kwargs):

        acc = self.kitti_dataset.evaluate(predictions, metric='mAP')
        ap_dict = {'KITTI/Car_3D_moderate_strict':acc}
        return ap_dict

    def load_kitti_3d_obj_annotation(self, anno_path, split_file):

        annos = []

        with open(split_file, 'r') as sf:
            in_files = sf.readlines()

        for file in in_files:
            annotations = {}
            annotations.update({
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': []
            })
            with open(os.path.join(anno_path, file.strip())+'.txt', 'r') as f:
                lines = f.readlines()
            # if len(lines) == 0 or len(lines[0]) < 15:
            #     content = []
            # else:
            content = [line.strip().split(' ') for line in lines]
            num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
            annotations['name'] = np.array([x[0] for x in content])
            num_gt = len(annotations['name'])
            annotations['truncated'] = np.array([float(x[1]) for x in content])
            annotations['occluded'] = np.array([int(x[2]) for x in content])
            annotations['alpha'] = np.array([float(x[3]) for x in content])
            annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                            for x in content]).reshape(-1, 4)
            # dimensions will convert hwl format to standard lhw(camera) format.
            annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                                for x in content
                                                ]).reshape(-1, 3)[:, [2, 0, 1]]
            annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                                for x in content]).reshape(-1, 3)
            annotations['rotation_y'] = np.array([float(x[14])
                                                for x in content]).reshape(-1)
            if len(content) != 0 and len(content[0]) == 16:  # have score
                annotations['score'] = np.array([float(x[15]) for x in content])
            else:
                annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)
            annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)

            annos.append(annotations)

        return annos

