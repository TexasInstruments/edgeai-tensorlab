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

#################################################################################

# Also includes parts from: https://github.com/pytorch/vision
# License: License: https://github.com/pytorch/vision/blob/master/LICENSE

# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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

################################################################################

# Also includes parts from: https://github.com/cocodataset/cocoapi (pycocotools)
# License: https://github.com/cocodataset/cocoapi/blob/master/license.txt

# Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

"""
Reference:

Microsoft COCO: Common Objects in Context,
Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays,
Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár,
https://arxiv.org/abs/1405.0312, https://cocodataset.org/
"""


import numbers
import os
import random
import json
import json_tricks
import shutil
import tempfile
import numpy as np
from collections import OrderedDict, defaultdict
from colorama import Fore
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from ..utils import *
# from ..utils import *
from ..datasets.dataset_base import *
# from .dataset_base import *

__all__ = ['COCOKeypoints', '_get_mapping_id_name']

def _get_mapping_id_name(imgs):
    """
    Args:
        imgs (dict): dict of image info.

    Returns:
        tuple: Image name & id mapping dicts.

        - id2name (dict): Mapping image id to name.
        - name2id (dict): Mapping image name to id.
    """
    id2name = {}
    name2id = {}
    for image_id, image in imgs.items():
        file_name = image['file_name']
        id2name[image_id] = file_name
        name2id[file_name] = image_id

    return id2name, name2id

class COCOKeypoints(DatasetBase):
    def __init__(self, num_joints=17, download=False, num_frames=None, name="cocokpts", **kwargs):
        super().__init__(num_joints=num_joints, num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'kwargs must have path and split'
        path = self.kwargs['path']
        split = self.kwargs['split']
        if download:
            self.download(path, split)
        #

        dataset_folders = os.listdir(self.kwargs['path'])
        assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
        annotations_dir = os.path.join(self.kwargs['path'], 'annotations')

        shuffle = self.kwargs.get('shuffle', False)
        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(self.kwargs['path'], image_base_dir)
        image_split_dirs = os.listdir(image_base_dir)
        assert self.kwargs['split'] in image_split_dirs, f'invalid path to coco dataset images/split {kwargs["split"]}'
        self.image_dir = os.path.join(image_base_dir, self.kwargs['split'])

        self.annotation_file = os.path.join(annotations_dir, f'person_keypoints_{self.kwargs["split"]}.json')
        self.coco_dataset = COCO(self.annotation_file)

        filter_imgs = self.kwargs['filter_imgs'] if 'filter_imgs' in self.kwargs else None
        if isinstance(filter_imgs, str):
            # filter images with the given list
            filter_imgs = os.path.join(self.kwargs['path'], filter_imgs)
            with open(filter_imgs) as filter_fp:
                filter = [int(id) for id in list(filter_fp)]
                orig_keys = list(self.coco_dataset.imgs)
                orig_keys = [k for k in orig_keys if k in filter]
                self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in orig_keys}
            #
        elif filter_imgs:
            all_keys = self.coco_dataset.getImgIds()
            sel_keys = []
            # filter and use images with gt having keypoints only.
            for img_id in all_keys:
                for ann in self.coco_dataset.imgToAnns[img_id]:
                    if ann['num_keypoints'] >0 :
                        sel_keys.append(img_id)
                        break

            self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in sel_keys}
        #

        max_frames = len(self.coco_dataset.imgs)
        num_frames = self.kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        imgs_list = list(self.coco_dataset.imgs.items())
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(imgs_list)

        self.coco_dataset.imgs = {k:v for k,v in imgs_list[:num_frames]}
        
        self.cats = [
            cat['name'] for cat in self.coco_dataset.loadCats(self.coco_dataset.getCatIds())
        ]

        self.classes = ['__background__'] + self.cats
        self.kwargs['num_classes'] = self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(self.cats, self.coco_dataset.getCatIds()))
        self.img_ids = self.coco_dataset.getImgIds()       

        self.num_frames = self.kwargs['num_frames'] = num_frames
        self.tempfiles = []
        
        self.ann_info = {}
        self.ann_info['num_joints'] = num_joints
        self.ann_info['flip_index'] = [
            0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
        ]

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        # joint index starts from 1
        self.ann_info['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13],
                                        [12, 13], [6, 12], [7, 13], [6, 7],
                                        [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                                        [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],
                                        [5, 7]]

        self.sigmas = np.array([
                    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                    .87, .87, .89, .89
                ]) / 10.0

        self.id2name, self.name2id = _get_mapping_id_name(self.coco_dataset.imgs)
        # store dataset info
        with open(self.annotation_file) as afp:
            self.dataset_store = json.load(afp)
        #
        self.kwargs['dataset_info'] = self.get_dataset_info()

    def download(self, path, split):
        root = path
        images_folder = os.path.join(path, split)
        annotations_folder = os.path.join(path, 'annotations')
        if (not self.force_download) and os.path.exists(path) and \
                os.path.exists(images_folder) and os.path.exists(annotations_folder):
            print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            return
        #
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(f'{Fore.YELLOW}'
              f'\nCOCO Dataset:'
              f'\n    Microsoft COCO: Common Objects in Context, '
              f'\n        Tsung-Yi Lin, et.al. https://arxiv.org/abs/1405.0312\n'
              f'\n    Visit the following url to know more about the COCO dataset. '
              f'\n        https://cocodataset.org/ '
              f'{Fore.RESET}\n')

        dataset_url = 'http://images.cocodataset.org/zips/val2017.zip'
        extra_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        download_root = os.path.join(root, 'download')
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=root)
        extra_path = utils.download_file(extra_url, root=download_root, extract_root=root)
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.coco_dataset.loadImgs([img_id])[0]
        image_path = os.path.join(self.image_dir, img['file_name'])
        return image_path

    def __len__(self):
        return self.num_frames

    def __del__(self):
        for t in self.tempfiles:
            t.cleanup()
        #

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, outputs, **kwargs):
        # label_offset = kwargs.get('label_offset_pred', 0)
        #run_dir = kwargs.get('run_dir', None)
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        self.tempfiles.append(temp_dir_obj)

        keypoints = self._valid_kpts(outputs)

        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                        if not cls == '__background__']

        cat_id = data_pack[0]['cat_id']
        keypoints = data_pack[0]['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                                self.ann_info['num_joints'] * 3)

            for img_kpt, key_point in zip(img_kpts, key_points):
                kpt = key_point.reshape((self.ann_info['num_joints'], 3))
                #left_top = np.amin(kpt, axis=0)
                #right_bottom = np.amax(kpt, axis=0)

                #w = right_bottom[0] - left_top[0]
                #h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpt['image_id'],
                    'category_id': cat_id,
                    'keypoints': key_point.tolist(),
                    'score': img_kpt['score'],
                    #'bbox': [left_top[0], left_top[1], w, h]
                })
        
        res_file = os.path.join(kwargs['run_dir'], 'keypoint_results.json')
        with open(res_file, 'w') as f:
            json_tricks.dump(cat_results, f, sort_keys=True, indent=4)

        coco_det = self.coco_dataset.loadRes(res_file)
        coco_eval = COCOeval(self.coco_dataset, coco_det, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        accuracy = {'accuracy_ap[.5:.95]%': coco_eval.stats[0]*100.0, 'accuracy_ap50%': coco_eval.stats[1]*100.0}
        return accuracy

    def _valid_kpts(self, outputs):

        preds = []
        scores = []
        image_paths = []
        areas = []
        bboxes = []

        for output in outputs:
            preds.append(output['preds'])
            scores.append(output['scores'])
            image_paths.append(output['image_paths'][0])
            if 'area' in output.keys():
                areas.append(output['area'])
            if 'bbox' in output.keys():
                bboxes.append(output['bbox'])

        kpts = defaultdict(list)

        for idx, _preds in enumerate(preds):
            str_image_path = image_paths[idx]
            image_id = self.name2id[os.path.basename(str_image_path)]
            for idx_person, kpt in enumerate(_preds):

                kpts[image_id].append({
                    'keypoints': kpt[:, 0:3],
                    'score': scores[idx][idx_person],
                    'image_id': image_id,
                })

        valid_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            # if use_nms:
            #     nms = soft_oks_nms if self.soft_nms else oks_nms
            #     keep = nms(img_kpts, self.oks_thr, sigmas=self.sigmas)
            #     valid_kpts.append([img_kpts[_keep] for _keep in keep])
            # else:
            valid_kpts.append(img_kpts)

        return valid_kpts

    def get_dataset_info(self):
        if 'dataset_info' in self.kwargs:
            return self.kwargs['dataset_info']
        #
        # return only info and categories for now as the whole thing could be quite large.
        dataset_store = dict()
        for key in ('info', 'categories'):
            if key in self.dataset_store.keys():
                dataset_store.update({key: self.dataset_store[key]})
            #
        #
        dataset_store.update(dict(color_map=self.get_color_map()))        
        return dataset_store


################################################################################################
if __name__ == '__main__':
    # from inside the folder jacinto_ai_benchmark, run the following:
    # python3 -m edgeai_benchmark.datasets.coco_det
    # to create a converted dataset if you wish to load it using the dataset loader ImageDetection() in image_det.py
    # to load it using CocoSegmentation dataset in this file, this conversion is not required.
    import shutil
    output_folder = './dependencies/datasets/coco-det-converted'
    split = 'val2017'
    coco_keypoint = COCOKeypoints(path='./dependencies/datasets/coco', split=split)
    num_frames = len(coco_keypoint)

    images_output_folder = os.path.join(output_folder, split, 'images')
    labels_output_folder = os.path.join(output_folder, split, 'labels')
    # os.makedirs(images_output_folder)
    # os.makedirs(labels_output_folder)

    output_filelist = os.path.join(output_folder, f'{split}.txt')
    with open(output_filelist, 'w') as list_fp:
        for n in range(num_frames):
            image_path= coco_keypoint.__getitem__(n)
            image_output_filename = os.path.join(images_output_folder, os.path.basename(image_path))
            shutil.copy2(image_path, image_output_filename)
            list_fp.write(f'images/{os.path.basename(image_output_filename)}\n')
        #
    #
    
