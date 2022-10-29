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
import json_tricks as json
import shutil
import tempfile
import numpy as np
from collections import OrderedDict, defaultdict
from colorama import Fore
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from edgeai_benchmark.utils import *
# from ..utils import *
from edgeai_benchmark.datasets.dataset_base import *
# from .dataset_base import *

__all__ = ['YCBV', '_get_mapping_id_name']

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

class YCBV(DatasetBase):
    def __init__(self, num_joints=17, download=False, num_frames=None, name="cocokpts", **kwargs):
        super().__init__(num_joints=num_joints, num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'kwargs must have path and split'
        path = self.kwargs['path']
        split = self.kwargs['split']
        if download:
            self.download(path, split)
        #
        self.class_to_name = {
               0: "002_master_chef_can" , 1: "003_cracker_box" ,  2: "004_sugar_box" , 3: "005_tomato_soup_can",  4: "006_mustard_bottle",
                5: "007_tuna_fish_can",  6: "008_pudding_box" , 7: "009_gelatin_box", 8: "010_potted_meat_can",  9: "011_banana",
                10: "019_pitcher_base", 11: "021_bleach_cleanser",  12: "024_bowl", 13: "025_mug", 14: "035_power_drill",
                15: "036_wood_block", 16: "037_scissors", 17: "040_large_marker", 18: "051_large_clamp", 19: "052_extra_large_clamp",
                20: "061_foam_brick"
                }
        dataset_folders = os.listdir(self.kwargs['path'])
        if 'annotations' not in dataset_folders:
            self.ycb2coco(path, split)  #convert ycb annotations to coco format
        annotations_dir = os.path.join(self.kwargs['path'], 'annotations')

        shuffle = self.kwargs.get('shuffle', False)
        image_split_dirs = os.listdir(self.kwargs['path'])
        assert self.kwargs['split'] in image_split_dirs, f'invalid path to coco dataset images/split {kwargs["split"]}'
        self.image_dir = os.path.join(self.kwargs['path'], self.kwargs['split'])

        self.coco_dataset = COCO(os.path.join(annotations_dir, 'instances_{}.json'.format(split)))

        # filter_imgs = self.kwargs['filter_imgs'] if 'filter_imgs' in self.kwargs else None
        # if isinstance(filter_imgs, str):
        #     # filter images with the given list
        #     filter_imgs = os.path.join(self.kwargs['path'], filter_imgs)
        #     with open(filter_imgs) as filter_fp:
        #         filter = [int(id) for id in list(filter_fp)]
        #         orig_keys = list(self.coco_dataset.imgs)
        #         orig_keys = [k for k in orig_keys if k in filter]
        #         self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in orig_keys}
        #     #
        # elif filter_imgs:
        #     all_keys = self.coco_dataset.getImgIds()
        #     sel_keys = []
        #     # filter and use images with gt having keypoints only.
        #     for img_id in all_keys:
        #         for ann in self.coco_dataset.imgToAnns[img_id]:
        #             if ann['num_keypoints'] >0 :
        #                 sel_keys.append(img_id)
        #                 break
        #
        #     self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in sel_keys}
        # #

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
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(self.cats, self.coco_dataset.getCatIds()))
        self.img_ids = self.coco_dataset.getImgIds()

        self.num_frames = self.kwargs['num_frames'] = num_frames
        self.tempfiles = []

        self.ann_info = {}
        self.id2name, self.name2id = _get_mapping_id_name(self.coco_dataset.imgs)


    def download(self, path, split):
        root = path
        images_folder = os.path.join(path, split)
        annotations_folder = os.path.join(path, 'annotations')
        if (not self.force_download) and os.path.exists(path) and \
                os.path.exists(images_folder) :
            print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            return
        #
        
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(f'{Fore.YELLOW}'
              f'\nPoseCNN:'
              f'\n   A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes, RSS 2018, '
              f'\n         Xiang et al. https://arxiv.org/abs/1711.00199\n'
              f'\n    Visit the following urls to know more about the YCBV dataset. '
              f'\n        https://rse-lab.cs.washington.edu/projects/posecnn/ '
              f'\n        https://bop.felk.cvut.cz/datasets/ '
              f'\n        https://www.ycbbenchmarks.com/ '
              f'{Fore.RESET}\n')

        dataset_url = 'https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_bop19.zip'  # test_split
        base_url = 'https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_base.zip'   # base url
        object_model_url = "https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip"   # object model url
        download_root = os.path.join(root, 'download')
        base_path = utils.download_file(base_url, root=download_root, extract_root=os.path.dirname(root))
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=root)
        object_model_path = utils.download_file(object_model_url, root=download_root, extract_root=root)
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def ycb2coco(self, root_path, split='test'):
        """
        Convert YCB annotation to COCO format
        """
        annotations_dir = os.path.join(root_path, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        basepath = os.path.join(root_path, split)
        data_folders = sorted(os.listdir(basepath))
        outfile = os.path.join(annotations_dir, 'instances_{}.json'.format(split))

        for data_folder_idx, data_folder in enumerate(data_folders):
            data_path = os.path.join(basepath, data_folder)
            annotations_gt = dict()
            for f in ('scene_gt_info.json', 'scene_gt.json'):
                path = os.path.join(data_path,  f)
                if os.path.exists(path):
                    print("Loading {}".format(path))
                    with open(path) as foo:
                        annotations_gt[f.split('.')[0]] = json.load(foo)

            if data_folder_idx==0:
                coco = dict()
                coco["images"] = []
                coco["type"] = "instance"
                coco["categories"] = []
                coco["annotations"] = []

                for obj_class in self.class_to_name:
                    category = dict([
                        ("supercategory", "object"),
                        ("id", obj_class),
                        ("name", self.class_to_name[obj_class])
                    ])

                    coco["categories"].append(category)
                obj_count = 0
                img_count = 0
            pbar = tqdm(enumerate(zip(list(annotations_gt['scene_gt'].items()), list(annotations_gt['scene_gt_info'].items()))), total=len(annotations_gt['scene_gt_info']))
            num_images = len(list(annotations_gt['scene_gt'].items()))
            for image_index, objects in pbar:
                objects_gt, objects_gt_info = objects[0], objects[1]
                filename = "{:06}".format(int(objects_gt[0])) + '.png'
                height, width = cv2.imread(data_path + '/rgb/' + filename).shape[:2]
                image = dict([
                    ("image_folder", data_folder),
                    ("id", img_count), #
                    ("file_name", filename),
                    ("height", height),
                    ("width", width),
                ])
                coco["images"].append(image)
                for object_gt, object_gt_info  in zip(objects_gt[1], objects_gt_info[1]):
                    if object_gt_info['visib_fract'] > 0:
                        annotation = dict([
                            ("image_id", img_count),
                            ("id", obj_count),
                            ("bbox", object_gt_info["bbox_obj"]),
                            ("area", object_gt_info["bbox_obj"][2] * object_gt_info["bbox_obj"][3]),
                            ("iscrowd", 0),
                            ("category_id", object_gt["obj_id"]-1),
                            ("R", object_gt["cam_R_m2c"]),
                            ("T", object_gt["cam_t_m2c"])
                        ])
                        obj_count += 1
                        coco["annotations"].append(annotation)
                img_count += 1
            pbar.close()
        json.dump(coco, outfile)



    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.coco_dataset.loadImgs([img_id])[0]
        image_path = os.path.join(self.image_dir,  img['image_folder'], 'rgb', img['file_name'])
        return image_path

    def __len__(self):
        return self.num_frames

    def __del__(self):
        for t in self.tempfiles:
            t.cleanup()
        #

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, preds, **kwargs):
        data_list = self.convert_to_coco_format(preds)

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
            json.dump(cat_results, f, sort_keys=True, indent=4)

        coco_det = self.coco_dataset.loadRes(res_file)
        coco_eval = COCOeval(self.coco_dataset, coco_det, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        accuracy = {'accuracy_ap[.5:.95]%': coco_eval.stats[0]*100.0, 'accuracy_ap50%': coco_eval.stats[1]*100.0}
        return accuracy


    def convert_to_coco_format(self, preds):
        data_list = []
        targets = self.coco_dataset.imgToAnns
        for frame_index, frame_targets in targets.items():
            pred = preds[frame_index]
            cls_pred = np.array(pred['cls'], dtype=np.int32)
            pred_gt_data = {}
            for target in frame_targets:
                pred_gt_data .update({
                    "image_id": target['image_id'],
                    "bbox_gt": target['bbox'],
                    "rotation_gt": target['R'],
                    "translation_gt": target['T'],
                    "category_id": target['category_id'],
                    "missing_det": True
                })
                if np.sum(cls_pred == target['category_id']) == 1:
                    matched_index =  np.where(cls_pred == target['category_id'])[0][0]
                    pred_gt_data.update(
                    {
                        "bbox_pred": pred['bbox'][matched_index],
                        "score": pred['scores'][matched_index],
                        "segmentation": [],
                        "rotation_pred" : pred['rotation'][matched_index],
                        "translation_pred": pred['translation'][matched_index],
                        "missing_det": False
                    })
            data_list.append(pred_gt_data)
        return data_list

    def evaluate_prediction_6dpose(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0
        data_dict_asym = []
        data_dict_sym = []
        for pred_data in data_dict:
            if not pred_data['missing_det']:
                if self.class_to_name[pred_data['category_id']] not in self.symmetric_objects.values():
                    data_dict_asym.extend([pred_data])
                else:
                    data_dict_sym.extend([pred_data])
        score_dict_asym = self.compute_add_score(data_dict_asym)
        score_dict_sym = self.compute_adds_score(data_dict_sym)
        score_dict = {}
        for metric in score_dict_asym.keys():
            score_dict[metric] = {**score_dict_asym[metric], **score_dict_sym[metric]}
            score_dict[metric + "_avg"] = np.mean(list(score_dict[metric].values()))

        score_dict_summmary = ""
        for metric in score_dict.keys():
            score_dict_summmary += metric + "\n"
            if "avg" not in metric:
                for cls in score_dict[metric].keys():
                    score_dict_summmary += "{:>5}".format(cls) + " ({:>12}) : ".format(self.class_to_name[cls]) + "{0:2f}".format(score_dict[metric][cls]) + "\n"
            else:
                score_dict_summmary += "{0:2f}".format(score_dict[metric]) + "\n"
        return score_dict, score_dict_summmary


    def compute_add_score(self, data_dict, percentage=0.1):
        distance_category = np.zeros((len(data_dict), 2))
        for index, pred_data in enumerate(data_dict):
            R_gt, t_gt = np.array(pred_data['rotation_gt']), np.array(pred_data['translation_gt'])
            R_gt, _ = cv2.Rodrigues(R_gt)
            R_pred, t_pred= np.array(pred_data['rotation_pred']), np.array(pred_data['translation_pred'])
            R_pred, _ = cv2.Rodrigues(R_pred)
            pts3d = self.class_to_model[pred_data['category_id']]
            #mean_distances = np.zeros((count,), dtype=np.float32)
            pts_xformed_gt = np.matmul(R_gt,  pts3d.transpose()) + t_gt[:, None]
            pts_xformed_pred = np.matmul(R_pred, pts3d.transpose()) + t_pred[:, None]
            distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
            distance_category[index, 0] = np.mean(distance)
            distance_category[index, 1] = pred_data['category_id']

        threshold = [self.class_to_diameter[category] * percentage for category in self.class_to_diameter.keys()]
        score_dict = {}
        for threshold_multiple in range(1, 6):
            score_dict["ADD_0p{}".format(threshold_multiple)] = {}
        for category_id, category_type in self.class_to_name.items():
            num_instances = len(distance_category[distance_category[:, 1] == category_id][:,0])
            if num_instances > 0:
                for threshold_multiple in range(1,6):
                    score = np.sum(distance_category[distance_category[:, 1] == category_id][:,0] < threshold_multiple * threshold[category_id]) / (num_instances + 1e-6)
                    score_dict["ADD_0p{}".format(threshold_multiple)].update({category_id: score})
        return score_dict


    def compute_adds_score(self, data_dict, percentage=0.1):
        distance_category = np.zeros((len(data_dict), 2))
        for index, pred_data in enumerate(data_dict):
            R_gt, t_gt = np.array(pred_data['rotation_gt']), np.array(pred_data['translation_gt'])
            R_gt, _ = cv2.Rodrigues(R_gt)
            R_pred, t_pred = np.array(pred_data['rotation_pred']), np.array(pred_data['translation_pred'])
            R_pred, _ = cv2.Rodrigues(R_pred)
            pts3d = self.class_to_model[pred_data['category_id']]
            pts_xformed_gt = np.matmul(R_gt, pts3d.transpose()) + t_gt[:, None]
            pts_xformed_pred = np.matmul(R_pred, pts3d.transpose()) + t_pred[:, None]
            kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
            distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
            # distance_np = np.sqrt(     #brute-force distance calculation
            #     np.min(np.sum((pts_xformed_gt[:, :, None] - pts_xformed_pred[:, None, :]) ** 2, axis=0), axis=1))
            distance_category[index, 0] = np.mean(distance)
            distance_category[index, 1] = pred_data['category_id']

        threshold = [self.class_to_diameter[category] * percentage for category in self.class_to_diameter.keys()]
        score_dict = {}
        for threshold_multiple in range(1, 6):
            score_dict["ADD_0p{}".format(threshold_multiple)] = {}

        for category_id, category_type in self.class_to_name.items():
            num_instances = len(distance_category[distance_category[:, 1] == category_id][:, 0])
            if num_instances > 0:
                for threshold_multiple in range(1,6):
                    score = np.sum(distance_category[distance_category[:, 1] == category_id][:,0] < threshold_multiple * threshold[category_id]) / (num_instances + 1e-6)
                    score_dict["ADD_0p{}".format(threshold_multiple)].update({category_id: score})
        return score_dict




################################################################################################
if __name__ == '__main__':
    # from inside the folder jacinto_ai_benchmark, run the following:
    # python3 -m edgeai_benchmark.datasets.coco_det
    # to create a converted dataset if you wish to load it using the dataset loader ImageDetection() in image_det.py
    # to load it using CocoSegmentation dataset in this file, this conversion is not required.
    import shutil
    output_folder = './dependencies/datasets/coco-det-converted'
    split = 'test'
    ycbv = YCBV(path='./dependencies/datasets/coco', split=split)
    num_frames = len(ycbv)

    images_output_folder = os.path.join(output_folder, split, 'images')
    labels_output_folder = os.path.join(output_folder, split, 'labels')
    # os.makedirs(images_output_folder)
    # os.makedirs(labels_output_folder)

    output_filelist = os.path.join(output_folder, f'{split}.txt')
    with open(output_filelist, 'w') as list_fp:
        for n in range(num_frames):
            image_path= ycbv.__getitem__(n)
            image_output_filename = os.path.join(images_output_folder, os.path.basename(image_path))
            shutil.copy2(image_path, image_output_filename)
            list_fp.write(f'images/{os.path.basename(image_output_filename)}\n')
        #
    #

